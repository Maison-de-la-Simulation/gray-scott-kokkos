#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#endif

#include "helpers.hpp"
#include "output_writer.hpp"
#include "parameters.hpp"

// -----------------------------------------------------------------------------
// Data type
// -----------------------------------------------------------------------------

#if PRECISION == 64
using real = double;
#define MPI_REAL_TYPE MPI_DOUBLE
#elif PRECISION == 32
using real = float;
#define MPI_REAL_TYPE MPI_FLOAT
#else
#error "unknown precision"
#endif

namespace constants {
constexpr real kill_rate{0.054};
constexpr real feed_rate{0.014};
constexpr real dt{1.0};
constexpr real diffusion_rate_u{0.1};
constexpr real diffusion_rate_v{0.05};
}  // namespace constants

using View = Kokkos::View<real**, Kokkos::LayoutRight>;
using Range = Kokkos::pair<std::size_t, std::size_t>;

constexpr int KC_PROC_NULL = -1;

// -----------------------------------------------------------------------------
// Backend selection
// -----------------------------------------------------------------------------

#if defined(KOKKOSCOMM_ENABLE_NCCL)
using CommSpace = KokkosComm::Experimental::NcclSpace;
using ExecSpace = Kokkos::Cuda;
static_assert(
    std::is_same_v<ExecSpace, Kokkos::DefaultExecutionSpace>,
    "Kokkos::DefaultExecutionSpace must be Kokkos::Cuda when "
    "KOKKOSCOMM_ENABLE_NCCL is defined.");
#else
using CommSpace = KokkosComm::MpiSpace;
using ExecSpace = Kokkos::DefaultExecutionSpace;
#endif

using CommHandle = KokkosComm::Communicator<CommSpace, ExecSpace>;

enum Direction { NW = 0, N, NE, W, E, SW, S, SE };
constexpr int n_directions = 8;

// The direction a neighbor sees us in, e.g. our north neighbor sees us to its south.
static int opposite(int dir) {
    static constexpr int opposite_dir[n_directions] = {SE, S, SW, E, W, NE, N, NW};
    return opposite_dir[dir];
}

// -----------------------------------------------------------------------------
// Communicator resources (RAII)
//
// Owns the Cartesian MPI communicator (and the NCCL communicator, if any) so
// they are released on every exit path, and only after everything that uses
// them (the decomposition) has been destroyed.
// -----------------------------------------------------------------------------

static MPI_Comm create_cartesian_comm(std::size_t global_rows, std::size_t global_columns) {
    MPI_Comm cart_comm = MPI_COMM_NULL;

    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    if (global_rows % static_cast<std::size_t>(dims[0]) != 0 ||
        global_columns % static_cast<std::size_t>(dims[1]) != 0) {
        throw std::runtime_error(
            "Global field dimensions must be divisible by the MPI "
            "Cartesian decomposition.");
    }

    if (global_rows / static_cast<std::size_t>(dims[0]) == 0 ||
        global_columns / static_cast<std::size_t>(dims[1]) == 0) {
        throw std::runtime_error("Local field dimensions must be greater than zero.");
    }

    // Non-periodic: ranks at the global edge have no neighbor there.
    const int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    if (cart_comm == MPI_COMM_NULL) {
        throw std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL.");
    }

    return cart_comm;
}

#ifdef KOKKOSCOMM_ENABLE_NCCL
static ncclComm_t create_nccl_comm(MPI_Comm cart_comm) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);

    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, cart_comm);

    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);

    return comm;
}
#endif

struct CommResources {
    MPI_Comm cart = MPI_COMM_NULL;
#ifdef KOKKOSCOMM_ENABLE_NCCL
    ncclComm_t nccl = nullptr;
#endif

    CommResources(std::size_t global_rows, std::size_t global_columns) {
        cart = create_cartesian_comm(global_rows, global_columns);
#ifdef KOKKOSCOMM_ENABLE_NCCL
        nccl = create_nccl_comm(cart);
#endif
    }

    ~CommResources() {
#ifdef KOKKOSCOMM_ENABLE_NCCL
        if (nccl != nullptr) ncclCommDestroy(nccl);
#endif
        if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
    }

    CommResources(const CommResources&) = delete;
    CommResources& operator=(const CommResources&) = delete;
};

// -----------------------------------------------------------------------------
// Cartesian decomposition (MPI is only used as a topology helper;
// KokkosComm moves the data)
// -----------------------------------------------------------------------------

class CartesianDecomposition {
public:
    CartesianDecomposition() = delete;
    CartesianDecomposition(const CartesianDecomposition&) = delete;
    CartesianDecomposition& operator=(const CartesianDecomposition&) = delete;

    CartesianDecomposition(const MPI_Comm& cart_comm_, std::size_t global_rows_,
                           std::size_t global_columns_)
        : comm_(CommHandle::from_raw(cart_comm_, ExecSpace())) {
        init_topology(cart_comm_, global_rows_, global_columns_);
    }

#ifdef KOKKOSCOMM_ENABLE_NCCL
    CartesianDecomposition(const ncclComm_t& nccl_comm_, const MPI_Comm& cart_comm_,
                           std::size_t global_rows_, std::size_t global_columns_)
        : comm_(CommHandle::from_raw(nccl_comm_, ExecSpace())) {
        init_topology(cart_comm_, global_rows_, global_columns_);
    }
#endif

    ~CartesianDecomposition() = default;

    CommHandle& comm() { return comm_; }

    int rank() const { return rank_; }
    int size() const { return size_; }
    std::size_t global_rows() const { return global_rows_; }
    std::size_t global_columns() const { return global_columns_; }
    std::size_t local_rows() const { return local_rows_; }
    std::size_t local_columns() const { return local_columns_; }
    int dims(int d) const { return dims_[d]; }
    int coords(int d) const { return coords_[d]; }

    int neighbor(int dir) const { return neighbors_[dir]; }

private:
    void init_topology(const MPI_Comm& cart_comm_, std::size_t global_rows_in,
                       std::size_t global_columns_in) {
        MPI_Comm_rank(cart_comm_, &rank_);
        MPI_Comm_size(cart_comm_, &size_);
        MPI_Cart_get(cart_comm_, 2, dims_, periods_, coords_);

        // Rank of the process at offset (dr, dc), or KC_PROC_NULL if outside the grid.
        auto find_neighbor = [&](int dr, int dc) {
            const int c[2] = {coords_[0] + dr, coords_[1] + dc};

            if (c[0] < 0 || c[0] >= dims_[0] || c[1] < 0 || c[1] >= dims_[1]) {
                return KC_PROC_NULL;
            }

            int r = MPI_PROC_NULL;
            MPI_Cart_rank(cart_comm_, c, &r);
            return r;
        };

        neighbors_[NW] = find_neighbor(-1, -1);
        neighbors_[N]  = find_neighbor(-1,  0);
        neighbors_[NE] = find_neighbor(-1, +1);
        neighbors_[W]  = find_neighbor( 0, -1);
        neighbors_[E]  = find_neighbor( 0, +1);
        neighbors_[SW] = find_neighbor(+1, -1);
        neighbors_[S]  = find_neighbor(+1,  0);
        neighbors_[SE] = find_neighbor(+1, +1);

        global_rows_ = global_rows_in;
        global_columns_ = global_columns_in;
        local_rows_ = global_rows_ / static_cast<std::size_t>(dims_[0]);
        local_columns_ = global_columns_ / static_cast<std::size_t>(dims_[1]);
    }

    CommHandle comm_;
    int rank_ = 0;
    int size_ = 0;

    int dims_[2] = {0, 0};
    int coords_[2] = {0, 0};
    int periods_[2] = {0, 0};

    int neighbors_[n_directions] = {
        KC_PROC_NULL, KC_PROC_NULL, KC_PROC_NULL, KC_PROC_NULL,
        KC_PROC_NULL, KC_PROC_NULL, KC_PROC_NULL, KC_PROC_NULL};

    std::size_t global_rows_ = 0;
    std::size_t global_columns_ = 0;

    std::size_t local_rows_ = 0;
    std::size_t local_columns_ = 0;
};

// -----------------------------------------------------------------------------
// Communication buffers
//
// send[dir] / recv[dir] are subviews that alias the field itself, so there is
// no separate pack/unpack step: once a receive completes, the halo is already
// in place in the field.
//
// The field has one layer of halo cells around the interior:
//   interior rows: [1, nr-2]   interior columns: [1, nc-2]
//   halo rows: 0 and nr-1      halo columns: 0 and nc-1
//
// IMPORTANT: edge windows cover only the interior extent (not the halo
// corners). Using Kokkos::ALL() here would (a) send halo cells that are
// being overwritten by other receives at the same time (a data race), and
// (b) make N/S/W/E receives overwrite the corner halo cells that belong to
// the NW/NE/SW/SE receives.
// -----------------------------------------------------------------------------

using HaloView = Kokkos::View<real*, Kokkos::LayoutStride>;

struct CommBuffers {
    std::array<HaloView, n_directions> send;
    std::array<HaloView, n_directions> recv;

    CommBuffers() = default;

    explicit CommBuffers(const View& field) {
        const std::size_t nr = field.extent(0);
        const std::size_t nc = field.extent(1);

        // Corners: 1-element windows, kept rank-1 via a length-1 Range.
        send[NW] = Kokkos::subview(field, Range(1, 2), 1);
        recv[NW] = Kokkos::subview(field, Range(0, 1), 0);

        send[NE] = Kokkos::subview(field, Range(1, 2), nc - 2);
        recv[NE] = Kokkos::subview(field, Range(0, 1), nc - 1);

        send[SW] = Kokkos::subview(field, Range(nr - 2, nr - 1), 1);
        recv[SW] = Kokkos::subview(field, Range(nr - 1, nr), 0);

        send[SE] = Kokkos::subview(field, Range(nr - 2, nr - 1), nc - 2);
        recv[SE] = Kokkos::subview(field, Range(nr - 1, nr), nc - 1);

        // Edges: interior extent only.
        send[N] = Kokkos::subview(field, 1, Range(1, nc - 1));
        recv[N] = Kokkos::subview(field, 0, Range(1, nc - 1));

        send[S] = Kokkos::subview(field, nr - 2, Range(1, nc - 1));
        recv[S] = Kokkos::subview(field, nr - 1, Range(1, nc - 1));

        send[W] = Kokkos::subview(field, Range(1, nr - 1), 1);
        recv[W] = Kokkos::subview(field, Range(1, nr - 1), 0);

        send[E] = Kokkos::subview(field, Range(1, nr - 1), nc - 2);
        recv[E] = Kokkos::subview(field, Range(1, nr - 1), nc - 1);
    }
};

// -----------------------------------------------------------------------------
// Non-blocking halo exchange, split into begin/end
//
// begin_exchange() only posts the operations and returns; the caller then
// runs halo-independent work (compute_interior) while the transfers are in
// flight. end_exchange() is where wait_all() actually blocks.
//
// Note: no tags are passed, so correctness relies on (a) each neighbor rank
// appearing at most once per exchange (true for a non-periodic grid), and
// (b) the u and v exchanges being posted in the same order on every rank
// (MPI non-overtaking rule). Do not reorder them.
// -----------------------------------------------------------------------------

struct ExchangeHandle {
    std::vector<KokkosComm::Request<>> requests;
};

static ExchangeHandle begin_exchange(CartesianDecomposition& d, CommBuffers& b) {
    ExchangeHandle h;
    h.requests.reserve(2 * n_directions);

    // Post all receives first.
    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbor(dir) == KC_PROC_NULL) continue;
        h.requests.push_back(KokkosComm::recv(d.comm(), b.recv[dir], d.neighbor(dir)));
    }

    // Then all sends. Nothing is waited on here.
    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbor(dir) == KC_PROC_NULL) continue;
        h.requests.push_back(KokkosComm::send(d.comm(), b.send[dir], d.neighbor(dir)));
    }

    return h;
}

static void end_exchange(ExchangeHandle& h) {
    KokkosComm::wait_all(h.requests);
    h.requests.clear();
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

static void initialize(const View& u, const View& v,
                       const View& u_temp, const View& v_temp,
                       const CartesianDecomposition& d) {
    using Kokkos::subview;

    // Everything to 0 (this also gives the global physical boundary halos
    // their fixed value of 0), then interior of u/u_temp to 1.
    Kokkos::deep_copy(u, 0);
    Kokkos::deep_copy(v, 0);
    Kokkos::deep_copy(u_temp, 0);
    Kokkos::deep_copy(v_temp, 0);

    const std::size_t local_rows    = d.local_rows();
    const std::size_t local_columns = d.local_columns();

    const Range interior_rows(1, local_rows + 1);
    const Range interior_cols(1, local_columns + 1);

    Kokkos::deep_copy(subview(u,      interior_rows, interior_cols), 1);
    Kokkos::deep_copy(subview(u_temp, interior_rows, interior_cols), 1);

    // Global drop box [drop_i0, drop_i1) x [drop_j0, drop_j1).
    // Adjust `half` if you want a different size.
    const std::size_t half = 1;
    const std::size_t ci = d.global_rows() / 2;
    const std::size_t cj = d.global_columns() / 2;

    const std::size_t drop_i0 = ci > half ? ci - half : 0;
    const std::size_t drop_i1 = ci + half;  // exclusive
    const std::size_t drop_j0 = cj > half ? cj - half : 0;
    const std::size_t drop_j1 = cj + half;  // exclusive

    // This rank's owned global block [first, last).
    const std::size_t first_i = static_cast<std::size_t>(d.coords(0)) * local_rows;
    const std::size_t first_j = static_cast<std::size_t>(d.coords(1)) * local_columns;
    const std::size_t last_i  = first_i + local_rows;
    const std::size_t last_j  = first_j + local_columns;

    // Intersection of drop box and local block, in global coordinates.
    const std::size_t gi0 = std::max(drop_i0, first_i);
    const std::size_t gi1 = std::min(drop_i1, last_i);
    const std::size_t gj0 = std::max(drop_j0, first_j);
    const std::size_t gj1 = std::min(drop_j1, last_j);

    if (gi0 < gi1 && gj0 < gj1) {
        // Convert to local indices (+1 for the halo).
        const Range drop_rows(gi0 - first_i + 1, gi1 - first_i + 1);
        const Range drop_cols(gj0 - first_j + 1, gj1 - first_j + 1);

        Kokkos::deep_copy(subview(u, drop_rows, drop_cols), 0);
        Kokkos::deep_copy(subview(v, drop_rows, drop_cols), 1);
    }
}

// -----------------------------------------------------------------------------
// Compute: split into interior (halo-independent) and ring (halo-dependent).
// Both share the same stencil logic.
// -----------------------------------------------------------------------------

struct GrayScottUpdate {
    View u, v, u_temp, v_temp;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j) const {
        const real u_full =
            u(i - 1, j - 1) + u(i - 1, j) + u(i - 1, j + 1) +
            u(i, j - 1) - real(8) * u(i, j) + u(i, j + 1) +
            u(i + 1, j - 1) + u(i + 1, j) + u(i + 1, j + 1);

        const real v_full =
            v(i - 1, j - 1) + v(i - 1, j) + v(i - 1, j + 1) +
            v(i, j - 1) - real(8) * v(i, j) + v(i, j + 1) +
            v(i + 1, j - 1) + v(i + 1, j) + v(i + 1, j + 1);

        const real uvv = u(i, j) * v(i, j) * v(i, j);

        const real u_delta =
            constants::diffusion_rate_u * u_full - uvv +
            constants::feed_rate * (1 - u(i, j));

        const real v_delta =
            constants::diffusion_rate_v * v_full + uvv -
            (constants::feed_rate + constants::kill_rate) * v(i, j);

        u_temp(i, j) = u(i, j) + u_delta * constants::dt;
        v_temp(i, j) = v(i, j) + v_delta * constants::dt;
    }
};

// Cells whose 3x3 stencil does NOT touch the halo:
// local index range [2, local_rows-1] x [2, local_columns-1].
static void compute_interior(const View& u, const View& v,
                             const View& u_temp, const View& v_temp,
                             std::size_t local_rows, std::size_t local_columns) {
    if (local_rows < 3 || local_columns < 3) return;  // no safe interior

    GrayScottUpdate update{u, v, u_temp, v_temp};

    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>;
    Kokkos::parallel_for(
        "compute interior",
        Policy({2, 2}, {static_cast<int>(local_rows), static_cast<int>(local_columns)}),
        update);
}

// The one-cell-wide ring of interior cells adjacent to the halo, computed
// after end_exchange(). Corners are covered exactly once via the top/bottom
// rows; left/right columns exclude those two rows.
static void compute_ring(const View& u, const View& v,
                         const View& u_temp, const View& v_temp,
                         std::size_t local_rows, std::size_t local_columns) {
    GrayScottUpdate update{u, v, u_temp, v_temp};

    const int nr = static_cast<int>(local_rows);
    const int nc = static_cast<int>(local_columns);

    // Top row (i = 1) and bottom row (i = nr), full width incl. corners.
    // If nr == 1 both updates hit the same cell, so the second is skipped.
    Kokkos::parallel_for(
        "compute ring top/bottom",
        Kokkos::RangePolicy<int>(1, nc + 1),
        KOKKOS_LAMBDA(const int j) {
            update(1, j);
            if (nr > 1) update(nr, j);
        });

    // Left column (j = 1) and right column (j = nc), excluding the corners
    // already done above, i.e. i in [2, nr-1]. Only meaningful if nr > 2.
    if (nr > 2) {
        Kokkos::parallel_for(
            "compute ring left/right",
            Kokkos::RangePolicy<int>(2, nr),
            KOKKOS_LAMBDA(const int i) {
                update(i, 1);
                if (nc > 1) update(i, nc);
            });
    }
}

// -----------------------------------------------------------------------------
// Checksum
// -----------------------------------------------------------------------------

static real check_local(const View& field) {
    real checksum = 0;
    const auto nr = static_cast<std::int64_t>(field.extent(0));
    const auto nc = static_cast<std::int64_t>(field.extent(1));

    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<std::int64_t>>;
    Kokkos::parallel_reduce(
        "check field",
        Policy({1, 1}, {nr - 1, nc - 1}),
        KOKKOS_LAMBDA(const std::int64_t i, const std::int64_t j, real& sum) {
            sum += field(i, j);
        },
        checksum);

    return checksum;
}

static real check_global(const View& field, MPI_Comm comm) {
    const real local = check_local(field);
    real global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_REAL_TYPE, MPI_SUM, comm);
    return global;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

#if defined(KOKKOSCOMM_ENABLE_NCCL)
    {
        // Bind each rank to a distinct GPU on its node before Kokkos/NCCL
        // initialize, based on node-local rank.
        int local_rank = 0;
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
        cudaSetDevice(local_rank);
    }
#endif

    int exit_code = 0;

    try {
        // Kokkos must be finalized before MPI_Finalize; the try scope ensures that.
        Kokkos::ScopeGuard kokkos{argc, argv};

        Parameters parameters{argc, argv};
        parameters.check();
        if (world_rank == 0) {
            parameters.describe();
            parameters.show_size<real>(4);
        }

        const std::size_t global_rows = parameters.n_rows_ext - 2;
        const std::size_t global_columns = parameters.n_columns_ext - 2;

        if (global_rows < 1 || global_columns < 1) {
            throw std::runtime_error("Global field dimensions must be positive.");
        }

        // Declared before the decomposition so it is destroyed after it.
        CommResources resources(global_rows, global_columns);

#if defined(KOKKOSCOMM_ENABLE_NCCL)
        CartesianDecomposition decomposition(
            resources.nccl, resources.cart, global_rows, global_columns);
#else
        CartesianDecomposition decomposition(
            resources.cart, global_rows, global_columns);
#endif

        if (decomposition.rank() == 0) {
            std::cout << "Local domain per rank: " << decomposition.local_rows() << "x"
                      << decomposition.local_columns() << '\n';
            if (decomposition.local_rows() < 3 || decomposition.local_columns() < 3) {
                std::cerr << "Warning: local domain smaller than 3x3; compute_interior() "
                             "is a no-op and all work happens after the halo exchange.\n";
            }
        }

        const std::size_t ext_rows = decomposition.local_rows() + 2;
        const std::size_t ext_cols = decomposition.local_columns() + 2;

        View u("u", ext_rows, ext_cols);
        View v("v", ext_rows, ext_cols);
        View u_temp("u_temp", ext_rows, ext_cols);
        View v_temp("v_temp", ext_rows, ext_cols);

        initialize(u, v, u_temp, v_temp, decomposition);

        // CommBuffers hold subviews of the View they were built from. The
        // main loop swaps u <-> u_temp (a shallow swap), so the buffers are
        // rebuilt after each swap; otherwise they would keep referencing the
        // old field.
        CommBuffers u_buffers(u);
        CommBuffers v_buffers(v);

        for (std::size_t iteration = 1; iteration <= parameters.n_iterations; ++iteration) {
            // Kick off both halo exchanges without blocking. Keep the u/v
            // order identical on every rank (no tags are used).
            ExchangeHandle u_handle = begin_exchange(decomposition, u_buffers);
            ExchangeHandle v_handle = begin_exchange(decomposition, v_buffers);

            // Halo-independent work, overlapped with the in-flight transfers.
            compute_interior(u, v, u_temp, v_temp,
                             decomposition.local_rows(), decomposition.local_columns());

            // Block until the halos have arrived. recv[dir] aliases the
            // field's halo cells, so there is nothing to unpack.
            end_exchange(u_handle);
            end_exchange(v_handle);

            Kokkos::fence();  // Halo data must be visible before the ring kernels read it.

            // Finish the cells that depend on the freshly received halo.
            compute_ring(u, v, u_temp, v_temp,
                         decomposition.local_rows(), decomposition.local_columns());

            Kokkos::fence();  // Make sure all computations are complete before swapping.

            std::swap(u, u_temp);
            std::swap(v, v_temp);

            u_buffers = CommBuffers(u);
            v_buffers = CommBuffers(v);
        }

        Kokkos::fence();

        const real u_checksum = check_global(u, resources.cart);
        const real v_checksum = check_global(v, resources.cart);

        if (decomposition.rank() == 0) {
            std::cout << "u checksum: " << u_checksum << '\n';
            std::cout << "v checksum: " << v_checksum << '\n';
        }

    } catch (const std::exception& e) {
        std::cerr << "Rank " << world_rank << ": " << e.what() << '\n';
        // A single failing rank would leave the others blocked in MPI calls.
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit_code = 1;
    } catch (...) {
        std::cerr << "Rank " << world_rank << ": unknown exception\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
