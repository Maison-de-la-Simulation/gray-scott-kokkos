#include <Kokkos_Core.hpp>
#include <mpi.h>

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

// data type
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

constexpr int KC_PROC_NULL = -1;

// -----------------------------------------------------------------------------
// Backend selection
// -----------------------------------------------------------------------------
//
// KokkosComm gives a single generic send/recv API regardless of transport.
// Which transport is actually used is decided here, at compile time, by
// picking CommSpace. Nothing below this block (exchange(), CommBuffers,
// compute(), ...) needs to know or care whether it is running over MPI or
// NCCL -- that is the point of going through KokkosComm rather than
// reimplementing an MPI-shaped API (Isend/Irecv/Waitall) on top of it.

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

enum Direction {
    NW = 0,
    N,
    NE,
    W,
    E,
    SW,
    S,
    SE
};

constexpr int n_directions = 8;

// -----------------------------------------------------------------------------
// Cartesian decomposition
// -----------------------------------------------------------------------------
//
// A Cartesian process grid (rank-to-coordinate mapping, neighbor discovery)
// is an MPI concept; NCCL has nothing equivalent. So regardless of which
// CommSpace we ultimately talk over, MPI is always used here purely as a
// *topology helper* to work out:
//   - dims / coords for this rank,
//   - the rank of each of the 8 neighbors.
//
// Once that is known, we build the actual KokkosComm communicator that the
// rest of the program uses for data movement:
//   - MPI backend: wrap the Cartesian communicator directly.
//   - NCCL backend: bootstrap a ncclUniqueId over the Cartesian communicator
//     (rank 0 creates it, MPI_Bcast distributes it), call ncclCommInitRank,
//     then hand the resulting ncclComm_t to KokkosComm. NCCL ranks are set
//     up to match the MPI Cartesian ranks 1:1, so `neighbor(dir)` (an MPI
//     rank) is valid as the peer argument to KokkosComm::send/recv under
//     either backend.

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

    const int periods[2] = {0, 0};

    MPI_Cart_create(
        MPI_COMM_WORLD,
        2,
        dims,
        periods,
        0,
        &cart_comm);

    if (cart_comm == MPI_COMM_NULL) {
        throw std::runtime_error(
            "MPI_Cart_create returned MPI_COMM_NULL.");
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

class CartesianDecomposition {
public:
    CartesianDecomposition() = delete;
    CartesianDecomposition(const CartesianDecomposition&) = delete;

    CartesianDecomposition(const MPI_Comm& cart_comm_, std::size_t global_rows_, std::size_t global_columns_)
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
    void init_topology(const MPI_Comm& cart_comm_, std::size_t global_rows_in, std::size_t global_columns_in) {
        MPI_Comm_rank(cart_comm_, &rank_);
        MPI_Comm_size(cart_comm_, &size_);
        MPI_Cart_get(cart_comm_, 2, dims_, periods_, coords_);

        auto find_neighbor = [&](int dr, int dc) {
            const int c[2] = {
                coords_[0] + dr,
                coords_[1] + dc
            };

            // Outside the global domain -> physical boundary.
            if (c[0] < 0 || c[0] >= dims_[0] ||
                c[1] < 0 || c[1] >= dims_[1]) {
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
        KC_PROC_NULL, KC_PROC_NULL,
        KC_PROC_NULL, KC_PROC_NULL,
        KC_PROC_NULL, KC_PROC_NULL,
        KC_PROC_NULL, KC_PROC_NULL
    };

    std::size_t global_rows_ = 0;
    std::size_t global_columns_ = 0;

    std::size_t local_rows_ = 0;
    std::size_t local_columns_ = 0;
};

// -----------------------------------------------------------------------------
// Communication buffers
// -----------------------------------------------------------------------------
//
// Corners are single-element halo cells. A 2-index subview of a rank-2 View
// collapses to rank-0, which cannot bind to a Kokkos::View<real*>, so each
// dimension must be sliced with a Kokkos::pair range to stay rank-1.

using HaloView = Kokkos::View<real*, Kokkos::LayoutStride>;

struct CommBuffers {
    std::array<HaloView, n_directions> send;
    std::array<HaloView, n_directions> recv;

    CommBuffers() = default;

    explicit CommBuffers(const View& field) {
        const std::size_t nr = field.extent(0);
        const std::size_t nc = field.extent(1);

        using Range = Kokkos::pair<std::size_t, std::size_t>;

        // Corners: 1x1 windows, kept rank-1 via a length-1 Range on one axis.
        send[NW] = Kokkos::subview(field, Range(1, 2), 1);
        recv[NW] = Kokkos::subview(field, Range(0, 1), 0);

        send[NE] = Kokkos::subview(field, Range(1, 2), nc - 2);
        recv[NE] = Kokkos::subview(field, Range(0, 1), nc - 1);

        send[SW] = Kokkos::subview(field, Range(nr - 2, nr - 1), 1);
        recv[SW] = Kokkos::subview(field, Range(nr - 1, nr), 0);

        send[SE] = Kokkos::subview(field, Range(nr - 2, nr - 1), nc - 2);
        recv[SE] = Kokkos::subview(field, Range(nr - 1, nr), nc - 1);

        // Edges.
        send[N] = Kokkos::subview(field, 1, Kokkos::ALL());
        recv[N] = Kokkos::subview(field, 0, Kokkos::ALL());

        send[S] = Kokkos::subview(field, nr - 2, Kokkos::ALL());
        recv[S] = Kokkos::subview(field, nr - 1, Kokkos::ALL());

        send[W] = Kokkos::subview(field, Kokkos::ALL(), 1);
        recv[W] = Kokkos::subview(field, Kokkos::ALL(), 0);

        send[E] = Kokkos::subview(field, Kokkos::ALL(), nc - 2);
        recv[E] = Kokkos::subview(field, Kokkos::ALL(), nc - 1);
    }
};

// -----------------------------------------------------------------------------
// Direction helper
// -----------------------------------------------------------------------------

static int opposite(int dir) {
    static constexpr int opposite_dir[n_directions] = {
        SE, S, SW,
        E, W,
        NE, N, NW
    };

    return opposite_dir[dir];
}

// -----------------------------------------------------------------------------
// Halo exchange
// -----------------------------------------------------------------------------
//
// Written entirely against KokkosComm's generic API (comm, send(), recv()),
// with no MPI-specific types or calls -- no MPI_Comm, no MPI_Datatype, no
// manual Isend/Irecv/Waitall bookkeeping. It is identical whether CommSpace
// is MpiSpace or NcclSpace: switching transport is the one-line CommSpace
// alias above, not a rewrite of this function.

static void exchange(CartesianDecomposition& d, CommBuffers& b) {
    std::vector<KokkosComm::Request<>> recv_requests;
    recv_requests.reserve(n_directions);

    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbor(dir) == KC_PROC_NULL) {
            continue;
        }
        recv_requests.push_back(
            KokkosComm::recv(d.comm(), b.recv[dir], d.neighbor(dir)));
    }

    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbor(dir) == KC_PROC_NULL) {
            continue;
        }
        KokkosComm::send(d.comm(), b.send[dir], d.neighbor(dir));
    }

    KokkosComm::wait_all(recv_requests);
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

static void initialize(
    const View& u,
    const View& v,
    const View& u_temp,
    const View& v_temp,
    const CartesianDecomposition& d) {

    Kokkos::deep_copy(u, 1);
    Kokkos::deep_copy(v, 0);

    Kokkos::deep_copy(u_temp, 1);
    Kokkos::deep_copy(v_temp, 0);

    const std::size_t local_rows = d.local_rows();
    const std::size_t local_columns = d.local_columns();

    // Global center.
    const std::size_t global_i_center = d.global_rows() / 2;
    const std::size_t global_j_center = d.global_columns() / 2;
    const std::size_t global_i_drop_first = global_i_center - 1;
    const std::size_t global_i_drop_last = global_i_center + 1;
    const std::size_t global_j_drop_first = global_j_center - 1;
    const std::size_t global_j_drop_last = global_j_center + 1;

    const std::size_t first_i =
        static_cast<std::size_t>(d.coords(0)) * local_rows;

    const std::size_t first_j =
        static_cast<std::size_t>(d.coords(1)) * local_columns;

    const std::size_t last_i = first_i + local_rows;
    const std::size_t last_j = first_j + local_columns;

    // Could be solved better with a single kernel, but this is easier.
    for (std::size_t gi = global_i_drop_first; gi < global_i_drop_last; ++gi) {
        for (std::size_t gj = global_j_drop_first; gj < global_j_drop_last; ++gj) {

            if (gi >= first_i && gi < last_i &&
                gj >= first_j && gj < last_j) {

                const std::size_t local_i = gi - first_i + 1;
                const std::size_t local_j = gj - first_j + 1;

                Kokkos::parallel_for(
                    "add drop",
                    Kokkos::RangePolicy<>(0, 1),
                    KOKKOS_LAMBDA(const int x) {
                        u(local_i, local_j) = 0;
                        v(local_i, local_j) = 1;
                    });
            }
        }
    }

    // Global physical boundary conditions for u and u_temp.

    Kokkos::parallel_for(
        "initialize vertical boundary",
        Kokkos::RangePolicy<>(0, local_rows + 2),
        KOKKOS_LAMBDA(const std::size_t i) {

            u(i, 0) = 0;
            u_temp(i, 0) = 0;

            u(i, local_columns + 1) = 0;
            u_temp(i, local_columns + 1) = 0;
        });

    Kokkos::parallel_for(
        "initialize horizontal boundary",
        Kokkos::RangePolicy<>(0, local_columns + 2),
        KOKKOS_LAMBDA(const std::size_t j) {

            u(0, j) = 0;
            u_temp(0, j) = 0;

            u(local_rows + 1, j) = 0;
            u_temp(local_rows + 1, j) = 0;
        });

    // The v fields were initialized to zero. Their global physical boundary
    // remains zero because compute() only modifies interior cells.
}

// -----------------------------------------------------------------------------
// Compute
// -----------------------------------------------------------------------------

static void compute(
    const View& u,
    const View& v,
    const View& u_temp,
    const View& v_temp) {

    const std::size_t nr = u.extent(0);
    const std::size_t nc = u.extent(1);

    Kokkos::parallel_for(
        "compute",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {1, 1},
            {static_cast<std::int64_t>(nr - 1), static_cast<std::int64_t>(nc - 1)}),
        KOKKOS_LAMBDA(const int i, const int j) {

            const real u_full =
                u(i - 1, j - 1) +
                u(i - 1, j) +
                u(i - 1, j + 1) +
                u(i, j - 1) -
                real(8) * u(i, j) +
                u(i, j + 1) +
                u(i + 1, j - 1) +
                u(i + 1, j) +
                u(i + 1, j + 1);

            const real v_full =
                v(i - 1, j - 1) +
                v(i - 1, j) +
                v(i - 1, j + 1) +
                v(i, j - 1) -
                real(8) * v(i, j) +
                v(i, j + 1) +
                v(i + 1, j - 1) +
                v(i + 1, j) +
                v(i + 1, j + 1);

            const real uvv =
                u(i, j) * v(i, j) * v(i, j);

            const real u_delta =
                constants::diffusion_rate_u * u_full -
                uvv +
                constants::feed_rate * (1 - u(i, j));

            const real v_delta =
                constants::diffusion_rate_v * v_full +
                uvv -
                (constants::feed_rate + constants::kill_rate) *
                    v(i, j);

            u_temp(i, j) =
                u(i, j) + u_delta * constants::dt;

            v_temp(i, j) =
                v(i, j) + v_delta * constants::dt;
        });
}

// -----------------------------------------------------------------------------
// Checksum
// -----------------------------------------------------------------------------
//
// Not on the performance path, so this stays plain MPI over MPI_COMM_WORLD
// rather than going through KokkosComm or the (possibly already-freed, in
// the NCCL case) cart_comm.

static real check_local(const View& field) {
    real checksum = 0;

    Kokkos::parallel_reduce(
        "check field",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {1, 1},
            {static_cast<std::int64_t>(field.extent(0) - 1),
             static_cast<std::int64_t>(field.extent(1) - 1)}),
        KOKKOS_LAMBDA(
            const int i,
            const int j,
            real& sum) {

            sum += field(i, j);
        },
        checksum);

    return checksum;
}

static real check_global(const View& field) {
    const real local = check_local(field);

    real global = 0;

    // Use the MPI type that matches `real` (float vs double) instead of
    // always MPI_DOUBLE, which would silently reinterpret the bytes when
    // PRECISION == 32.
    MPI_Allreduce(
        &local,
        &global,
        1,
        MPI_REAL_TYPE,
        MPI_SUM,
        MPI_COMM_WORLD);

    return global;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

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
        Kokkos::ScopeGuard kokkos{argc, argv};

        Parameters parameters{argc, argv};

        parameters.check();
        parameters.describe();
        parameters.show_size<real>(4);

        const std::size_t global_rows =
            parameters.n_rows_ext - 2;

        const std::size_t global_columns =
            parameters.n_columns_ext - 2;

        if (global_rows < 1 || global_columns < 1) {
            throw std::runtime_error(
                "Global field dimensions must be positive.");
        }

        MPI_Comm cart_comm = create_cartesian_comm(global_rows, global_columns);

#if defined(KOKKOSCOMM_ENABLE_NCCL)
        ncclComm_t nccl_comm = create_nccl_comm(cart_comm);
        CartesianDecomposition decomposition(
            nccl_comm, cart_comm, global_rows, global_columns);
#else
        CartesianDecomposition decomposition(
            cart_comm, global_rows, global_columns);
#endif
        
        // Print rank info for debugging.
        std::cout
            << "Rank "
            << decomposition.rank()
            << " of "
            << decomposition.size()
            << " (coords "
            << decomposition.coords(0)
            << ","
            << decomposition.coords(1)
            << ") has local domain "
            << decomposition.local_rows()
            << "x"
            << decomposition.local_columns()
            << '\n';

        View u(
            "u",
            decomposition.local_rows() + 2,
            decomposition.local_columns() + 2);

        View v(
            "v",
            decomposition.local_rows() + 2,
            decomposition.local_columns() + 2);

        View u_temp(
            "u_temp",
            decomposition.local_rows() + 2,
            decomposition.local_columns() + 2);

        View v_temp(
            "v_temp",
            decomposition.local_rows() + 2,
            decomposition.local_columns() + 2);

        initialize(
            u,
            v,
            u_temp,
            v_temp,
            decomposition);

        // CommBuffers hold subviews of whichever View instance they were
        // built from. The main loop swaps u <-> u_temp (a shallow pointer
        // swap, not a data copy), so the buffers are rebuilt each iteration
        // against the current u/v -- otherwise they would keep referencing
        // stale memory after the first swap.
        CommBuffers u_buffers(u);
        CommBuffers v_buffers(v);

        for (std::size_t iteration = 1;
             iteration <= parameters.n_iterations;
             ++iteration) {

            exchange(decomposition, u_buffers);
            exchange(decomposition, v_buffers);

            compute(
                u,
                v,
                u_temp,
                v_temp);

            std::swap(u, u_temp);
            std::swap(v, v_temp);

            u_buffers = CommBuffers(u);
            v_buffers = CommBuffers(v);
        }

        Kokkos::fence();

        const real u_checksum = check_global(u);
        const real v_checksum = check_global(v);

        if (decomposition.rank() == 0) {
            std::cout
                << "u checksum: "
                << u_checksum
                << '\n';

            std::cout
                << "v checksum: "
                << v_checksum
                << '\n';
        }

        MPI_Comm_free(&cart_comm);

    } catch (const std::exception& e) {

        int initialized = 0;
        MPI_Initialized(&initialized);

        if (initialized) {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            std::cerr
                << "Rank "
                << rank
                << ": "
                << e.what()
                << '\n';
        }

        exit_code = 1;
    }

    MPI_Finalize();

    return exit_code;
}
