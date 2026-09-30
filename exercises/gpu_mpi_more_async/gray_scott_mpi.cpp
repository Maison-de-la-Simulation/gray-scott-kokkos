#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "helpers.hpp"
#include "output_writer.hpp"
#include "parameters.hpp"

// data type
#if PRECISION == 64
using real = double;
#elif PRECISION == 32
using real = float;
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

struct CartesianDecomposition {
    MPI_Comm comm = MPI_COMM_NULL;

    int rank = MPI_PROC_NULL;
    int size = 0;

    int dims[2] = {0, 0};
    int coords[2] = {0, 0};

    int neighbors[n_directions] = {
        MPI_PROC_NULL, MPI_PROC_NULL,
        MPI_PROC_NULL, MPI_PROC_NULL,
        MPI_PROC_NULL, MPI_PROC_NULL,
        MPI_PROC_NULL, MPI_PROC_NULL
    };

    std::size_t global_rows = 0;
    std::size_t global_columns = 0;

    std::size_t local_rows = 0;
    std::size_t local_columns = 0;

    explicit CartesianDecomposition(std::size_t global_rows_,
                                    std::size_t global_columns_)
        : global_rows(global_rows_),
          global_columns(global_columns_) {

        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int dims_tmp[2] = {0, 0};
        MPI_Dims_create(size, 2, dims_tmp);

        dims[0] = dims_tmp[0];  // rows / north-south
        dims[1] = dims_tmp[1];  // columns / west-east

        if (global_rows % static_cast<std::size_t>(dims[0]) != 0 ||
            global_columns % static_cast<std::size_t>(dims[1]) != 0) {
            throw std::runtime_error(
                "Global field dimensions must be divisible by the MPI "
                "Cartesian decomposition.");
        }

        local_rows =
            global_rows / static_cast<std::size_t>(dims[0]);

        local_columns =
            global_columns / static_cast<std::size_t>(dims[1]);

        if (local_rows == 0 || local_columns == 0) {
            throw std::runtime_error(
                "Local field dimensions must be greater than zero.");
        }

        const int periods[2] = {0, 0};

        MPI_Cart_create(
            MPI_COMM_WORLD,
            2,
            dims,
            periods,
            0,
            &comm);

        if (comm == MPI_COMM_NULL) {
            throw std::runtime_error(
                "MPI_Cart_create returned MPI_COMM_NULL.");
        }

        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 2, coords);

        auto neighbor = [&](int dr, int dc) {
            const int c[2] = {
                coords[0] + dr,
                coords[1] + dc
            };

            // Outside the global domain -> physical boundary.
            if (c[0] < 0 || c[0] >= dims[0] ||
                c[1] < 0 || c[1] >= dims[1]) {
                return MPI_PROC_NULL;
            }

            int r = MPI_PROC_NULL;
            MPI_Cart_rank(comm, c, &r);
            return r;
        };

        neighbors[NW] = neighbor(-1, -1);
        neighbors[N]  = neighbor(-1,  0);
        neighbors[NE] = neighbor(-1, +1);

        neighbors[W]  = neighbor( 0, -1);
        neighbors[E]  = neighbor( 0, +1);

        neighbors[SW] = neighbor(+1, -1);
        neighbors[S]  = neighbor(+1,  0);
        neighbors[SE] = neighbor(+1, +1);
    }

    ~CartesianDecomposition() {
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
        }
    }

    CartesianDecomposition(const CartesianDecomposition&) = delete;
    CartesianDecomposition& operator=(
        const CartesianDecomposition&) = delete;
};


// -----------------------------------------------------------------------------
// Communication buffers
// -----------------------------------------------------------------------------

struct CommBuffers {
        std::array<Kokkos::View<real*>, n_directions> send;
        std::array<Kokkos::View<real*>, n_directions> recv;

        CommBuffers(int rows, int columns) {
            const int lengths[n_directions] = {1, columns, 1, rows, rows, 1, columns, 1};
            for (int direction = 0; direction < n_directions; ++direction) {
                send[direction] = Kokkos::View<real*>("send", lengths[direction]);
                recv[direction] = Kokkos::View<real*>("recv", lengths[direction]);
            }
        }
    };


// -----------------------------------------------------------------------------
// MPI datatype
// -----------------------------------------------------------------------------

static MPI_Datatype mpi_real_type() {
    if constexpr (std::is_same_v<real, double>) {
        return MPI_DOUBLE;
    } else {
        return MPI_FLOAT;
    }
}


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
// Pack/unpack via subviews.
//
// Every direction is a 1-D slice of the field: contiguous for N/S (fixed
// row, ranged column, and LayoutRight makes a row contiguous), strided for
// W/E (fixed column, ranged row), and size-1 for the four corners either
// way. Holding these in a single LayoutStride view covers all cases, so
// pack/unpack are one deep_copy each instead of eight bespoke kernels.
// -----------------------------------------------------------------------------

using StridedRow = Kokkos::View<real*, Kokkos::LayoutStride>;

// Slice of the interior edge/corner that is sent out for a given direction.
static StridedRow pack_subview(int dir, const View& field,
                                std::size_t r, std::size_t c) {
    switch (dir) {
        case N:  return Kokkos::subview(field, 1, Kokkos::make_pair(std::size_t(1), c + 1));
        case S:  return Kokkos::subview(field, r, Kokkos::make_pair(std::size_t(1), c + 1));
        case W:  return Kokkos::subview(field, Kokkos::make_pair(std::size_t(1), r + 1), 1);
        case E:  return Kokkos::subview(field, Kokkos::make_pair(std::size_t(1), r + 1), c);
        case NW: return Kokkos::subview(field, 1, Kokkos::make_pair(std::size_t(1), std::size_t(2)));
        case NE: return Kokkos::subview(field, 1, Kokkos::make_pair(c, c + 1));
        case SW: return Kokkos::subview(field, r, Kokkos::make_pair(std::size_t(1), std::size_t(2)));
        case SE: return Kokkos::subview(field, r, Kokkos::make_pair(c, c + 1));
    }
    throw std::logic_error("pack_subview: invalid direction");
}

// Slice of the halo ring that a given direction's data is unpacked into.
static StridedRow halo_subview(int dir, View& field,
                                std::size_t r, std::size_t c) {
    switch (dir) {
        case N:  return Kokkos::subview(field, 0,     Kokkos::make_pair(std::size_t(1), c + 1));
        case S:  return Kokkos::subview(field, r + 1, Kokkos::make_pair(std::size_t(1), c + 1));
        case W:  return Kokkos::subview(field, Kokkos::make_pair(std::size_t(1), r + 1), 0);
        case E:  return Kokkos::subview(field, Kokkos::make_pair(std::size_t(1), r + 1), c + 1);
        case NW: return Kokkos::subview(field, 0,     Kokkos::make_pair(std::size_t(0), std::size_t(1)));
        case NE: return Kokkos::subview(field, 0,     Kokkos::make_pair(c + 1, c + 2));
        case SW: return Kokkos::subview(field, r + 1, Kokkos::make_pair(std::size_t(0), std::size_t(1)));
        case SE: return Kokkos::subview(field, r + 1, Kokkos::make_pair(c + 1, c + 2));
    }
    throw std::logic_error("halo_subview: invalid direction");
}

static void pack_direction(int dir, const View& field, CommBuffers& b) {
    const std::size_t r = field.extent(0) - 2;
    const std::size_t c = field.extent(1) - 2;
    Kokkos::deep_copy(b.send[dir], pack_subview(dir, field, r, c));
}

static void unpack_direction(int dir, View& field, const CommBuffers& b) {
    const std::size_t r = field.extent(0) - 2;
    const std::size_t c = field.extent(1) - 2;
    Kokkos::deep_copy(halo_subview(dir, field, r, c), b.recv[dir]);
}


// -----------------------------------------------------------------------------
// Non-blocking, pipelined halo exchange.
//
// Receives are posted up front for every direction. Sends are issued one
// direction at a time, immediately after that direction is packed, instead
// of packing everything before sending anything. Unpacking happens as each
// receive completes (MPI_Waitany), instead of waiting for every receive to
// land first. Waiting on the sends is deferred to end_sends(), which the
// caller invokes after doing useful work, since the sent data is no longer
// needed by this rank once it has been handed to MPI_Isend.
// -----------------------------------------------------------------------------

struct ExchangeHandle {
    std::array<MPI_Request, n_directions> recv_requests{};
    std::array<int, n_directions> recv_dir{};
    int n_recv = 0;

    std::array<MPI_Request, n_directions> send_requests{};
    int n_send = 0;
};

static ExchangeHandle begin_exchange(const View& field, const CartesianDecomposition& d,
                                      CommBuffers& b, int tag_base) {
    ExchangeHandle h;

    // Post all receives first so incoming messages always have somewhere
    // to land as soon as they arrive.
    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;

        h.recv_dir[h.n_recv] = dir;
        MPI_Irecv(b.recv[dir].data(),
                  static_cast<int>(b.recv[dir].size()),
                  mpi_real_type(),
                  d.neighbors[dir], tag_base + dir, d.comm,
                  &h.recv_requests[h.n_recv++]);
    }

    // Pack and send one direction at a time, so the Isend for a direction
    // fires as soon as that direction is packed rather than after every
    // direction has been packed.
    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;

        pack_direction(dir, field, b);

        // The Isend below reads b.send[dir] directly, so the copy into it
        // must have actually completed on the device first.
        Kokkos::fence("pack_direction fence");

        MPI_Isend(b.send[dir].data(),
                  static_cast<int>(b.send[dir].size()),
                  mpi_real_type(),
                  d.neighbors[dir], tag_base + opposite(dir), d.comm,
                  &h.send_requests[h.n_send++]);
    }

    return h;
}

// Waits for this exchange's sends to complete. Safe to call late (e.g.
// after this iteration's compute is done), since the sent data isn't
// touched again until the next iteration's pack overwrites the buffer.
static void end_sends(ExchangeHandle& h) {
    MPI_Waitall(h.n_send, h.send_requests.data(), MPI_STATUSES_IGNORE);
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

    const std::size_t local_rows = d.local_rows;
    const std::size_t local_columns = d.local_columns;

    // Global center.
    const std::size_t global_i_center = d.global_rows / 2;
    const std::size_t global_j_center = d.global_columns / 2;
    const std::size_t global_i_drop_first = global_i_center - 1;
    const std::size_t global_i_drop_last = global_i_center + 1;
    const std::size_t global_j_drop_first = global_j_center - 1;
    const std::size_t global_j_drop_last = global_j_center + 1;

    const std::size_t first_i =
        static_cast<std::size_t>(d.coords[0]) * local_rows;

    const std::size_t first_j =
        static_cast<std::size_t>(d.coords[1]) * local_columns;

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

// Cells whose 3x3 stencil does NOT touch row/col 0 or nr-1/nc-1 (the halo).
// i.e. local index range [2, local_rows-1] x [2, local_columns-1].
static void compute_interior(const View& u, const View& v,
                              const View& u_temp, const View& v_temp,
                              std::size_t local_rows, std::size_t local_columns) {
    if (local_rows < 3 || local_columns < 3) return;  // no safe interior

    GrayScottUpdate update{u, v, u_temp, v_temp};

    Kokkos::parallel_for(
        "compute interior",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {2, 2},
            {static_cast<int>(local_rows), static_cast<int>(local_columns)}),
        update);
}

// -----------------------------------------------------------------------------
// Ring, split into 8 pieces so each can start as soon as its own halo
// dependency has arrived, instead of waiting for every direction.
//
// Non-corner edge pieces need only their own direction's halo (e.g. the top
// edge's 3x3 stencil only reaches into the N halo, since it stays within
// columns [2, nc-1], which is covered entirely by the N buffer).
// Corner pieces need three: the two adjacent edges plus the diagonal corner
// buffer (e.g. cell (1,1) reads row 0 at columns 0 and 1, and column 0 at
// row 1 -> NW, N, and W).
// -----------------------------------------------------------------------------

enum RingPiece {
    RING_N = 0,
    RING_S,
    RING_W,
    RING_E,
    RING_NW,
    RING_NE,
    RING_SW,
    RING_SE,
    n_ring_pieces
};

// Up to 3 halo directions each piece depends on; -1 = unused slot.
static constexpr std::array<std::array<int, 3>, n_ring_pieces> ring_deps = {{
    {N, -1, -1},   // RING_N
    {S, -1, -1},   // RING_S
    {W, -1, -1},   // RING_W
    {E, -1, -1},   // RING_E
    {N, W, NW},    // RING_NW
    {N, E, NE},    // RING_NE
    {S, W, SW},    // RING_SW
    {S, E, SE},    // RING_SE
}};

static void compute_ring_piece(int piece, const View& u, const View& v,
                                const View& u_temp, const View& v_temp,
                                int nr, int nc) {
    GrayScottUpdate update{u, v, u_temp, v_temp};

    switch (piece) {
        case RING_N:
            if (nc > 2)
                Kokkos::parallel_for(
                    "ring N", Kokkos::RangePolicy<int>(2, nc),
                    KOKKOS_LAMBDA(const int j) { update(1, j); });
            break;

        case RING_S:
            if (nc > 2)
                Kokkos::parallel_for(
                    "ring S", Kokkos::RangePolicy<int>(2, nc),
                    KOKKOS_LAMBDA(const int j) { update(nr, j); });
            break;

        case RING_W:
            if (nr > 2)
                Kokkos::parallel_for(
                    "ring W", Kokkos::RangePolicy<int>(2, nr),
                    KOKKOS_LAMBDA(const int i) { update(i, 1); });
            break;

        case RING_E:
            if (nr > 2)
                Kokkos::parallel_for(
                    "ring E", Kokkos::RangePolicy<int>(2, nr),
                    KOKKOS_LAMBDA(const int i) { update(i, nc); });
            break;

        case RING_NW:
            Kokkos::parallel_for(
                "ring NW", Kokkos::RangePolicy<int>(0, 1),
                KOKKOS_LAMBDA(const int) { update(1, 1); });
            break;

        case RING_NE:
            Kokkos::parallel_for(
                "ring NE", Kokkos::RangePolicy<int>(0, 1),
                KOKKOS_LAMBDA(const int) { update(1, nc); });
            break;

        case RING_SW:
            Kokkos::parallel_for(
                "ring SW", Kokkos::RangePolicy<int>(0, 1),
                KOKKOS_LAMBDA(const int) { update(nr, 1); });
            break;

        case RING_SE:
            Kokkos::parallel_for(
                "ring SE", Kokkos::RangePolicy<int>(0, 1),
                KOKKOS_LAMBDA(const int) { update(nr, nc); });
            break;
    }
}

// Tracks which ring pieces have already been launched this iteration, so
// try_launch_ready_pieces() can be called repeatedly (once per arriving
// message) without relaunching a piece twice.
struct PendingRing {
    bool launched[n_ring_pieces] = {};
};

static bool ring_deps_satisfied(int piece,
                                 const bool arrived_u[n_directions],
                                 const bool arrived_v[n_directions]) {
    for (int dep : ring_deps[piece]) {
        if (dep < 0) continue;
        if (!arrived_u[dep] || !arrived_v[dep]) return false;
    }
    return true;
}

// Launches every not-yet-launched ring piece whose dependencies are now
// fully satisfied for both u and v.
static void try_launch_ready_pieces(PendingRing& pending,
                                     const bool arrived_u[n_directions],
                                     const bool arrived_v[n_directions],
                                     const View& u, const View& v,
                                     const View& u_temp, const View& v_temp,
                                     int nr, int nc) {
    for (int piece = 0; piece < n_ring_pieces; ++piece) {
        if (pending.launched[piece]) continue;

        if (ring_deps_satisfied(piece, arrived_u, arrived_v)) {
            compute_ring_piece(piece, u, v, u_temp, v_temp, nr, nc);
            pending.launched[piece] = true;
        }
    }
}

// Drives both fields' receives to completion, unpacking each halo direction
// (for whichever field it belongs to) as soon as it lands, and launching
// each ring piece the moment both its u and v dependencies have arrived —
// rather than waiting for all of u's halo and all of v's halo first.
static void end_recvs_and_ring(const CartesianDecomposition& d,
                                View& u, const CommBuffers& u_buffers, ExchangeHandle& u_handle,
                                View& v, const CommBuffers& v_buffers, ExchangeHandle& v_handle,
                                const View& u_temp, const View& v_temp) {
    bool arrived_u[n_directions];
    bool arrived_v[n_directions];

    for (int dir = 0; dir < n_directions; ++dir) {
        // A direction with no neighbor is a physical boundary: its halo
        // value was already fixed during initialize() and no message is
        // coming, so treat it as already "arrived".
        const bool boundary = (d.neighbors[dir] == MPI_PROC_NULL);
        arrived_u[dir] = boundary;
        arrived_v[dir] = boundary;
    }

    const int nr = static_cast<int>(d.local_rows);
    const int nc = static_cast<int>(d.local_columns);

    PendingRing pending;

    // Some pieces may depend only on physical boundaries (e.g. a corner
    // rank with no neighbors at all) and can run before any message
    // arrives.
    try_launch_ready_pieces(pending, arrived_u, arrived_v, u, v, u_temp, v_temp, nr, nc);

    // Merge both fields' outstanding receive requests into one array so we
    // can react to whichever completes next, regardless of which field it
    // belongs to.
    std::array<MPI_Request, 2 * n_directions> requests{};
    std::array<int, 2 * n_directions> req_is_v{};
    std::array<int, 2 * n_directions> req_dir{};
    int n_outstanding = 0;

    for (int k = 0; k < u_handle.n_recv; ++k) {
        requests[n_outstanding] = u_handle.recv_requests[k];
        req_is_v[n_outstanding] = 0;
        req_dir[n_outstanding] = u_handle.recv_dir[k];
        ++n_outstanding;
    }
    for (int k = 0; k < v_handle.n_recv; ++k) {
        requests[n_outstanding] = v_handle.recv_requests[k];
        req_is_v[n_outstanding] = 1;
        req_dir[n_outstanding] = v_handle.recv_dir[k];
        ++n_outstanding;
    }

    for (int done = 0; done < n_outstanding; ++done) {
        int index = MPI_UNDEFINED;
        MPI_Waitany(n_outstanding, requests.data(), &index, MPI_STATUS_IGNORE);

        const int dir = req_dir[index];

        if (req_is_v[index]) {
            unpack_direction(dir, v, v_buffers);
            arrived_v[dir] = true;
        } else {
            unpack_direction(dir, u, u_buffers);
            arrived_u[dir] = true;
        }

        try_launch_ready_pieces(pending, arrived_u, arrived_v, u, v, u_temp, v_temp, nr, nc);
    }
}


// -----------------------------------------------------------------------------
// Checksum
// -----------------------------------------------------------------------------

static real check_local(const View& field) {
    real checksum = 0;

    Kokkos::parallel_reduce(
        "check field",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {1, 1},
            {field.extent(0) - 1,
             field.extent(1) - 1}),
        KOKKOS_LAMBDA(
            const int i,
            const int j,
            real& sum) {

            sum += field(i, j);
        },
        checksum);

    return checksum;
}


static real check_global(
    const View& field,
    const CartesianDecomposition& d) {

    const real local = check_local(field);

    real global = 0;

    MPI_Allreduce(
        &local,
        &global,
        1,
        mpi_real_type(),
        MPI_SUM,
        d.comm);

    return global;
}


// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int exit_code = 0;

    try {
        Kokkos::ScopeGuard kokkos{argc, argv};

        Parameters parameters{argc, argv};

        parameters.check();
        parameters.describe();
        parameters.show_size<real>(4);

        // Existing Parameters stores the extended global dimensions.
        // Remove the two global halo cells to obtain the physical global
        // domain.
        const std::size_t global_rows =
            parameters.n_rows_ext - 2;

        const std::size_t global_columns =
            parameters.n_columns_ext - 2;

        if (global_rows < 1 || global_columns < 1) {
            throw std::runtime_error(
                "Global field dimensions must be positive.");
        }

        CartesianDecomposition decomposition(
            global_rows,
            global_columns);

        // Each rank owns:
        //
        //     local_rows x local_columns
        //
        // interior cells plus a one-cell halo on every side.
        View u(
            "u",
            decomposition.local_rows + 2,
            decomposition.local_columns + 2);

        View v(
            "v",
            decomposition.local_rows + 2,
            decomposition.local_columns + 2);

        View u_temp(
            "u_temp",
            decomposition.local_rows + 2,
            decomposition.local_columns + 2);

        View v_temp(
            "v_temp",
            decomposition.local_rows + 2,
            decomposition.local_columns + 2);

        initialize(
            u,
            v,
            u_temp,
            v_temp,
            decomposition);

        CommBuffers u_buffers(
            decomposition.local_rows,
            decomposition.local_columns);

        CommBuffers v_buffers(
            decomposition.local_rows,
            decomposition.local_columns);


        for (std::size_t iteration = 1;
            iteration <= parameters.n_iterations;
            ++iteration) {

            // Kick off both halo exchanges without blocking.
            ExchangeHandle u_handle = begin_exchange(u, decomposition, u_buffers, 100);
            ExchangeHandle v_handle = begin_exchange(v, decomposition, v_buffers, 200);

            compute_interior(u, v, u_temp, v_temp,
                            decomposition.local_rows, decomposition.local_columns);

            // Unpack each halo direction as soon as it arrives, and launch
            // each of the 8 ring pieces the moment its own dependencies
            // (u and v halo for that direction, plus adjacent directions
            // for corners) have arrived, rather than waiting for every
            // direction to land first.
            end_recvs_and_ring(decomposition,
                                u, u_buffers, u_handle,
                                v, v_buffers, v_handle,
                                u_temp, v_temp);

            // The sends are no longer needed by this rank once handed to
            // MPI, so waiting for them to finish is deferred until after
            // compute_ring instead of blocking compute on them.
            end_sends(u_handle);
            end_sends(v_handle);

            std::swap(u, u_temp);
            std::swap(v, v_temp);
        }

        Kokkos::fence();

        const real u_checksum =
            check_global(u, decomposition);

        const real v_checksum =
            check_global(v, decomposition);

        if (decomposition.rank == 0) {
            std::cout
                << "u checksum: "
                << u_checksum
                << '\n';

            std::cout
                << "v checksum: "
                << v_checksum
                << '\n';
        }

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