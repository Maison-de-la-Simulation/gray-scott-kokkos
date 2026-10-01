#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "helpers.hpp"
#include "parameters.hpp"

// -----------------------------------------------------------------------------
// Data type
// -----------------------------------------------------------------------------

#if PRECISION == 64
using real = double;
#elif PRECISION == 32
using real = float;
#else
#error "unknown precision"
#endif

static MPI_Datatype mpi_real_type() {
    if constexpr (std::is_same_v<real, double>) {
        return MPI_DOUBLE;
    } else {
        return MPI_FLOAT;
    }
}

namespace constants {
constexpr real kill_rate{0.054};
constexpr real feed_rate{0.014};
constexpr real dt{1.0};
constexpr real diffusion_rate_u{0.1};
constexpr real diffusion_rate_v{0.05};
}  // namespace constants

using View = Kokkos::View<real**, Kokkos::LayoutRight>;
using StridedRow = Kokkos::View<real*, Kokkos::LayoutStride>;
using Range = Kokkos::pair<std::size_t, std::size_t>;

// -----------------------------------------------------------------------------
// Directions
// -----------------------------------------------------------------------------

enum Direction { NW = 0, N, NE, W, E, SW, S, SE };
constexpr int n_directions = 8;

// The direction a neighbor sees us in, e.g. our north neighbor sees us to its south.
static int opposite(int dir) {
    static constexpr int opposite_dir[n_directions] = {SE, S, SW, E, W, NE, N, NW};
    return opposite_dir[dir];
}

// -----------------------------------------------------------------------------
// Cartesian decomposition
// -----------------------------------------------------------------------------

struct CartesianDecomposition {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = MPI_PROC_NULL;
    int size = 0;
    int dims[2] = {0, 0};
    int coords[2] = {0, 0};
    int neighbors[n_directions];

    std::size_t global_rows = 0, global_columns = 0;
    std::size_t local_rows = 0, local_columns = 0;

    CartesianDecomposition(std::size_t global_rows_, std::size_t global_columns_)
        : global_rows(global_rows_), global_columns(global_columns_) {

        std::fill(std::begin(neighbors), std::end(neighbors), MPI_PROC_NULL);

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Dims_create(size, 2, dims);  // dims[0]: rows (north-south), dims[1]: columns (west-east)

        if (global_rows % static_cast<std::size_t>(dims[0]) != 0 ||
            global_columns % static_cast<std::size_t>(dims[1]) != 0) {
            throw std::runtime_error(
                "Global field dimensions must be divisible by the MPI "
                "Cartesian decomposition.");
        }

        local_rows = global_rows / static_cast<std::size_t>(dims[0]);
        local_columns = global_columns / static_cast<std::size_t>(dims[1]);

        // The ring/interior split and its halo dependencies assume at least
        // two cells per direction.
        if (local_rows < 2 || local_columns < 2) {
            throw std::runtime_error("Local field dimensions must be at least 2x2.");
        }

        // Non-periodic: ranks at the global edge have MPI_PROC_NULL neighbors.
        const int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
        if (comm == MPI_COMM_NULL) {
            throw std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL.");
        }

        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 2, coords);

        // Rank of the process at offset (dr, dc), or MPI_PROC_NULL if outside the grid.
        auto neighbor = [&](int dr, int dc) -> int {
            const int c[2] = {coords[0] + dr, coords[1] + dc};
            if (c[0] < 0 || c[0] >= dims[0] || c[1] < 0 || c[1] >= dims[1]) {
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
        if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    }

    CartesianDecomposition(const CartesianDecomposition&) = delete;
    CartesianDecomposition& operator=(const CartesianDecomposition&) = delete;
};

// -----------------------------------------------------------------------------
// Communication buffers
// -----------------------------------------------------------------------------

struct CommBuffers {
    std::array<Kokkos::View<real*>, n_directions> send;
    std::array<Kokkos::View<real*>, n_directions> recv;

    CommBuffers(std::size_t rows, std::size_t columns) {
        // Edges (N, S, W, E) carry a full row/column, corners carry one value.
        const std::size_t lengths[n_directions] = {1, columns, 1, rows, rows, 1, columns, 1};
        for (int direction = 0; direction < n_directions; ++direction) {
            send[direction] = Kokkos::View<real*>("send", lengths[direction]);
            recv[direction] = Kokkos::View<real*>("recv", lengths[direction]);
        }
    }
};

// -----------------------------------------------------------------------------
// Pack/unpack via subviews
//
// The field has one layer of halo cells around the interior:
//   interior rows: [1, r]   interior columns: [1, c]
//   halo rows:     0, r+1   halo columns:     0, c+1
//
// Every direction is a 1-D slice of the field: contiguous for N/S (fixed row,
// LayoutRight), strided for W/E (fixed column), and size 1 for the corners.
// A single LayoutStride view type covers all cases, so pack/unpack are one
// deep_copy each.
// -----------------------------------------------------------------------------

// Interior cells adjacent to the boundary in `dir` (these get sent).
static StridedRow pack_subview(int dir, const View& field) {
    const std::size_t r = field.extent(0) - 2;
    const std::size_t c = field.extent(1) - 2;
    switch (dir) {
        case N:  return Kokkos::subview(field, 1, Range(1, c + 1));
        case S:  return Kokkos::subview(field, r, Range(1, c + 1));
        case W:  return Kokkos::subview(field, Range(1, r + 1), 1);
        case E:  return Kokkos::subview(field, Range(1, r + 1), c);
        case NW: return Kokkos::subview(field, 1, Range(1, 2));
        case NE: return Kokkos::subview(field, 1, Range(c, c + 1));
        case SW: return Kokkos::subview(field, r, Range(1, 2));
        case SE: return Kokkos::subview(field, r, Range(c, c + 1));
    }
    throw std::logic_error("pack_subview: invalid direction");
}

// Halo cells in direction `dir` (these get received into).
static StridedRow halo_subview(int dir, const View& field) {
    const std::size_t r = field.extent(0) - 2;
    const std::size_t c = field.extent(1) - 2;
    switch (dir) {
        case N:  return Kokkos::subview(field, 0,     Range(1, c + 1));
        case S:  return Kokkos::subview(field, r + 1, Range(1, c + 1));
        case W:  return Kokkos::subview(field, Range(1, r + 1), 0);
        case E:  return Kokkos::subview(field, Range(1, r + 1), c + 1);
        case NW: return Kokkos::subview(field, 0,     Range(0, 1));
        case NE: return Kokkos::subview(field, 0,     Range(c + 1, c + 2));
        case SW: return Kokkos::subview(field, r + 1, Range(0, 1));
        case SE: return Kokkos::subview(field, r + 1, Range(c + 1, c + 2));
    }
    throw std::logic_error("halo_subview: invalid direction");
}

static void pack_direction(int dir, const View& field, CommBuffers& b) {
    Kokkos::deep_copy(b.send[dir], pack_subview(dir, field));
}

static void unpack_direction(int dir, const View& field, const CommBuffers& b) {
    Kokkos::deep_copy(halo_subview(dir, field), b.recv[dir]);
}

// -----------------------------------------------------------------------------
// Non-blocking, pipelined halo exchange
//
// Receives are posted up front for every direction. Sends are issued one
// direction at a time, right after that direction is packed. Unpacking happens
// as each receive completes (see end_recvs_and_ring). Waiting on the sends is
// deferred to end_sends(), since the sent data is no longer needed by this rank
// once it has been handed to MPI_Isend.
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

    // Post all receives first so incoming messages always have somewhere to land.
    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;

        h.recv_dir[h.n_recv] = dir;
        MPI_Irecv(b.recv[dir].data(), static_cast<int>(b.recv[dir].size()),
                  mpi_real_type(), d.neighbors[dir], tag_base + dir, d.comm,
                  &h.recv_requests[h.n_recv++]);
    }

    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;

        pack_direction(dir, field, b);
    }

    Kokkos::fence();

    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;
        MPI_Isend(b.send[dir].data(), static_cast<int>(b.send[dir].size()),
                  mpi_real_type(), d.neighbors[dir], tag_base + opposite(dir), d.comm,
                  &h.send_requests[h.n_send++]);
    }

    return h;
}

// Safe to call late: the send buffers are not touched again until the next
// iteration's pack.
static void end_sends(ExchangeHandle& h) {
    MPI_Waitall(h.n_send, h.send_requests.data(), MPI_STATUSES_IGNORE);
}

// -----------------------------------------------------------------------------
// Stencil
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

        const real u_delta = constants::diffusion_rate_u * u_full - uvv +
                             constants::feed_rate * (1 - u(i, j));
        const real v_delta = constants::diffusion_rate_v * v_full + uvv -
                             (constants::feed_rate + constants::kill_rate) * v(i, j);

        u_temp(i, j) = u(i, j) + u_delta * constants::dt;
        v_temp(i, j) = v(i, j) + v_delta * constants::dt;
    }
};

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

static void initialize(const View& u, const View& v,
                       const View& u_temp, const View& v_temp,
                       const CartesianDecomposition& d) {
    using Kokkos::subview;

    // Everything to 0, then interior of u/u_temp to 1.
    Kokkos::deep_copy(u, 0);
    Kokkos::deep_copy(v, 0);
    Kokkos::deep_copy(u_temp, 0);
    Kokkos::deep_copy(v_temp, 0);

    const std::size_t local_rows    = d.local_rows;
    const std::size_t local_columns = d.local_columns;

    const Range interior_rows(1, local_rows + 1);
    const Range interior_cols(1, local_columns + 1);

    Kokkos::deep_copy(subview(u,      interior_rows, interior_cols), 1);
    Kokkos::deep_copy(subview(u_temp, interior_rows, interior_cols), 1);

    // Global drop box [drop_i0, drop_i1) x [drop_j0, drop_j1).
    // Adjust `half` if you want a different size.
    const std::size_t half = 1;
    const std::size_t ci = d.global_rows / 2;
    const std::size_t cj = d.global_columns / 2;

    const std::size_t drop_i0 = ci > half ? ci - half : 0;
    const std::size_t drop_i1 = ci + half;  // exclusive
    const std::size_t drop_j0 = cj > half ? cj - half : 0;
    const std::size_t drop_j1 = cj + half;  // exclusive

    // This rank's owned global block [first, last).
    const std::size_t first_i = static_cast<std::size_t>(d.coords[0]) * local_rows;
    const std::size_t first_j = static_cast<std::size_t>(d.coords[1]) * local_columns;
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
// Compute, split into interior (halo-independent) and ring (halo-dependent)
// -----------------------------------------------------------------------------

// Cells whose 3x3 stencil does not touch the halo:
// local indices [2, local_rows - 1] x [2, local_columns - 1].
static void compute_interior(const View& u, const View& v,
                              const View& u_temp, const View& v_temp) {
    const int local_rows = u.extent(0) - 2;
    const int local_columns = u.extent(1) - 2;

    if (local_rows < 3 || local_columns < 3) return;  // no safe interior

    GrayScottUpdate update{u, v, u_temp, v_temp};

    Kokkos::parallel_for(
        "compute interior",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {2, 2},
            {static_cast<int>(local_rows), static_cast<int>(local_columns)}),
        update);
}

// Halo directions each ring piece depends on, indexed by Direction
// (NW, N, NE, W, E, SW, S, SE); -1 = unused slot.
static constexpr std::array<std::array<int, 3>, n_directions> ring_deps = {{
    {N, W, NW},   // NW
    {N, -1, -1},  // N
    {N, E, NE},   // NE
    {W, -1, -1},  // W
    {E, -1, -1},  // E
    {S, W, SW},   // SW
    {S, -1, -1},  // S
    {S, E, SE},   // SE
}};

// Small launch helpers, so the switch below stays readable.
static void update_row(const char* name, const GrayScottUpdate& update, int i, int j_begin, int j_end) {
    Kokkos::parallel_for(
        name, Kokkos::RangePolicy<int>(j_begin, j_end),
        KOKKOS_LAMBDA(const int j) { update(i, j); });
}

static void update_column(const char* name, const GrayScottUpdate& update, int j, int i_begin, int i_end) {
    Kokkos::parallel_for(
        name, Kokkos::RangePolicy<int>(i_begin, i_end),
        KOKKOS_LAMBDA(const int i) { update(i, j); });
}

static void update_cell(const char* name, const GrayScottUpdate& update, int i, int j) {
    Kokkos::parallel_for(
        name, Kokkos::RangePolicy<int>(0, 1),
        KOKKOS_LAMBDA(const int) { update(i, j); });
}

// nr, nc: number of interior rows / columns.
static void compute_ring_piece(int piece, const View& u, const View& v,
                               const View& u_temp, const View& v_temp) {
    const int nr = static_cast<int>(u.extent(0) - 2);
    const int nc = static_cast<int>(u.extent(1) - 2);
    const GrayScottUpdate update{u, v, u_temp, v_temp};

    switch (piece) {
        case N:  if (nc > 2) update_row("ring N", update, 1, 2, nc);      break;
        case S:  if (nc > 2) update_row("ring S", update, nr, 2, nc);     break;
        case W:  if (nr > 2) update_column("ring W", update, 1, 2, nr);   break;
        case E:  if (nr > 2) update_column("ring E", update, nc, 2, nr);  break;
        case NW: update_cell("ring NW", update, 1, 1);   break;
        case NE: update_cell("ring NE", update, 1, nc);  break;
        case SW: update_cell("ring SW", update, nr, 1);  break;
        case SE: update_cell("ring SE", update, nr, nc); break;
    }
}

// Tracks which ring pieces have been launched this iteration, so
// try_launch_ready_pieces() can be called after every message without
// launching a piece twice.
struct PendingRing {
    bool launched[n_directions] = {};
};

static bool ring_deps_satisfied(int piece, const bool arrived[2*n_directions]) {
    for (int dep : ring_deps[piece]) {
        if (dep < 0) continue;
        if (!arrived[dep] || !arrived[dep + n_directions]) return false;
    }
    return true;
}

// Launches every not-yet-launched ring piece whose u and v dependencies have
// both arrived.
static void try_launch_ready_pieces(PendingRing& pending,
                                    const bool arrived[2*n_directions],
                                    const View& u, const View& v,
                                    const View& u_temp, const View& v_temp) {
    const int nr = static_cast<int>(u.extent(0) - 2);
    const int nc = static_cast<int>(u.extent(1) - 2);
    for (int piece = 0; piece < n_directions; ++piece) {
        if (pending.launched[piece]) continue;

        if (ring_deps_satisfied(piece, arrived)) {
            compute_ring_piece(piece, u, v, u_temp, v_temp);
            pending.launched[piece] = true;
        }
    }
}

// Drives both fields' receives to completion. Each halo direction is unpacked
// as soon as it lands, and each ring piece is launched the moment both its u
// and v dependencies are there.
static void end_recvs_and_ring(const CartesianDecomposition& d,
                               const View& u, const CommBuffers& u_buffers, const ExchangeHandle& u_handle,
                               const View& v, const CommBuffers& v_buffers, const ExchangeHandle& v_handle,
                               const View& u_temp, const View& v_temp) {
    bool arrived[2*n_directions];

    for (int dir = 0; dir < n_directions; ++dir) {
        // No neighbor means a physical boundary: the halo was fixed in
        // initialize() and no message is coming, so it counts as arrived.
        const bool boundary = (d.neighbors[dir] == MPI_PROC_NULL);
        arrived[dir] = boundary;
        arrived[dir + n_directions] = boundary;
    }

    const int nr = static_cast<int>(d.local_rows);
    const int nc = static_cast<int>(d.local_columns);

    PendingRing pending;

    // Pieces that depend only on physical boundaries can run before any message.
    try_launch_ready_pieces(pending, arrived, u, v, u_temp, v_temp);

    // Merge both fields' receive requests into one array, so we can react to
    // whichever completes next.
    std::array<MPI_Request, 2 * n_directions> requests{};
    requests.fill(MPI_REQUEST_NULL);
    int n_outstanding = 0;

    for (int k = 0; k < u_handle.n_recv; ++k) {
        requests[n_outstanding] = u_handle.recv_requests[k];
        requests[u_handle.n_recv + n_outstanding] = v_handle.recv_requests[k];
        n_outstanding += 1;
    }

    const int n_requests = n_outstanding * 2;
    for (int done = 0; done < n_requests; ++done) {
        int index = MPI_UNDEFINED;
        MPI_Waitany(n_requests, requests.data(), &index, MPI_STATUS_IGNORE);
        if (index == MPI_UNDEFINED) break;

        const bool is_v = index >= u_handle.n_recv;
        const int request_index = is_v ? index - u_handle.n_recv : index;
        const int dir = is_v ? v_handle.recv_dir[request_index]
                             : u_handle.recv_dir[request_index];

        if (!is_v) {
            unpack_direction(dir, u, u_buffers);
            arrived[dir] = true;
        }else {
            unpack_direction(dir, v, v_buffers);
            arrived[dir + n_directions] = true;
        }
        try_launch_ready_pieces(pending, arrived, u, v, u_temp, v_temp);
    }
}

// -----------------------------------------------------------------------------
// Checksum
// -----------------------------------------------------------------------------

static real check_local(const View& field) {
    real checksum = 0;
    const auto nr = static_cast<std::int64_t>(field.extent(0));
    const auto nc = static_cast<std::int64_t>(field.extent(1));

    Kokkos::parallel_reduce(
        "check field",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nr - 1, nc - 1}),
        KOKKOS_LAMBDA(const int i, const int j, real& sum) { sum += field(i, j); },
        checksum);
    return checksum;
}

static real check_global(const View& field, const CartesianDecomposition& d) {
    const real local = check_local(field);
    real global = 0;
    MPI_Allreduce(&local, &global, 1, mpi_real_type(), MPI_SUM, d.comm);
    return global;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int exit_code = 0;

    try {
        // Scoped so that Kokkos is finalized before MPI_Finalize.
        Kokkos::ScopeGuard kokkos{argc, argv};

        Parameters parameters{argc, argv};
        parameters.check();
        if (world_rank == 0) {
            parameters.describe();
            parameters.show_size<real>(4);
        }

        // Parameters stores the extended dimensions; remove the two global
        // halo cells to get the physical domain.
        const std::size_t global_rows = parameters.n_rows_ext - 2;
        const std::size_t global_columns = parameters.n_columns_ext - 2;

        if (global_rows < 1 || global_columns < 1) {
            throw std::runtime_error("Global field dimensions must be positive.");
        }

        CartesianDecomposition decomposition(global_rows, global_columns);

        // Each rank owns local_rows x local_columns interior cells plus a
        // one-cell halo on every side.
        const std::size_t ext_rows = decomposition.local_rows + 2;
        const std::size_t ext_cols = decomposition.local_columns + 2;

        View u("u", ext_rows, ext_cols);
        View v("v", ext_rows, ext_cols);
        View u_temp("u_temp", ext_rows, ext_cols);
        View v_temp("v_temp", ext_rows, ext_cols);

        initialize(u, v, u_temp, v_temp, decomposition);

        CommBuffers u_buffers(decomposition.local_rows, decomposition.local_columns);
        CommBuffers v_buffers(decomposition.local_rows, decomposition.local_columns);

        for (std::size_t iteration = 1; iteration <= parameters.n_iterations; ++iteration) {
            // Kick off both halo exchanges without blocking.
            ExchangeHandle u_handle = begin_exchange(u, decomposition, u_buffers, 100);
            ExchangeHandle v_handle = begin_exchange(v, decomposition, v_buffers, 200);

            // Overlap: the interior needs no halo data.
            compute_interior(u, v, u_temp, v_temp);

            // Unpack each halo as it arrives and launch each ring piece as soon
            // as its dependencies are met.
            end_recvs_and_ring(decomposition,
                               u, u_buffers, u_handle,
                               v, v_buffers, v_handle,
                               u_temp, v_temp);

            // Sends are only waited on now, after the compute has been launched.
            end_sends(u_handle);
            end_sends(v_handle);

            Kokkos::fence();
            std::swap(u, u_temp);
            std::swap(v, v_temp);
        }

        Kokkos::fence();

        const real u_checksum = check_global(u, decomposition);
        const real v_checksum = check_global(v, decomposition);

        if (decomposition.rank == 0) {
            std::cout << "u checksum: " << u_checksum << '\n';
            std::cout << "v checksum: " << v_checksum << '\n';
        }

    } catch (const std::exception& e) {
        std::cerr << "Rank " << world_rank << ": " << e.what() << '\n';
        // A single failing rank would leave the others blocked in MPI calls.
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}