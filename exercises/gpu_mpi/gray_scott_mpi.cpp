#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "helpers.hpp"
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

enum Direction { NW = 0, N, NE, W, E, SW, S, SE };
constexpr int n_directions = 8;

// The direction a neighbor sees us in, e.g. our north neighbor sees us to its
// south.
static int opposite(int dir) {
    static constexpr int opposite_dir[n_directions] = {SE, S,  SW, E,
                                                       W,  NE, N,  NW};
    return opposite_dir[dir];
}

using View = Kokkos::View<real**, Kokkos::LayoutRight>;
using StridedRow = Kokkos::View<real*, Kokkos::LayoutStride>;
using Range = Kokkos::pair<std::size_t, std::size_t>;

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

    CartesianDecomposition(std::size_t global_rows_,
                           std::size_t global_columns_)
        : global_rows(global_rows_), global_columns(global_columns_) {
        std::fill(std::begin(neighbors), std::end(neighbors), MPI_PROC_NULL);

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Dims_create(size, 2, dims);

        if (global_rows % static_cast<std::size_t>(dims[0]) != 0 ||
            global_columns % static_cast<std::size_t>(dims[1]) != 0) {
            throw std::runtime_error(
                "Global field dimensions must be divisible by the MPI "
                "Cartesian decomposition.");
        }

        local_rows = global_rows / static_cast<std::size_t>(dims[0]);
        local_columns = global_columns / static_cast<std::size_t>(dims[1]);

        if (local_rows == 0 || local_columns == 0) {
            throw std::runtime_error(
                "Local field dimensions must be greater than zero.");
        }

        // Non-periodic: ranks at the global edge have MPI_PROC_NULL neighbors.
        const int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
        if (comm == MPI_COMM_NULL) {
            throw std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL.");
        }

        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 2, coords);

        // Rank of the process at offset (dr, dc), or MPI_PROC_NULL if outside
        // the grid.
        auto neighbor = [&](int dr, int dc) {
            const int c[2] = {coords[0] + dr, coords[1] + dc};
            if (c[0] < 0 || c[0] >= dims[0] || c[1] < 0 || c[1] >= dims[1]) {
                return static_cast<int>(MPI_PROC_NULL);
            }
            int r = MPI_PROC_NULL;
            MPI_Cart_rank(comm, c, &r);
            return r;
        };

        neighbors[NW] = neighbor(-1, -1);
        neighbors[N] = neighbor(-1, 0);
        neighbors[NE] = neighbor(-1, +1);
        neighbors[W] = neighbor(0, -1);
        neighbors[E] = neighbor(0, +1);
        neighbors[SW] = neighbor(+1, -1);
        neighbors[S] = neighbor(+1, 0);
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
        const std::size_t lengths[n_directions] = {1,    columns, 1,       rows,
                                                   rows, 1,       columns, 1};
        for (int direction = 0; direction < n_directions; ++direction) {
            send[direction] = Kokkos::View<real*>("send", lengths[direction]);
            recv[direction] = Kokkos::View<real*>("recv", lengths[direction]);
        }
    }
};

// -----------------------------------------------------------------------------
// Sub-views of the field
//
// The field has one layer of halo cells around the interior:
//   interior rows:    [1, r]      interior columns: [1, c]
//   halo rows:        0 and r+1   halo columns:     0 and c+1
// -----------------------------------------------------------------------------

// Interior cells adjacent to the boundary in `dir` (these get sent).
static StridedRow pack_subview(int dir, const View& field) {
    const std::size_t r = field.extent(0) - 2;
    const std::size_t c = field.extent(1) - 2;
    switch (dir) {
        case N:
            return Kokkos::subview(field, 1, Range(1, c + 1));
        case S:
            return Kokkos::subview(field, r, Range(1, c + 1));
        case W:
            return Kokkos::subview(field, Range(1, r + 1), 1);
        case E:
            return Kokkos::subview(field, Range(1, r + 1), c);
        case NW:
            return Kokkos::subview(field, 1, Range(1, 2));
        case NE:
            return Kokkos::subview(field, 1, Range(c, c + 1));
        case SW:
            return Kokkos::subview(field, r, Range(1, 2));
        case SE:
            return Kokkos::subview(field, r, Range(c, c + 1));
        default:
            throw std::logic_error("pack_subview: invalid direction");
    }
}

// Halo cells in direction `dir` (these get received into).
static StridedRow halo_subview(int dir, const View& field) {
    const std::size_t r = field.extent(0) - 2;
    const std::size_t c = field.extent(1) - 2;
    switch (dir) {
        case N:
            return Kokkos::subview(field, 0, Range(1, c + 1));
        case S:
            return Kokkos::subview(field, r + 1, Range(1, c + 1));
        case W:
            return Kokkos::subview(field, Range(1, r + 1), 0);
        case E:
            return Kokkos::subview(field, Range(1, r + 1), c + 1);
        case NW:
            return Kokkos::subview(field, 0, Range(0, 1));
        case NE:
            return Kokkos::subview(field, 0, Range(c + 1, c + 2));
        case SW:
            return Kokkos::subview(field, r + 1, Range(0, 1));
        case SE:
            return Kokkos::subview(field, r + 1, Range(c + 1, c + 2));
        default:
            throw std::logic_error("halo_subview: invalid direction");
    }
}

// Copy the boundary cells of `field` in one direction into the send buffer.
static void pack(int direction, const View& field, CommBuffers& buffers) {
    Kokkos::deep_copy(buffers.send[direction], pack_subview(direction, field));
}

// Copy the receive buffer into the halo of `field` in one direction.
static void unpack(int direction, const View& field,
                   const CommBuffers& buffers) {
    Kokkos::deep_copy(halo_subview(direction, field), buffers.recv[direction]);
}

// -----------------------------------------------------------------------------
// Halo exchange
// -----------------------------------------------------------------------------

static void exchange(const View& field, const CartesianDecomposition& d,
                     CommBuffers& b, int tag_base) {
    std::array<MPI_Request, 2 * n_directions> requests{};
    int nreq = 0;

    // Data arriving from the neighbor in direction `dir` was sent by that
    // neighbor towards us, so it carries the tag of the opposite direction
    // as seen from the neighbor. Receives and sends use matching tags below.
    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;
        MPI_Irecv(b.recv[dir].data(), static_cast<int>(b.recv[dir].size()),
                  MPI_REAL_TYPE, d.neighbors[dir], tag_base + dir, d.comm,
                  &requests[nreq++]);
    }

    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;
        pack(dir, field, b);
    }
    // Make sure packing has finished before MPI reads the buffers.
    Kokkos::fence();

    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;
        MPI_Isend(b.send[dir].data(), static_cast<int>(b.send[dir].size()),
                  MPI_REAL_TYPE, d.neighbors[dir], tag_base + opposite(dir),
                  d.comm, &requests[nreq++]);
    }

    MPI_Waitall(nreq, requests.data(), MPI_STATUSES_IGNORE);

    for (int dir = 0; dir < n_directions; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) continue;
        unpack(dir, field, b);
    }
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

static void initialize(const View& u, const View& v, const View& u_temp,
                       const View& v_temp, const CartesianDecomposition& d) {
    using Kokkos::subview;

    // Everything to 0, then interior of u/u_temp to 1.
    Kokkos::deep_copy(u, 0);
    Kokkos::deep_copy(v, 0);
    Kokkos::deep_copy(u_temp, 0);
    Kokkos::deep_copy(v_temp, 0);

    const std::size_t local_rows = d.local_rows;
    const std::size_t local_columns = d.local_columns;

    const Range interior_rows(1, local_rows + 1);
    const Range interior_cols(1, local_columns + 1);

    Kokkos::deep_copy(subview(u, interior_rows, interior_cols), 1);
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
    const std::size_t first_i =
        static_cast<std::size_t>(d.coords[0]) * local_rows;
    const std::size_t first_j =
        static_cast<std::size_t>(d.coords[1]) * local_columns;
    const std::size_t last_i = first_i + local_rows;
    const std::size_t last_j = first_j + local_columns;

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
// Compute
// -----------------------------------------------------------------------------

static void compute(const View& u, const View& v, const View& u_temp,
                    const View& v_temp) {
    const auto nr = static_cast<std::int64_t>(u.extent(0));
    const auto nc = static_cast<std::int64_t>(u.extent(1));

    // Loop over the interior only; the halo provides the stencil neighbors.
    Kokkos::parallel_for(
        "compute",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nr - 1, nc - 1}),
        KOKKOS_LAMBDA(const std::int64_t i, const std::int64_t j) {
            const real u_full = u(i - 1, j - 1) + u(i - 1, j) +
                                u(i - 1, j + 1) + u(i, j - 1) -
                                real(8) * u(i, j) + u(i, j + 1) +
                                u(i + 1, j - 1) + u(i + 1, j) + u(i + 1, j + 1);

            const real v_full = v(i - 1, j - 1) + v(i - 1, j) +
                                v(i - 1, j + 1) + v(i, j - 1) -
                                real(8) * v(i, j) + v(i, j + 1) +
                                v(i + 1, j - 1) + v(i + 1, j) + v(i + 1, j + 1);

            const real uvv = u(i, j) * v(i, j) * v(i, j);

            const real u_delta = constants::diffusion_rate_u * u_full - uvv +
                                 constants::feed_rate * (1 - u(i, j));
            const real v_delta =
                constants::diffusion_rate_v * v_full + uvv -
                (constants::feed_rate + constants::kill_rate) * v(i, j);

            u_temp(i, j) = u(i, j) + u_delta * constants::dt;
            v_temp(i, j) = v(i, j) + v_delta * constants::dt;
        });
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
        KOKKOS_LAMBDA(const std::int64_t i, const std::int64_t j, real& sum) {
            sum += field(i, j);
        },
        checksum);
    return checksum;
}

static real check_global(const View& field, const CartesianDecomposition& d) {
    const real local = check_local(field);
    real global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_REAL_TYPE, MPI_SUM, d.comm);
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
        // Kokkos must be finalized before MPI_Finalize; the try scope ensures
        // that.
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
            throw std::runtime_error(
                "Global field dimensions must be positive.");
        }

        CartesianDecomposition decomposition(global_rows, global_columns);

        const std::size_t ext_rows = decomposition.local_rows + 2;
        const std::size_t ext_cols = decomposition.local_columns + 2;

        View u("u", ext_rows, ext_cols);
        View v("v", ext_rows, ext_cols);
        View u_temp("u_temp", ext_rows, ext_cols);
        View v_temp("v_temp", ext_rows, ext_cols);

        initialize(u, v, u_temp, v_temp, decomposition);

        CommBuffers u_buffers(decomposition.local_rows,
                              decomposition.local_columns);
        CommBuffers v_buffers(decomposition.local_rows,
                              decomposition.local_columns);

        for (std::size_t iteration = 1; iteration <= parameters.n_iterations;
             ++iteration) {
            exchange(u, decomposition, u_buffers, 100);
            exchange(v, decomposition, v_buffers, 200);

            compute(u, v, u_temp, v_temp);

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