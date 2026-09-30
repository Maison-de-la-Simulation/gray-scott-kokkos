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

static StridedRow pack_subview(int dir, const View& field, int r, int c) {
    switch (dir) {
        case N:  return Kokkos::subview(field, 1, Kokkos::make_pair(int(1), c + 1));
        case S:  return Kokkos::subview(field, r, Kokkos::make_pair(int(1), c + 1));
        case W:  return Kokkos::subview(field, Kokkos::make_pair(int(1), r + 1), 1);
        case E:  return Kokkos::subview(field, Kokkos::make_pair(int(1), r + 1), c);
        case NW: return Kokkos::subview(field, 1, Kokkos::make_pair(int(1), int(2)));
        case NE: return Kokkos::subview(field, 1, Kokkos::make_pair(c, c + 1));
        case SW: return Kokkos::subview(field, r, Kokkos::make_pair(int(1), int(2)));
        case SE: return Kokkos::subview(field, r, Kokkos::make_pair(c, c + 1));
        default: throw std::logic_error("pack_subview: invalid direction");
    }
}

static StridedRow halo_subview(int dir, View& field, int r, int c) {
    switch (dir) {
        case N:  return Kokkos::subview(field, 0,     Kokkos::make_pair(int(1), c + 1));
        case S:  return Kokkos::subview(field, r + 1, Kokkos::make_pair(int(1), c + 1));
        case W:  return Kokkos::subview(field, Kokkos::make_pair(int(1), r + 1), 0);
        case E:  return Kokkos::subview(field, Kokkos::make_pair(int(1), r + 1), c + 1);
        case NW: return Kokkos::subview(field, 0,     Kokkos::make_pair(int(0), int(1)));
        case NE: return Kokkos::subview(field, 0,     Kokkos::make_pair(c + 1, c + 2));
        case SW: return Kokkos::subview(field, r + 1, Kokkos::make_pair(int(0), int(1)));
        case SE: return Kokkos::subview(field, r + 1, Kokkos::make_pair(c + 1, c + 2));
        default: throw std::logic_error("halo_subview: invalid direction");
    }
}


void pack(int direction, const View& field, CommBuffers& buffers) {
    Kokkos::deep_copy(buffers.send[direction], pack_subview(
        direction, field, this->local_rows, this->local_columns));
}

// -----------------------------------------------------------------------------
// Unpack one field.
//
// For an internal boundary, use the received MPI halo.
//
// For a global physical boundary, set the halo to zero. This implements the
// same zero boundary condition as the original serial code.
// -----------------------------------------------------------------------------

void unpack(int direction, View& field, const CommBuffers& buffers) {
    Kokkos::deep_copy(this->halo_subview(
        direction, field, this->local_rows, this->local_columns), buffers.recv[direction]);
}


// -----------------------------------------------------------------------------
// Halo exchange
// -----------------------------------------------------------------------------

static void exchange(View& field, const CartesianDecomposition& d,
                     CommBuffers& b, int tag_base) {
    // This first MPI reference version explicitly packs and unpacks.
    // Communication is posted for all eight neighbors before Waitall.
    pack(field, b, d);

    std::array<MPI_Request, 16> requests{};
    int nreq = 0;

    for (int dir = 0; dir < 8; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) {
            continue;
        }
        MPI_Irecv(b.recv[dir].data(),
                  static_cast<int>(b.recv[dir].size()),
                  MPI_REAL_TYPE,
                  d.neighbors[dir], tag_base + dir, d.comm,
                  &requests[nreq++]);
    }

    for (int dir = 0; dir < 8; ++dir) {
        if (d.neighbors[dir] == MPI_PROC_NULL) {
            continue;
        }
        MPI_Isend(b.send[dir].data(),
                  static_cast<int>(b.send[dir].size()),
                  MPI_REAL_TYPE,
                  d.neighbors[dir], tag_base + opposite(dir), d.comm,
                  &requests[nreq++]);
    }

    MPI_Waitall(nreq, requests.data(), MPI_STATUSES_IGNORE);
    unpack(field, b, d);
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
            {nr - 1, nc - 1}),
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
        MPI_REAL_TYPE,
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
            
            exchange(u, decomposition, u_buffers, 100);
            exchange(v, decomposition, v_buffers, 200);

            compute(
                u,
                v,
                u_temp,
                v_temp);

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