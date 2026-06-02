#include <Kokkos_Core.hpp>
#include <utility>

#include "helpers.hpp"
#include "output_writer.hpp"
#include "parameters.hpp"

// data type
using real = double;

/**
 * @brief Constants for the Gray-Scott equation.
 */
namespace constants {

constexpr real kill_rate{0.054};
constexpr real feed_rate{0.014};
constexpr real dt{1.0};
constexpr real diffusion_rate_u{0.1};
constexpr real diffusion_rate_v{0.05};

}  // namespace constants

// views type
using View = Kokkos::View<real **>;

/**
 * @brief Add a drop at the center of the fields.
 * @param u U field.
 * @param v V field.
 */
void add_drop(const View &u, const View &v) {
    const std::size_t n_rows_ext = u.extent(0);
    const std::size_t n_columns_ext = u.extent(1);

    // find drop location
    // central cell + 1
    const std::size_t i_center = n_rows_ext / 2;
    const std::size_t j_center = n_columns_ext / 2;
    const std::size_t i_drop_first = i_center - 1;
    const std::size_t i_drop_last = i_center + 1;
    const std::size_t j_drop_first = j_center - 1;
    const std::size_t j_drop_last = j_center + 1;

    Kokkos::parallel_for(
        "add drop",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i_drop_first, j_drop_first},
                                               {i_drop_last, j_drop_last}),
        KOKKOS_LAMBDA(const int i, const int j) {
            u(i, j) = 0;
            v(i, j) = 1;
        });
}

/**
 * @brief Compute the Gray-Scott equation for one iteration.
 * @param u U field.
 * @param v V field.
 * @param u_temp U temporary field.
 * @param v_temp V temporary field.
 */
void compute(const View &u, const View &v, const View &u_temp,
             const View &v_temp) {
    const std::size_t n_rows_ext = u.extent(0);
    const std::size_t n_columns_ext = u.extent(1);

    Kokkos::parallel_for(
        "compute",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {1, 1},
            {n_rows_ext - 1, n_columns_ext - 1}),  // do not iterate on the halo
        KOKKOS_LAMBDA(const int i, const int j) {
            // clang-format off
            real u_full = u(i - 1, j - 1) +     u(i - 1, j) + u(i - 1, j + 1) +
                          u(i    , j - 1) - 8 * u(i    , j) + u(i    , j + 1) +
                          u(i + 1, j - 1) +     u(i + 1, j) + u(i + 1, j + 1);

            real v_full = v(i - 1, j - 1) +     v(i - 1, j) + v(i - 1, j + 1) +
                          v(i    , j - 1) - 8 * v(i    , j) + v(i    , j + 1) +
                          v(i + 1, j - 1) +     v(i + 1, j) + v(i + 1, j + 1);
            // clang-format on

            const real uvv = u(i, j) * v(i, j) * v(i, j);

            const real u_delta = constants::diffusion_rate_u * u_full - uvv +
                                 constants::feed_rate * (1 - u(i, j));
            const real v_delta =
                constants::diffusion_rate_v * v_full + uvv -
                (constants::feed_rate + constants::kill_rate) * v(i, j);

            u_temp(i, j) = u(i, j) + u_delta * constants::dt;
            v_temp(i, j) = v(i, j) + v_delta * constants::dt;
        });

    Kokkos::fence("wait for compute");
}

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard kokkos{argc, argv};

    Parameters parameters{argc, argv};
    parameters.check();
    parameters.describe();

    // fields (with halo)
    View u("u", parameters.n_rows_ext, parameters.n_columns_ext);
    View v("v", parameters.n_rows_ext, parameters.n_columns_ext);

    // mirrors of the fields (with halo)
    auto u_h = Kokkos::create_mirror_view(u);
    auto v_h = Kokkos::create_mirror_view(v);

    // create writer
    OutputWriter<real> writer(
        "gray_scott.h5", parameters.n_iterations / parameters.images_interval,
        parameters.n_rows_ext, parameters.n_columns_ext);

    // initialize fields
    Kokkos::deep_copy(u, 1);
    Kokkos::deep_copy(v, 0);

    // add a drop at the center of the domain
    add_drop(u, v);

    // transfer fields
    Kokkos::deep_copy(u_h, u);
    Kokkos::deep_copy(v_h, v);

    // print init if requested
    if (parameters.display_fields) {
        helpers::print_field(u_h, 0);
        helpers::print_field(v_h, 0);
    }

    // write init
    writer.write(v_h.data());

    // temporary fields (with halo)
    View u_temp("u_temp", parameters.n_rows_ext, parameters.n_columns_ext);
    View v_temp("v_temp", parameters.n_rows_ext, parameters.n_columns_ext);

    // time loop
    for (int iteration = 0; iteration < parameters.n_iterations; iteration++) {
        compute(u, v, u_temp, v_temp);
        std::swap(u, u_temp);
        std::swap(v, v_temp);

        // write image every images_interval iterations
        if (iteration % parameters.images_interval == 0) {
            Kokkos::deep_copy(v_h, v);
            writer.write(v_h.data());
        }
    }

    // checksum
    helpers::print_checksum(u, parameters.n_iterations);
    helpers::print_checksum(v, parameters.n_iterations);

    // transfer fields
    Kokkos::deep_copy(u_h, u);
    Kokkos::deep_copy(v_h, v);

    // print last if requested
    if (parameters.display_fields) {
        helpers::print_field(u_h, parameters.n_iterations);
        helpers::print_field(v_h, parameters.n_iterations);
    }
}
