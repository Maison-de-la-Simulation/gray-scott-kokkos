/**
 * WARNING
 *
 * This Kokkos implementation is incomplete and can only run on CPU. This is
 * not valid Kokkos code! It requires modifications to be fully portable on
 * GPU.
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <iostream>
#include <utility>

#include "helpers.hpp"
#include "output_writer.hpp"
#include "parameters.hpp"

namespace KE = Kokkos::Experimental;

// data type
using real = double;

// SIMD types
using simd_t = KE::simd<real>;
using simd_scalar_t = KE::basic_simd<real, KE::simd_abi::scalar>;

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
using View = Kokkos::View<real **, Kokkos::LayoutRight>;

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
        Kokkos::MDRangePolicy<
            Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Right>>(
            {i_drop_first, j_drop_first}, {i_drop_last, j_drop_last}),
        KOKKOS_LAMBDA(const int i, const int j) {
            u(i, j) = 0;
            v(i, j) = 1;
        });
}

/**
 * @brief Compute the Gray-Scott update over a subrange of the domain.
 *
 * Applies one timestep of the Gray-Scott equations to elements in the
 * index range [start, end), enabling chunked or parallel execution.
 *
 * @param u U field.
 * @param v V field.
 * @param u_temp U temporary field.
 * @param v_temp V temporary field.
 * @param start Start index (inclusive).
 * @param end End index (exclusive).
 */
template <typename SimdType>
void compute_simd_kernel(const View &u, const View &v, const View &u_temp,
                         const View &v_temp, int start, int end) {
    namespace KE = Kokkos::Experimental;

    const int n_rows = u.extent(0);
    const int n_cols = u.extent(1);

    constexpr int simd_width = SimdType::size();

    int const n_blocks = (end - start) / simd_width;

    int const i_stride = u.stride(0);

    Kokkos::parallel_for(
        "compute_simd",
        Kokkos::MDRangePolicy<
            Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Right>>(
            {1, 0}, {n_rows - 1, n_blocks}),

        KOKKOS_LAMBDA(const int i, const int j_block) {
            // starting column of SIMD lane block (contiguous j-dimension
            // segment) j points to the first element of the SIMD block and
            // advances in steps of simd_width
            const int j = j_block * simd_width + start;

            // compute stencil
            // clang-format off
            SimdType u_full = SimdType(&u(i - 1, j - 1), KE::simd_flag_default) +     SimdType(&u(i - 1, j), KE::simd_flag_default) + SimdType(&u(i - 1, j + 1), KE::simd_flag_default) +
                              SimdType(&u(i    , j - 1), KE::simd_flag_default) - 8 * SimdType(&u(i    , j), KE::simd_flag_default) + SimdType(&u(i    , j + 1), KE::simd_flag_default) +
                              SimdType(&u(i + 1, j - 1), KE::simd_flag_default) +     SimdType(&u(i + 1, j), KE::simd_flag_default) + SimdType(&u(i + 1, j + 1), KE::simd_flag_default);

            SimdType v_full = SimdType(&v(i - 1, j - 1), KE::simd_flag_default) +     SimdType(&v(i - 1, j), KE::simd_flag_default) + SimdType(&v(i - 1, j + 1), KE::simd_flag_default) +
                              SimdType(&v(i    , j - 1), KE::simd_flag_default) - 8 * SimdType(&v(i    , j), KE::simd_flag_default) + SimdType(&v(i    , j + 1), KE::simd_flag_default) +
                              SimdType(&v(i + 1, j - 1), KE::simd_flag_default) +     SimdType(&v(i + 1, j), KE::simd_flag_default) + SimdType(&v(i + 1, j + 1), KE::simd_flag_default);
            // clang-format on

            SimdType uvv = SimdType(&u(i, j), KE::simd_flag_default) * SimdType(&v(i, j), KE::simd_flag_default) * SimdType(&v(i, j), KE::simd_flag_default);

            SimdType u_next =
                u_center + (constants::diffusion_rate_u * u_full - uvv +
                            constants::feed_rate * (1 - SimdType(&u(i, j), KE::simd_flag_default))) *
                               constants::dt;

            SimdType v_next =
                v_center +
                (constants::diffusion_rate_v * v_full + uvv -
                 (constants::feed_rate + constants::kill_rate) * v_center) *
                    constants::dt;

            KE::simd_unchecked_store(u_next, &u_temp(i, j), KE::simd_flag_default);
            KE::simd_unchecked_store(v_next, &v_temp(i, j), KE::simd_flag_default);
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
    const auto simd_width = simd_t::size();

    const int n_cols = u.extent(1);

    const int j_begin = 1;
    const int interior = n_cols - 2;

    const int vec_end = j_begin + (interior / simd_width) * simd_width;

    // SIMD
    compute_simd_kernel<simd_t>(u, v, u_temp, v_temp, j_begin, vec_end);

    // Scalar tail
    if (vec_end < n_cols - 1) {
        compute_simd_kernel<simd_scalar_t>(u, v, u_temp, v_temp, vec_end,
                                           n_cols - 1);
    }
}

/**
 * @brief Compute and print the checksum of an array.
 * @param field Field to compute the checksum of.
 * @param iteration Current iteration.
 * @return Checksum value.
 */
View::value_type check(const View &field, const std::size_t iteration) {
    typename View::value_type checksum;
    Kokkos::parallel_reduce(
        "check fields",
        Kokkos::MDRangePolicy<
            Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Right>>(
            {1, 1}, {field.extent(0) - 1, field.extent(1) - 1}),
        KOKKOS_LAMBDA(const int i, const int j,
                      View::value_type &checksum_local) {
            checksum_local += field(i, j);
        },
        checksum);

    helpers::print_checksum(field.label(), checksum, iteration);

    return checksum;
}

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard kokkos{argc, argv};

    Parameters parameters{argc, argv};
    parameters.check();
    parameters.describe();
    parameters.show_size<real>(4);
    std::cout << "SIMD lane size: " << simd_t::size() << std::endl;

    // fields (with halo)
    View u("u", parameters.n_rows_ext, parameters.n_columns_ext);
    View v("v", parameters.n_rows_ext, parameters.n_columns_ext);

    // create writer
    OutputWriter<real> writer;
    if (parameters.write_results) {
        writer.prepare("gray_scott.h5",
                       parameters.n_iterations / parameters.images_interval,
                       parameters.n_rows_ext, parameters.n_columns_ext);
    }

    // initialize fields
    Kokkos::deep_copy(u, 1);
    Kokkos::deep_copy(v, 0);

    // add a drop at the center of the domain
    add_drop(u, v);

    // print init if requested
    if (parameters.display_fields) {
        helpers::print_field(u, 0);
        helpers::print_field(v, 0);
    }

    // write init
    if (parameters.write_results) {
        writer.write(v.data());
    }

    // temporary fields (with halo)
    View u_temp("u_temp", parameters.n_rows_ext, parameters.n_columns_ext);
    View v_temp("v_temp", parameters.n_rows_ext, parameters.n_columns_ext);

    // time loop
    for (int iteration = 1; iteration <= parameters.n_iterations; iteration++) {
        compute(u, v, u_temp, v_temp);
        std::swap(u, u_temp);
        std::swap(v, v_temp);

        // write image every images_interval iterations
        if (iteration % parameters.images_interval == 0 and
            parameters.write_results) {
            writer.write(v.data());
        }
    }

    // checksum
    check(u, parameters.n_iterations);
    check(v, parameters.n_iterations);

    // print last if requested
    if (parameters.display_fields) {
        helpers::print_field(u, parameters.n_iterations);
        helpers::print_field(v, parameters.n_iterations);
    }
}
