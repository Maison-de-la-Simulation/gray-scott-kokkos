/**
 * WARNING
 *
 * This Kokkos implementation is incomplete and can only run on CPU. This is
 * not valid Kokkos code! It requires modifications to be fully portable on
 * GPU.
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
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
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i_drop_first, j_drop_first},
                                               {i_drop_last, j_drop_last}),
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
        Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right>>(
            {1, 0}, {n_rows - 1, n_blocks}),

        KOKKOS_LAMBDA(const int i, const int j_block) {
            // starting column of SIMD lane block (contiguous j-dimension
            // segment) j0 points to the first element of the SIMD block and
            // advances in steps of simd_width
            const int j0 = j_block * simd_width + start;

            // flattened 2D index using the row stride of the LayoutRight view
            // corresponds to the grid cell at position (i, j0)
            const int base_center = i_stride * i + j0;

            // center u pointer
            const auto u_center_ptr = u.data() + base_center;

            // center v pointer
            const auto v_center_ptr = v.data() + base_center;

            // Load SIMD registers for 3x3 stencil

            // west-east u
            SimdType u_center(u_center_ptr, KE::simd_flag_default);
            SimdType u_west(u_center_ptr - 1, KE::simd_flag_default);
            SimdType u_east(u_center_ptr + 1, KE::simd_flag_default);

            // west-east v
            SimdType v_center(v_center_ptr, KE::simd_flag_default);
            SimdType v_west(v_center_ptr - 1, KE::simd_flag_default);
            SimdType v_east(v_center_ptr + 1, KE::simd_flag_default);

            // north-south u pointers
            const auto u_north_ptr = u_center_ptr - i_stride;
            const auto u_south_ptr = u_center_ptr + i_stride;

            // north-south v pointers
            const auto v_north_ptr = v_center_ptr - i_stride;
            const auto v_south_ptr = v_center_ptr + i_stride;

            // north-south u
            SimdType u_north(u_north_ptr, KE::simd_flag_default);
            SimdType u_south(u_south_ptr, KE::simd_flag_default);

            // north-south v
            SimdType v_north(v_north_ptr, KE::simd_flag_default);
            SimdType v_south(v_south_ptr, KE::simd_flag_default);

            // corners u
            SimdType u_north_west(u_north_ptr - 1, KE::simd_flag_default);
            SimdType u_north_east(u_north_ptr + 1, KE::simd_flag_default);
            SimdType u_south_west(u_south_ptr - 1, KE::simd_flag_default);
            SimdType u_south_east(u_south_ptr + 1, KE::simd_flag_default);

            // corners v
            SimdType v_north_west(v_north_ptr - 1, KE::simd_flag_default);
            SimdType v_north_east(v_north_ptr + 1, KE::simd_flag_default);
            SimdType v_south_west(v_south_ptr - 1, KE::simd_flag_default);
            SimdType v_south_east(v_south_ptr + 1, KE::simd_flag_default);

            // compute stencil
            // clang-format off
            SimdType u_full = u_north_west +     u_north  + u_north_east +
                              u_west       - 8 * u_center + u_east +
                              u_south_west +     u_south  + u_south_east;

            SimdType v_full = v_north_west +     v_north  + v_north_east +
                              v_west       - 8 * v_center + v_east +
                              v_south_west +     v_south  + v_south_east;
            // clang-format on

            SimdType uvv = u_center * v_center * v_center;

            SimdType u_next =
                u_center + (constants::diffusion_rate_u * u_full - uvv +
                            constants::feed_rate * (1 - u_center)) *
                               constants::dt;

            SimdType v_next =
                v_center +
                (constants::diffusion_rate_v * v_full + uvv -
                 (constants::feed_rate + constants::kill_rate) * v_center) *
                    constants::dt;

            KE::simd_unchecked_store(u_next, u_temp.data() + base_center,
                                     KE::simd_flag_default);
            KE::simd_unchecked_store(v_next, v_temp.data() + base_center,
                                     KE::simd_flag_default);
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
    namespace KE = Kokkos::Experimental;

    using simd_t = KE::simd<real>;
    using simd_scalar_t = KE::basic_simd<real, KE::simd_abi::scalar>;

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
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {field.extent(0), field.extent(1)}),
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

    // fields (with halo)
    View u("u", parameters.n_rows_ext, parameters.n_columns_ext);
    View v("v", parameters.n_rows_ext, parameters.n_columns_ext);

    // create writer
    OutputWriter<real> writer(
        "gray_scott.h5", parameters.n_iterations / parameters.images_interval,
        parameters.n_rows_ext, parameters.n_columns_ext);

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
    writer.write(v.data());

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
