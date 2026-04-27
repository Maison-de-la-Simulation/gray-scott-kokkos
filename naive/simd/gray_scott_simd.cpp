#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

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

} // namespace constants

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
void compute_simd_kernal(const View &u, const View &v, const View &u_temp,
                         const View &v_temp, int start, int end) {
  namespace KE = Kokkos::Experimental;

  const int n_rows = u.extent(0);
  const int n_cols = u.extent(1);

  constexpr int simd_width = SimdType::size();

  int const n_blocks = (end - start) / simd_width;

  int strides[2];
  u.stride(strides);
  int const i_stride = strides[0];

  Kokkos::parallel_for(
      "compute_simd",
      Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Right>>(
          {1, 0}, {n_rows - 1, n_blocks}),

      KOKKOS_LAMBDA(const int i, const int j_block) {
        const int j0 = j_block * simd_width + start;
        const int base_center = i_stride * i + j0;

        // Load SIMD registers for 3x3 stencil

        SimdType u_center(u.data() + base_center, KE::simd_flag_default);
        SimdType v_center(v.data() + base_center, KE::simd_flag_default);

        SimdType u_left(u.data() + base_center - 1, KE::simd_flag_default);
        SimdType u_right(u.data() + base_center + 1, KE::simd_flag_default);

        SimdType v_left(v.data() + base_center - 1, KE::simd_flag_default);
        SimdType v_right(v.data() + base_center + 1, KE::simd_flag_default);

        // Up and Down Center
        const int base_up = base_center - i_stride;
        const int base_down = base_center + i_stride;

        SimdType u_up(u.data() + base_up, KE::simd_flag_default);
        SimdType u_down(u.data() + base_down, KE::simd_flag_default);

        SimdType v_up(v.data() + base_up, KE::simd_flag_default);
        SimdType v_down(v.data() + base_down, KE::simd_flag_default);

        // Left Corners
        const int base_down_left = base_down - 1;
        const int base_up_left = base_up - 1;

        // Right Corners
        const int base_down_right = base_down + 1;
        const int base_up_right = base_up + 1;

        // Corners u
        SimdType u_ul(u.data() + base_up_left, KE::simd_flag_default);
        SimdType u_ur(u.data() + base_up_right, KE::simd_flag_default);
        SimdType u_dl(u.data() + base_down_left, KE::simd_flag_default);
        SimdType u_dr(u.data() + base_down_right, KE::simd_flag_default);

        // Corners v
        SimdType v_ul(v.data() + base_up_left, KE::simd_flag_default);
        SimdType v_ur(v.data() + base_up_right, KE::simd_flag_default);
        SimdType v_dl(v.data() + base_down_left, KE::simd_flag_default);
        SimdType v_dr(v.data() + base_down_right, KE::simd_flag_default);

        // Compute stencil
        SimdType u_full =
            (u_left + u_right + u_up + u_down + u_ul + u_ur + u_dl + u_dr) -
            8 * u_center;

        SimdType v_full =
            (v_left + v_right + v_up + v_down + v_ul + v_ur + v_dl + v_dr) -
            8 * v_center;

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
  compute_simd_kernal<simd_t>(u, v, u_temp, v_temp, j_begin, vec_end);

  // Scalar tail
  if (vec_end < n_cols - 1) {
    compute_simd_kernal<simd_scalar_t>(u, v, u_temp, v_temp, vec_end,
                                       n_cols - 1);
  }
}

int main(int argc, char *argv[]) {
  Kokkos::ScopeGuard kokkos{argc, argv};

  Parameters parameters{argc, argv};
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
    Kokkos::kokkos_swap(u, u_temp);
    Kokkos::kokkos_swap(v, v_temp);

    // write image every images_interval iterations
    if (iteration % parameters.images_interval == 0) {
      writer.write(v.data());
    }
  }

  // checksum
  helpers::print_checksum(u, parameters.n_iterations);
  helpers::print_checksum(v, parameters.n_iterations);

  // print last if requested
  if (parameters.display_fields) {
    helpers::print_field(u, parameters.n_iterations);
    helpers::print_field(v, parameters.n_iterations);
  }
}
