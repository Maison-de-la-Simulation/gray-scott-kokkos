#include <cstddef>
#include <utility>

#include "helpers.hpp"
#include "macros_pod.hpp"
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

/**
 * @brief Add a drop at the center of the fields.
 * @param u U field.
 * @param v V field.
 * @param n_rows_ext Number of rows + halo.
 * @param n_columns_ext Number of columens + halo.
 */
void add_drop(real *u, real *v, const std::size_t n_rows_ext,
              const std::size_t n_columns_ext) {
    // find drop location
    // central cell + 1
    const std::size_t i_center = n_rows_ext / 2;
    const std::size_t j_center = n_columns_ext / 2;
    const std::size_t i_drop_first = i_center - 1;
    const std::size_t i_drop_last = i_center + 1;
    const std::size_t j_drop_first = j_center - 1;
    const std::size_t j_drop_last = j_center + 1;

    for (std::size_t i = i_drop_first; i < i_drop_last; i++) {
        for (std::size_t j = j_drop_first; j < j_drop_last; j++) {
            u[ACCESS(i, j)] = 0;
            v[ACCESS(i, j)] = 1;
        }
    }
}

/**
 * @brief Compute the Gray-Scott equation for one iteration.
 * @param u U field.
 * @param v V field.
 * @param u_temp U temporary field.
 * @param v_temp V temporary field.
 * @param n_rows_ext Number of rows + halo.
 * @param n_columns_ext Number of columns + halo.
 */
void compute(real const *u, real const *v, real *u_temp, real *v_temp,
             const std::size_t n_rows_ext, const std::size_t n_columns_ext) {
    for (std::size_t i = 1; i < n_rows_ext - 1; i++) {
        for (std::size_t j = 1; j < n_columns_ext - 1; j++) {
            // clang-format off
            real u_full = u[ACCESS(i - 1, j - 1)] +     u[ACCESS(i - 1, j)] + u[ACCESS(i - 1, j + 1)] +
                          u[ACCESS(i    , j - 1)] - 8 * u[ACCESS(i    , j)] + u[ACCESS(i    , j + 1)] +
                          u[ACCESS(i + 1, j - 1)] +     u[ACCESS(i + 1, j)] + u[ACCESS(i + 1, j + 1)];

            real v_full = v[ACCESS(i - 1, j - 1)] +     v[ACCESS(i - 1, j)] + v[ACCESS(i - 1, j + 1)] +
                          v[ACCESS(i    , j - 1)] - 8 * v[ACCESS(i    , j)] + v[ACCESS(i    , j + 1)] +
                          v[ACCESS(i + 1, j - 1)] +     v[ACCESS(i + 1, j)] + v[ACCESS(i + 1, j + 1)];
            // clang-format on

            const real uvv =
                u[ACCESS(i, j)] * v[ACCESS(i, j)] * v[ACCESS(i, j)];

            const real u_delta = constants::diffusion_rate_u * u_full - uvv +
                                 constants::feed_rate * (1 - u[ACCESS(i, j)]);
            const real v_delta =
                constants::diffusion_rate_v * v_full + uvv -
                (constants::feed_rate + constants::kill_rate) * v[ACCESS(i, j)];

            u_temp[ACCESS(i, j)] = u[ACCESS(i, j)] + u_delta * constants::dt;
            v_temp[ACCESS(i, j)] = v[ACCESS(i, j)] + v_delta * constants::dt;
        }
    }
}

/**
 * @brief Compute and print the checksum of an array.
 * @param field Pointer of data.
 * @param label Name of field.
 * @param n_rows Number of rows.
 * @param n_columns Number of columns.
 * @param iteration Current iteration.
 * @return Checksum value.
 */
real check(const real *field, const char *label, const std::size_t n_rows_ext,
           const std::size_t n_columns_ext, const std::size_t iteration) {
    real checksum = 0;
    for (int i = 1; i < n_rows_ext - 1; i++) {
        for (int j = 1; j < n_columns_ext - 1; j++) {
            checksum += field[ACCESS(i, j)];
        }
    }

    helpers::print_checksum(label, checksum, iteration);

    return checksum;
}

int main(int argc, char *argv[]) {
    Parameters parameters{argc, argv};
    parameters.check();
    parameters.describe();
    parameters.show_size<real>(4);

    // fields (with halo)
    real *u = new real[parameters.n_rows_ext * parameters.n_columns_ext];
    real *v = new real[parameters.n_rows_ext * parameters.n_columns_ext];

    // create writer
    OutputWriter<real> writer;
    if (parameters.write_results) {
        writer.prepare("gray_scott.h5",
                       parameters.n_iterations / parameters.images_interval,
                       parameters.n_rows_ext, parameters.n_columns_ext);
    }

    // initialize fields
    for (std::size_t i = 0;
         i < parameters.n_rows_ext * parameters.n_columns_ext; i++) {
        u[i] = 1;
        v[i] = 0;
    }

    // add a drop at the center of the domain
    add_drop(u, v, parameters.n_rows_ext, parameters.n_columns_ext);

    // print init if requested
    if (parameters.display_fields) {
        helpers::print_field(u, "u", parameters.n_rows_ext,
                             parameters.n_columns_ext, 0);
        helpers::print_field(v, "v", parameters.n_rows_ext,
                             parameters.n_columns_ext, 0);
    }

    // write init
    if (parameters.write_results) {
        writer.write(v);
    }

    // temporary fields (with halo)
    real *u_temp = new real[parameters.n_rows_ext * parameters.n_columns_ext];
    real *v_temp = new real[parameters.n_rows_ext * parameters.n_columns_ext];

    // time loop
    for (std::size_t iteration = 1; iteration <= parameters.n_iterations;
         iteration++) {
        compute(u, v, u_temp, v_temp, parameters.n_rows_ext,
                parameters.n_columns_ext);
        std::swap(u, u_temp);
        std::swap(v, v_temp);

        // write image every images_interval iterations
        if (iteration % parameters.images_interval == 0 and
            parameters.write_results) {
            writer.write(v);
        }
    }

    // checksum
    check(u, "u", parameters.n_rows_ext, parameters.n_columns_ext,
          parameters.n_iterations);
    check(v, "v", parameters.n_rows_ext, parameters.n_columns_ext,
          parameters.n_iterations);

    // print last if requested
    if (parameters.display_fields) {
        helpers::print_field(u, "u", parameters.n_rows_ext,
                             parameters.n_columns_ext, parameters.n_iterations);
        helpers::print_field(v, "v", parameters.n_rows_ext,
                             parameters.n_columns_ext, parameters.n_iterations);
    }

    // free memory
    delete[] u;
    delete[] v;
    delete[] u_temp;
    delete[] v_temp;
}
