#include <format>
#include <iostream>

#include "macros_pod.hpp"

namespace helpers_pod {

/**
 * @brief Display a view on screen.
 * @tparam real Type of data.
 * @param field Pointer of data.
 * @param label Name of the field.
 * @param n_rows Number of rows.
 * @param n_columns Number of columns.
 * @param iteration Current iteration number.
 */
template <typename real>
void print_field(real *field, char const *label, const std::size_t n_rows_ext,
                 const std::size_t n_columns_ext, const int iteration) {
    std::cout << std::format("Field {:s} at iteration {}:\n", label, iteration);
    for (int i = 0; i < n_rows_ext; i++) {
        for (int j = 0; j < n_columns_ext; j++) {
            std::cout << std::format("{:3.2f} ", field[ACCESS(i, j)]);
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Compute and print the checksum of an array.
 * @tparam real Type of data.
 * @param field Pointer of data.
 * @param label Name of field.
 * @param n_rows Number of rows.
 * @param n_columns Number of columns.
 * @param iteration Current iteration.
 * @return Checksum value.
 */
template <typename real>
real print_checksum(const real *field, char const *label,
                    const std::size_t n_rows_ext,
                    const std::size_t n_columns_ext,
                    const std::size_t iteration) {
    real checksum = 0;
    for (int i = 0; i < n_rows_ext; i++) {
        for (int j = 0; j < n_columns_ext; j++) {
            checksum += field[ACCESS(i, j)];
        }
    }

    std::cout << std::format("Checksum field {:s} at iteration {}: {:3.2f}\n",
                             label, iteration, checksum);

    return checksum;
}

}  // namespace helpers_pod
