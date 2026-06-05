#pragma once

#include <iomanip>
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
                 const std::size_t n_columns_ext, const int iteration);

// implementations

template <typename real>
void print_field(real *field, char const *label, const std::size_t n_rows_ext,
                 const std::size_t n_columns_ext, const int iteration) {
    std::cout << "Field " << label << " at iteration " << iteration
              << std::endl;

    // set precision
    const auto default_precision{std::cout.precision()};
    std::cout << std::setprecision(2) << std::fixed;

    for (int i = 0; i < n_rows_ext; i++) {
        for (int j = 0; j < n_columns_ext; j++) {
            std::cout << field[ACCESS(i, j)] << " ";
        }
        std::cout << std::endl;
    }

    // reset precision
    std::cout << std::setprecision(default_precision);
}

}  // namespace helpers_pod
