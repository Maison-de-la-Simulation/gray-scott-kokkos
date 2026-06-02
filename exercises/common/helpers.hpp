#pragma once

#include <Kokkos_Core.hpp>
#include <iomanip>
#include <iostream>
#include <string>

namespace helpers {

/**
 * @brief Display a view on screen.
 * @tparam View Type of view.
 * @param field Field to display.
 * @param iteration Current iteration number.
 */
template <typename View>
void print_field(const View &field, const std::size_t iteration);

/**
 * @brief Print a checksum on screen.
 * @tparam real Type of float.
 * @param label Name of the field.
 * @param checksum Value of the checksum.
 * @param iteration Current iteration.
 */
template <typename real>
void print_checksum(const std::string &label, const real checksum,
                    const std::size_t iteration);

/**
 * @brief Print a checksum on screen.
 * @tparam real Type of float.
 * @param label Name of the field.
 * @param checksum Value of the checksum.
 * @param iteration Current iteration.
 */
template <typename real>
void print_checksum(const char *label, const real checksum,
                    const std::size_t iteration);

// implementations

template <typename View>
void print_field(const View &field, const std::size_t iteration) {
    std::cout << "Field " << field.label() << " at iteration " << iteration
              << std::endl;

    // set precision
    const auto default_precision{std::cout.precision()};
    std::cout << std::setprecision(2) << std::fixed;

    for (int i = 0; i < field.extent(0); i++) {
        for (int j = 0; j < field.extent(1); j++) {
            std::cout << field(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // reset precision
    std::cout << std::setprecision(default_precision);
}

template <typename real>
void print_checksum(const std::string &label, const real checksum,
                    const std::size_t iteration) {
    print_checksum(label.c_str(), checksum, iteration);
}

template <typename real>
void print_checksum(const char *label, const real checksum,
                    const std::size_t iteration) {
    // set precision
    const auto default_precision{std::cout.precision()};
    std::cout << std::setprecision(2) << std::fixed;

    std::cout << "Checksum field " << label << " at iteration " << iteration
              << ": " << checksum << std::endl;

    // reset precision
    std::cout << std::setprecision(default_precision);
}

}  // namespace helpers
