#pragma once

#include <Kokkos_Core.hpp>
#include <iomanip>
#include <iostream>

namespace helpers {

/**
 * @brief Display a view on screen.
 * @tparam View Type of view.
 * @param field Field to display.
 * @param iteration Current iteration number.
 */
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

/**
 * @brief Compute and print the checksum of an array.
 * @tparam View Type of view.
 * @param field Field to compute the checksum of.
 * @param iteration Current iteration.
 * @return Checksum value.
 */
template <typename View>
View::value_type print_checksum(const View &field,
                                const std::size_t iteration) {
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

    // set precision
    const auto default_precision{std::cout.precision()};
    std::cout << std::setprecision(2) << std::fixed;

    std::cout << "Checksum field " << field.label() << " at iteration "
              << iteration << ": " << checksum << std::endl;

    // reset precision
    std::cout << std::setprecision(default_precision);

    return checksum;
}

}  // namespace helpers
