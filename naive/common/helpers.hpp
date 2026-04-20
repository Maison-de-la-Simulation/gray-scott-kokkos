#pragma once

#include <fmt/format.h>
#include <Kokkos_Core.hpp>

namespace helpers {

/**
 * @brief Display a view on screen.
 * @tparam View Type of view.
 * @param field Field to display.
 * @param iteration Current iteration number.
 */
template <typename View>
void print_field(const View &field, const std::size_t iteration) {
    fmt::print("Field {:s} at iteration {}:\n", field.label(), iteration);
    for (int i = 0; i < field.extent(0); i++) {
        for (int j = 0; j < field.extent(1); j++) {
            fmt::print("{:3.2f} ", field(i, j));
        }
        fmt::print("\n");
    }
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

    fmt::print("Checksum field {:s} at iteration {}: {:3.2f}\n", field.label(),
               iteration, checksum);

    return checksum;
}

}  // namespace helpers
