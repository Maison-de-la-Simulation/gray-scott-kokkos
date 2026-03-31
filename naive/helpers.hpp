#include <Kokkos_Core.hpp>
#include <format>
#include <iostream>

namespace helpers {

/**
 * @brief Display a view on screen.
 * @param field Field to display.
 * @param iteration Current iteration number.
 */
template <typename View>
void print_field(const View &field, const int iteration) {
    std::cout << std::format("field {:s} at iteration {}:\n",
                             field.label().c_str(), iteration);
    for (int i = 0; i < field.extent(0); i++) {
        for (int j = 0; j < field.extent(1); j++) {
            std::cout << std::format("{:3.2f} ", field(i, j));
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Compute and print the checksum of an array.
 * @param field Field to compute the checksum of.
 * @param iteration Current iteration.
 * @return Checksum value.
 */
template <typename View>
View::value_type print_checksum(const View &field, const int iteration) {
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

    std::cout << std::format("checksum field {:s} at iteration {}: {:3.2f}\n",
                             field.label().c_str(), iteration, checksum);

    return checksum;
}

}  // namespace helpers
