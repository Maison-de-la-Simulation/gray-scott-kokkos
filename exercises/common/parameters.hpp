#pragma once

#include <cstdlib>
#include <iostream>

/**
 * @brief Structure containing parameters of the program.
 */
struct Parameters {
    // problem size
    std::size_t n_rows{10};
    std::size_t n_columns{10};

    // iterations
    std::size_t n_iterations{100};
    std::size_t images_interval{10};

    // problem size including ghost cells
    std::size_t n_rows_ext{n_rows + 2};
    std::size_t n_columns_ext{n_columns + 2};

    // debug
    bool display_fields = false;

    // ios
    bool write_results = true;

    /**
     * @brief Default constructor.
     */
    Parameters() = default;

    /**
     * @brief Constructor from the command line.
     * @param argc Number of arguments.
     * @param argv Pointer of array of arguments.
     */
    Parameters(int argc, char* argv[]);

    /**
     * @brief Describe the current parameters.
     */
    void describe() const;

    /**
     * @brief Show the size of elements for the number of arrays.
     * @tparam real Type of data in the array.
     * @param n_arrays The number of arrays that will be used.
     * @param label Label to show (usually CPU, GPU, or CPU/GPU).
     */
    template <typename real>
    void show_size(const std::size_t n_arrays,
                   const char* label = nullptr) const;

    /**
     * @brief Populate parameters values from the command line.
     * @param argc Number of arguments.
     * @param argv Pointer of array of arguments.
     */
    void parse(int argc, char* argv[]);

    /**
     * @brief Check the validity of the entered values.
     * Exits the program if values are invalid.
     */
    void check() const;
};

// implementations

template <typename real>
void Parameters::show_size(const std::size_t n_arrays,
                           const char* label) const {
    std::cout << "Memory size (" << n_arrays << " arrays): "
              << n_arrays * this->n_rows_ext * this->n_columns_ext *
                     sizeof(real)
              << " bytes";

    if (label) {
        std::cout << "(" << label << ")";
    }

    std::cout << std::endl;
}
