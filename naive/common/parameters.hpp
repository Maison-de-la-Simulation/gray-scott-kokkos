#pragma once

#include <cstdlib>

/**
 * @brief Structure containing parameters of the program.
 */
struct Parameters {
    // problem size
    std::size_t n_rows = 10;
    std::size_t n_columns = 10;

    // iterations
    std::size_t n_iterations = 100;
    std::size_t images_interval = 10;

    // problem size including ghost cells
    std::size_t n_rows_ext = n_rows + 2;
    std::size_t n_columns_ext = n_columns + 2;

    // debug
    bool display_fields = false;

    /**
     * @brief Default constructor.
     */
    Parameters() = default;

    /**
     * @brief Constructor from the command line.
     * @param argc Number of arguments.
     * @param argv Pointer of array of arguments.
     */
    Parameters(int argc, char *argv[]) { parse(argc, argv); }

    /**
     * @brief Describe the current parameters.
     */
    void describe() const;

    /**
     * @brief Populate parameters values from the command line.
     * @param argc Number of arguments.
     * @param argv Pointer of array of arguments.
     */
    void parse(int argc, char *argv[]);
};
