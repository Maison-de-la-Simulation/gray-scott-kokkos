#include <format>
#include <iostream>

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
    bool display_fields = true;

    void describe() const {
        std::cout << std::format("Number of rows: %d\n", n_rows);
        std::cout << std::format("Number of columns: %d\n", n_columns);
        std::cout << std::format("Number of rows with halo: %d\n", n_rows + 2);
        std::cout << std::format("Number of columns with halo: %d\n",
                                 n_columns_ext);
        std::cout << std::format("Number of elements: %d\n",
                                 (n_rows_ext) * (n_columns_ext));
        std::cout << std::format("Number of iterations: %d\n", n_iterations);
        std::cout << std::format("Number of images: %d\n",
                                 n_iterations / images_interval);
    }
};
