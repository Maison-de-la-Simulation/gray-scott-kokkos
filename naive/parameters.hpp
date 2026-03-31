#include <CLI/CLI.hpp>
#include <format>
#include <iostream>

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
    Parameters(int argc, char *argv[]) {
        parse(argc, argv);
    }

    /**
     * @brief Describe the current parameters.
     */
    void describe() const {
        std::cout << std::format("Number of rows: {}\n", n_rows);
        std::cout << std::format("Number of columns: {}\n", n_columns);
        std::cout << std::format("Number of rows with halo: {}\n", n_rows + 2);
        std::cout << std::format("Number of columns with halo: {}\n",
                                 n_columns_ext);
        std::cout << std::format("Number of elements: {}\n",
                                 (n_rows_ext) * (n_columns_ext));
        std::cout << std::format("Number of iterations: {}\n", n_iterations);
        std::cout << std::format("Number of images: {}\n",
                                 n_iterations / images_interval);
    }

    /**
     * @brief Populate parameters values from the command line.
     * @param argc Number of arguments.
     * @param argv Pointer of array of arguments.
     */
    void parse(int argc, char *argv[]) {
        CLI::App app{"Gray-Scott equation solver implemented with Kokkos"};
        argv = app.ensure_utf8(argv);

        app.add_option("-n,--rows", n_rows, "Number of rows");
        app.add_option("-m,--columns", n_columns, "Number of columns");

        app.add_option("-i,--iterations", n_iterations, "Number of iterations");
        app.add_option("-t,--interval", images_interval,
                       "Number of iterations between two snapshots");

        app.add_flag("-d", display_fields,
                     "Display fields on screen at the beginning and at the end "
                     "of the execution");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            exit(app.exit(e));
        }

        n_rows_ext = n_rows + 2;
        n_columns_ext = n_columns + 2;
    }
};
