#include "parameters.hpp"

#include <CLI/CLI.hpp>
#include <iostream>

void Parameters::describe() const {
    std::cout << "Number of rows: " << n_rows << std::endl;
    std::cout << "Number of columns: " << n_columns << std::endl;
    std::cout << "Number of rows (with halo): " << n_rows + 2 << std::endl;
    std::cout << "Number of columns (with halo): " << n_columns_ext
              << std::endl;
    std::cout << "Number of elements: " << n_rows_ext * n_columns_ext
              << std::endl;
    std::cout << "Number of iterations: " << n_iterations << std::endl;
    std::cout << "Number of images: " << n_iterations / images_interval
              << std::endl;
}

void Parameters::parse(int argc, char *argv[]) {
    CLI::App app{"Gray-Scott equation solver implemented with Kokkos"};
    argv = app.ensure_utf8(argv);

    app.add_option("-n,--rows", this->n_rows, "Number of rows");
    app.add_option("-m,--columns", this->n_columns, "Number of columns");

    app.add_option("-i,--iterations", this->n_iterations,
                   "Number of iterations");
    app.add_option("-t,--interval", this->images_interval,
                   "Number of iterations between two snapshots");

    app.add_flag("-d", this->display_fields,
                 "Display fields on screen at the beginning and at the end "
                 "of the execution");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        exit(app.exit(e));
    }

    this->n_rows_ext = this->n_rows + 2;
    this->n_columns_ext = this->n_columns + 2;
}
