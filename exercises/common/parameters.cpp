#include "parameters.hpp"

#include <CLI/CLI.hpp>
#include <iostream>

Parameters::Parameters(int argc, char *argv[]) { parse(argc, argv); }

void Parameters::describe() const {
    std::cout << "Number of rows: " << n_rows << std::endl;
    std::cout << "Number of columns: " << n_columns << std::endl;
    std::cout << "Number of rows (with halo): " << n_rows + 2 << std::endl;
    std::cout << "Number of columns (with halo): " << n_columns_ext
              << std::endl;
    std::cout << "Number of elements: " << n_rows_ext * n_columns_ext
              << std::endl;
    std::cout << "Number of iterations: " << n_iterations << std::endl;
    if (this->write_results) {
        std::cout << "Write solution: yes" << std::endl;
        std::cout << "Number of snapshots: " << n_iterations / images_interval
                  << std::endl;
        std::cout << "HDF5 file: " << this->file << std::endl;
    } else {
        std::cout << "Write solution: no" << std::endl;
    }
}

void Parameters::parse(int argc, char *argv[]) {
    CLI::App app{"Gray-Scott equation solver implemented with Kokkos"};

    app.add_option("-n,--rows", this->n_rows, "Number of rows")
        ->capture_default_str();
    app.add_option("-m,--columns", this->n_columns, "Number of columns")
        ->capture_default_str();

    auto iterations = app.add_option("-i,--iterations", this->n_iterations,
                                     "Number of iterations")
                          ->capture_default_str();
    app.add_option("-t,--interval", this->images_interval,
                   "Number of iterations between two snapshots")
        ->capture_default_str();

    std::size_t n_snapshots;
    auto snapshots =
        app.add_option("-s,--snapshots", n_snapshots,
                       "Number of snapshots (will set the number of iterations "
                       "accordingly)")
            ->default_val(this->n_iterations / this->images_interval)
            ->excludes(iterations);

    app.add_flag("-d,--display", this->display_fields,
                 "Display fields on screen at the beginning and at the end "
                 "of the execution");

    app.add_flag("-N{false},--no-write{false}", this->write_results,
                 "Do not write result file on disk");

    app.add_option("-f,--file", this->file, "HDF5 file for the solution")
        ->capture_default_str();

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        exit(app.exit(e));
    }

    if (*snapshots) {
        this->n_iterations = n_snapshots * this->images_interval;
    }

    this->n_rows_ext = this->n_rows + 2;
    this->n_columns_ext = this->n_columns + 2;
}

void Parameters::check() const {
    if (this->n_rows < 3) {
        std::cerr << "Invalid number of rows: " << this->n_rows
                  << " (must be greater or equal to 3)" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (this->n_columns < 3) {
        std::cerr << "Invalid number of columns: " << this->n_columns
                  << " (must be greater or equal to 3)" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (this->n_iterations < this->images_interval) {
        std::cerr << "Number of iterations lower than number of iterations "
                     "between snapshots: "
                  << this->n_iterations << " < " << this->images_interval
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}
