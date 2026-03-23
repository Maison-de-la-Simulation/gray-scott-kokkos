#include <H5Cpp.h>
#include <H5DataSet.h>
#include <H5DataSpace.h>
#include <H5File.h>
#include <H5PredType.h>

#include <Kokkos_Core.hpp>
#include <filesystem>

namespace constants {

constexpr double kill_rate{0.054};
constexpr double feed_rate{0.014};
constexpr double dt{1.0};
constexpr double diffusion_rate_u{0.1};
constexpr double diffusion_rate_v{0.05};

// data type in views
const auto real_hdf5{H5::PredType::NATIVE_DOUBLE};

}  // namespace constants

// views type
using View = Kokkos::View<double **>;

/**
 * @brief HDF5 file writer.
 */
class OutputWriter {
    H5::DataSpace space_2d;
    H5::DataSpace space_3d;
    H5::DataSet dataset;
    H5::H5File file;
    hsize_t current_image_id{0};

   public:
    /**
     * @brief Prepare the output file in HDF5 format.
     * @param filename Name of the file.
     * @param n_images Number of images to store.
     * @param field Field that will be outputed.
     */
    void prepare(const char *filename, const int n_images, const View &field) {
        const int n_rows = field.extent(0);
        const int n_columns = field.extent(1);

        // add attributes on how to manipulate the file
        H5::FileAccPropList fapl;
        fapl.setFcloseDegree(H5F_CLOSE_STRONG);  // Ensure immediate flush
        fapl.setCache(0, 0, 0, 0.0);  // Optional: Set chunk cache size to 0

        // create file
        const std::filesystem::path filepath =
            std::filesystem::current_path() / filename;
        this->file = H5::H5File(filepath, H5F_ACC_TRUNC,
                                H5::FileCreatPropList::DEFAULT, fapl);

        // create spaces
        hsize_t dims_3d[3] = {static_cast<hsize_t>(n_images + 1),
                              static_cast<hsize_t>(n_rows),
                              static_cast<hsize_t>(n_columns)};
        hsize_t dims_2d[2] = {static_cast<hsize_t>(n_rows),
                              static_cast<hsize_t>(n_columns)};
        this->space_3d = H5::DataSpace(3, dims_3d);
        this->space_2d = H5::DataSpace(2, dims_2d);

        // create dataset
        this->dataset =
            file.createDataSet("matrix", constants::real_hdf5, space_3d);
    }

    /**
     * @brief Write an image.
     * @param field The field to output.
     */
    void write(const View &field) {
        Kokkos::printf("Writing image %d\n", this->current_image_id);

        // set the amount of data to write
        hsize_t start[3]{this->current_image_id, 0, 0};
        hsize_t count[3]{1, static_cast<hsize_t>(field.extent(0)),
                         static_cast<hsize_t>(field.extent(1))};
        this->space_3d.selectHyperslab(H5S_SELECT_SET, count, start);

        // write data to file
        this->dataset.write(field.data(), constants::real_hdf5, this->space_2d,
                            this->space_3d);

        // update current image
        this->current_image_id++;
    }
};

/**
 * @brief Display a view on screen.
 * @param field Field to display.
 * @param iteration Current iteration number.
 */
void print_field(const View &field, const int iteration) {
    Kokkos::printf("field %s at iteration %d:\n", field.label().c_str(),
                   iteration);
    for (int i = 0; i < field.extent(0); i++) {
        for (int j = 0; j < field.extent(1); j++) {
            Kokkos::printf("%3.2f ", field(i, j));
        }
        Kokkos::printf("\n");
    }
}

/**
 * @brief Compute and print the checksum of an array.
 * @param field Field to compute the checksum of.
 * @param iteration Current iteration.
 * @return Checksum value.
 */
View::value_type print_checksum(const View &field, const int iteration) {
    View::value_type checksum;
    Kokkos::parallel_reduce(
        "check fields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {field.extent(0), field.extent(1)}),
        KOKKOS_LAMBDA(const int i, const int j,
                      View::value_type &checksum_local) {
            checksum_local += field(i, j);
        },
        checksum);

    Kokkos::printf("checksum field %s at iteration %d: %3.2f\n",
                   field.label().c_str(), iteration, checksum);

    return checksum;
}

/**
 * @brief Add a drop at the center of the fields.
 * @param u U field.
 * @param v V field.
 */
void add_drop(const View &u, const View &v) {
    const int n_rows_ext = u.extent(0);
    const int n_columns_ext = u.extent(1);

    // find drop location
    // central cell + 1
    const int i_center = n_rows_ext / 2;
    const int j_center = n_columns_ext / 2;
    const int i_drop_first = i_center - 1;
    const int i_drop_last = i_center + 1;
    const int j_drop_first = j_center - 1;
    const int j_drop_last = j_center + 1;

    Kokkos::parallel_for(
        "add drop",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i_drop_first, j_drop_first},
                                               {i_drop_last, j_drop_last}),
        KOKKOS_LAMBDA(const int i, const int j) {
            u(i, j) = 0;
            v(i, j) = 1;
        });
}

/**
 * @brief Compute the Gray-Scott equation for one iteration.
 * @param u U field.
 * @param v V field.
 */
void compute(const View &u, const View &v, const View &u_temp,
             const View &v_temp) {
    const int n_rows_ext = u.extent(0);
    const int n_columns_ext = u.extent(1);

    Kokkos::parallel_for(
        "compute",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {1, 1}, {n_rows_ext - 1, n_columns_ext - 1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            double u_full = 0;
            double v_full = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    u_full += u(i + k, j + l) - u(i, j);
                    v_full += v(i + k, j + l) - v(i, j);
                }
            }

            const double uvv = u(i, j) * u(i, j) * v(i, j);

            const double u_delta = constants::diffusion_rate_u * u_full - uvv +
                                   constants::feed_rate * (1 - u(i, j));
            const double v_delta =
                constants::diffusion_rate_v * v_full + uvv -
                (constants::feed_rate + constants::kill_rate) * v(i, j);

            u_temp(i, j) = u(i, j) + u_delta * constants::dt;
            v_temp(i, j) = v(i, j) + v_delta * constants::dt;
        });
}

void describe(const int n_rows, const int n_columns, const int n_iterations,
              const int images_interval) {
    Kokkos::printf("Number of rows: %d\n", n_rows);
    Kokkos::printf("Number of columns: %d\n", n_columns);
    Kokkos::printf("Number of rows with halo: %d\n", n_rows + 2);
    Kokkos::printf("Number of columns with halo: %d\n", n_columns + 2);
    Kokkos::printf("Number of elements: %d\n", (n_rows + 2) * (n_columns + 2));
    Kokkos::printf("Number of iterations: %d\n", n_iterations);
    Kokkos::printf("Number of images: %d\n", n_iterations / images_interval);
}

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    // problem size
    const int n_rows = 10;
    const int n_columns = 10;

    // iterations
    const int n_iterations = 100;
    const int images_interval = 10;

    // problem size including ghost cells
    const int n_rows_ext = n_rows + 2;
    const int n_columns_ext = n_columns + 2;

    // describe problem
    describe(n_rows, n_columns, n_iterations, images_interval);

    // fields (with halo)
    View u("u", n_rows_ext, n_columns_ext);
    View v("v", n_rows_ext, n_columns_ext);

    // create writer
    OutputWriter writer;
    writer.prepare("gray_scott.h5", n_iterations / images_interval, v);

    // initialize fields
    Kokkos::deep_copy(u, 1);
    Kokkos::deep_copy(v, 0);

    // add a drop at the center of the domain
    add_drop(u, v);

#ifdef ENABLE_PRINT
    // print init
    print_field(u, 0);
    print_field(v, 0);
#endif

    // write init
    writer.write(v);

    // temporary fields (with halo)
    View u_temp("u_temp", n_rows_ext, n_columns_ext);
    View v_temp("v_temp", n_rows_ext, n_columns_ext);

    // time loop
    for (int iteration = 0; iteration < n_iterations; iteration++) {
        compute(u, v, u_temp, v_temp);
        Kokkos::kokkos_swap(u, u_temp);
        Kokkos::kokkos_swap(v, v_temp);

        if (iteration % images_interval == 0) {
            writer.write(v);
        }
    }

    // checksum
    print_checksum(u, n_iterations);
    print_checksum(v, n_iterations);

#ifdef ENABLE_PRINT
    // print init
    print_field(u, n_iterations);
    print_field(v, n_iterations);
#endif
}
