#pragma once

#include <H5Cpp.h>
#include <H5DataSet.h>
#include <H5DataSpace.h>
#include <H5File.h>
#include <H5PredType.h>

#include <cstddef>
#include <iostream>
#include <memory>

/**
 * @brief HDF5 file writer.
 * @tparam real Type of data to store. This is required to set HDF5 variable
 * type.
 */
template <typename real>
class OutputWriter {
    H5::DataSpace space_2d;
    H5::DataSpace space_3d;
    H5::DataSet dataset;
    H5::H5File file;
    std::unique_ptr<H5::PredType> real_type;
    hsize_t current_image_id{0};
    std::size_t n_rows_ext{0};
    std::size_t n_columns_ext{0};

   public:
    /**
     * @brief Default constructor.
     */
    OutputWriter() = default;

    /**
     * @brief Create and prepare the output writer in HDF5 format.
     * @param filename Name of the file.
     * @param n_images Number of images to store.
     * @param field Field that will be outputed.
     * @see `prepare` method.
     */
    OutputWriter(const char *filename, const std::size_t n_images,
                 const std::size_t n_rows_ext,
                 const std::size_t n_columns_ext) {
        this->prepare(filename, n_images, n_rows_ext, n_columns_ext);
    }

    /**
     * @brief Prepare the output writer in HDF5 format.
     * @param filename Name of the file.
     * @param n_images Number of images to store.
     * @param field Field that will be outputed. Only the shape of the view is
     * used at this step.
     */
    void prepare(const char *filename, const std::size_t n_images,
                 const std::size_t n_rows_ext,
                 const std::size_t n_columns_ext) {
        this->n_rows_ext = n_rows_ext;
        this->n_columns_ext = n_columns_ext;

        // set real type dependenig on the given view
        static_assert(
            std::is_same_v<real, double> or std::is_same_v<real, float>,
            "Unexpected real type");
        if constexpr (std::is_same_v<real, double>) {
            this->real_type =
                std::make_unique<H5::PredType>(H5::PredType::NATIVE_DOUBLE);
        } else {
            this->real_type =
                std::make_unique<H5::PredType>(H5::PredType::NATIVE_FLOAT);
        }

        // add attributes on how to manipulate the file
        H5::FileAccPropList fapl;
        fapl.setFcloseDegree(H5F_CLOSE_STRONG);  // Ensure immediate flush
        fapl.setCache(0, 0, 0, 0.0);  // Optional: Set chunk cache size to 0

        // create file in current working directory
        this->file = H5::H5File(filename, H5F_ACC_TRUNC,
                                H5::FileCreatPropList::DEFAULT, fapl);

        // create spaces
        hsize_t dims_3d[3] = {
            static_cast<hsize_t>(n_rows_ext),
            static_cast<hsize_t>(n_columns_ext),
            static_cast<hsize_t>(n_images + 1),
        };
        hsize_t dims_2d[2] = {static_cast<hsize_t>(n_rows_ext),
                              static_cast<hsize_t>(n_columns_ext)};
        this->space_3d = H5::DataSpace(3, dims_3d);
        this->space_2d = H5::DataSpace(2, dims_2d);

        // create dataset
        this->dataset =
            file.createDataSet("matrix", *this->real_type, this->space_3d);
    }

    /**
     * @brief Write an image.
     * @param field The field to output.
     * @note `prepare` must have been called beforehand.
     */
    void write(real const *field) {
        // check `prepare` was called
        if (this->file.getId() == -1) {
            throw std::runtime_error("OutputWriter is not prepared");
        }

        std::cout << "Writing image " << this->current_image_id << std::endl;

        // set the amount of data to write
        hsize_t start[3]{0, 0, this->current_image_id};
        hsize_t count[3]{static_cast<hsize_t>(this->n_rows_ext),
                         static_cast<hsize_t>(this->n_columns_ext), 1};
        this->space_3d.selectHyperslab(H5S_SELECT_SET, count, start);

        // write data to file
        this->dataset.write(field, *this->real_type, this->space_2d,
                            this->space_3d);

        // update current image index
        this->current_image_id++;
    }
};
