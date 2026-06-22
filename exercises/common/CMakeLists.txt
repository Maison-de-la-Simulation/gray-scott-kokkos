add_library(common STATIC)

target_sources(
    common
    PUBLIC
        FILE_SET HEADERS
        FILES
            helpers.hpp
            output_writer.hpp
            parameters.hpp
    PRIVATE
        parameters.cpp
)

target_link_libraries(
    common
    PUBLIC
        HDF5::HDF5
    PRIVATE
        CLI11::CLI11
)
