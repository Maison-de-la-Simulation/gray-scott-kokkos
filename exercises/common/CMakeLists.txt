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
        Kokkos::kokkos
        HDF5::HDF5
    PRIVATE
        CLI11::CLI11
)

add_library(common_pod INTERFACE)

target_sources(
    common_pod
    INTERFACE
        FILE_SET HEADERS
        FILES
            helpers_pod.hpp
)

target_link_libraries(
    common_pod
    INTERFACE
        HDF5::HDF5
        common
)
