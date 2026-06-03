add_library(common STATIC)

target_sources(
    common
    PUBLIC
        FILE_SET HEADERS
        FILES
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
        common
)

add_library(common_kokkos INTERFACE)

target_sources(
    common_kokkos
    INTERFACE
        FILE_SET HEADERS
        FILES
            helpers_kokkos.hpp
)

target_link_libraries(
    common_kokkos
    INTERFACE
        common
)
