cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosNaiveImplementation
    LANGUAGES CXX
    VERSION 0.1.0
)

##
# Dependencies
#

find_package(Kokkos 5.0.0 QUIET)

# download Kokkos if not found
if(NOT Kokkos_FOUND)
    include(FetchContent)
    message(STATUS "Downloading Kokkos from the repository")
    FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/releases/download/5.0.2/kokkos-5.0.2.zip
        URL_HASH SHA256=c4ec42b952a0c8474d370cf0aac025adeed81326c78c23ecdfc98c18818a575a
    )
    FetchContent_MakeAvailable(Kokkos)
endif()

find_package(HDF5 1.10 REQUIRED COMPONENTS CXX)

find_package(CLI11 2.4 REQUIRED)

##
# Libraries
#

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
            output_writer_pod.hpp
)

target_link_libraries(
    common_pod
    INTERFACE
        HDF5::HDF5
        common
)

##
# Executables
#

add_executable(naive_seq)

target_sources(
    naive_seq
    PRIVATE
        naive_seq.cpp
)

target_link_libraries(
    naive_seq
    PRIVATE
        common_pod
)

add_executable(naive_cpu)

target_sources(
    naive_cpu
    PRIVATE
        naive_cpu.cpp
)

target_link_libraries(
    naive_cpu
    PRIVATE
        Kokkos::kokkos
        common
)

add_executable(naive_gpu)

target_sources(
    naive_gpu
    PRIVATE
        naive_gpu.cpp
)

target_link_libraries(
    naive_gpu
    PRIVATE
        Kokkos::kokkos
        common
)
