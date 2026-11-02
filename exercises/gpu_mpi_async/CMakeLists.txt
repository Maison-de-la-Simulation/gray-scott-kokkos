cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUMoreAsync
    LANGUAGES CXX
    VERSION 0.1.2
)

include(../dependencies.cmake)
include(../common.cmake)

find_package(MPI REQUIRED)

add_executable(gray_scott_mpi)

target_sources(
    gray_scott_mpi
    PRIVATE
        gray_scott_mpi.cpp
)

target_link_libraries(
    gray_scott_mpi
    PRIVATE
        Kokkos::kokkos
        common
        MPI::MPI_CXX
)

target_compile_definitions(
    gray_scott_mpi
    PRIVATE
        $<IF:$<BOOL:${ENABLE_DOUBLE}>,PRECISION=64,PRECISION=32>
)

add_test(
    NAME gray_scott_mpi
    COMMAND gray_scott_mpi
)