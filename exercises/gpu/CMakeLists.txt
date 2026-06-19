cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPU
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott_gpu)

target_sources(
    gray_scott_gpu
    PRIVATE
        gray_scott_gpu.cpp
)

target_link_libraries(
    gray_scott_gpu
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_gpu
    COMMAND gray_scott_gpu
)
