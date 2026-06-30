cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUBase
    LANGUAGES CXX
    VERSION 0.1.1
)

include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott_gpu_base)

target_sources(
    gray_scott_gpu_base
    PRIVATE
        gray_scott_gpu_base.cpp
)

target_link_libraries(
    gray_scott_gpu_base
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_gpu_base
    COMMAND gray_scott_gpu_base
)
