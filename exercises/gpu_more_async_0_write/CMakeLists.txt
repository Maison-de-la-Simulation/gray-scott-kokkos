cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUMoreAsync0Write
    LANGUAGES CXX
    VERSION 0.1.2
)

include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott_gpu_more_async_0_write)

target_sources(
    gray_scott_gpu_more_async_0_write
    PRIVATE
        gray_scott_gpu_more_async_0_write.cpp
)

target_link_libraries(
    gray_scott_gpu_more_async_0_write
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_gpu_more_async_0_write
    COMMAND gray_scott_gpu_more_async_0_write
)
