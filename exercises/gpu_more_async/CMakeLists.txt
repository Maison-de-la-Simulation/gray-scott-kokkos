cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUMoreAsync
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott_gpu_more_async)

target_sources(
    gray_scott_gpu_more_async
    PRIVATE
        gray_scott_gpu_more_async.cpp
)

target_link_libraries(
    gray_scott_gpu_more_async
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_gpu_more_async
    COMMAND gray_scott_gpu_more_async
)
