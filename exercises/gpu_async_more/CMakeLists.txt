cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUAsyncMore
    LANGUAGES CXX
    VERSION 0.1.1
)

include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott_gpu_async_more)

target_sources(
    gray_scott_gpu_async_more
    PRIVATE
        gray_scott_gpu_async_more.cpp
)

target_link_libraries(
    gray_scott_gpu_async_more
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_gpu_async_more
    COMMAND gray_scott_gpu_async_more
)
