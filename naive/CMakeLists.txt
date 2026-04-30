cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkos
    LANGUAGES CXX
    VERSION 0.1.0
)

include(dependencies.cmake)
include(common.cmake)

add_subdirectory(sequential)
add_subdirectory(cpu)
add_subdirectory(simd)
add_subdirectory(gpu)
add_subdirectory(gpu_async)
add_subdirectory(gpu_graph)
