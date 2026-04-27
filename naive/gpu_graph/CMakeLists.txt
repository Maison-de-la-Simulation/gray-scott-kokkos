cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUGraph
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott_gpu_graph)

target_sources(
    gray_scott_gpu_graph
    PRIVATE
        gray_scott_gpu_graph.cpp
)

target_link_libraries(
    gray_scott_gpu_graph
    PRIVATE
        Kokkos::kokkos
        common
)
