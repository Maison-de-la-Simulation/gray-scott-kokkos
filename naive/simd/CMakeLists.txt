cmake_minimum_required(VERSION 3.28)

project(
  GrayScottKokkosSIMD
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott_simd)

target_sources(
    gray_scott_simd
    PRIVATE
        gray_scott_simd.cpp
)

target_link_libraries(
    gray_scott_simd
    PRIVATE
        Kokkos::kokkos
        common
)
