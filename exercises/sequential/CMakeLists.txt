cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosSequential
    LANGUAGES CXX
    VERSION 0.1.0
)

set(NO_KOKKOS ON)
include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott_sequential)

target_sources(
    gray_scott_sequential
    PRIVATE
        gray_scott_sequential.cpp
)

target_link_libraries(
    gray_scott_sequential
    PRIVATE
        common
)

add_test(
    NAME gray_scott_sequential
    COMMAND gray_scott_sequential
)
