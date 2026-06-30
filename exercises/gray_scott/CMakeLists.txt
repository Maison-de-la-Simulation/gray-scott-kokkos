cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkos
    LANGUAGES CXX
    VERSION 0.1.1
)

include(../dependencies.cmake)
include(../common.cmake)

add_executable(gray_scott)

target_sources(
    gray_scott
    PRIVATE
        gray_scott.cpp
)

target_link_libraries(
    gray_scott
    PRIVATE
        common
        # TODO
)
