cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosHelloWorld
    LANGUAGES CXX
    VERSION 0.1.0
)

set(NO_HDF5 ON)
set(NO_CLI11 ON)
include(../dependencies.cmake)

add_executable(gray_scott_hello)

target_sources(
    gray_scott_hello
    PRIVATE
        gray_scott_hello.cpp
)

target_link_libraries(
    gray_scott_hello
    PRIVATE
        Kokkos::kokkos
)
