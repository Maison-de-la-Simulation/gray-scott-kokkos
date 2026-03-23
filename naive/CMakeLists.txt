cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosNaiveImplementation
    LANGUAGES CXX
    VERSION 0.1.0
)

find_package(Kokkos REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

option(ENABLE_PRINT "Enable printing arrays on screen (debug purpose)" OFF)

add_executable(naive)

target_sources(
    naive
    PRIVATE
        naive.cpp
)

target_link_libraries(
    naive
    PRIVATE
        Kokkos::kokkos
        HDF5::HDF5
)

target_compile_definitions(
    naive
    PRIVATE
        $<$<BOOL:${ENABLE_PRINT}>:ENABLE_PRINT>
)
