cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosCPU
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    message(WARNING "This implementation cannot run on GPU without modifications")
endif()

add_executable(gray_scott_cpu)

target_sources(
    gray_scott_cpu
    PRIVATE
        gray_scott_cpu.cpp
)

target_link_libraries(
    gray_scott_cpu
    PRIVATE
        Kokkos::kokkos
        common
)
