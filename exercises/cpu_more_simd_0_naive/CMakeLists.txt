cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosCPUSIMDSimple
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    message(WARNING "This implementation cannot run on GPU without modifications")
endif()

add_executable(gray_scott_cpu_more_simd_0_naive)

target_sources(
    gray_scott_cpu_more_simd_0_naive
    PRIVATE
        gray_scott_cpu_more_simd_0_naive.cpp
)

target_link_libraries(
    gray_scott_cpu_more_simd_0_naive
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_cpu_more_simd_0_naive
    COMMAND gray_scott_cpu_more_simd_0_naive
)

set_tests_properties(
    gray_scott_cpu_more_simd_0_naive
    PROPERTIES
        SKIP_RETURN_CODE 139
)
