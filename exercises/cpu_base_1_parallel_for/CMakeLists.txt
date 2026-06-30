cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosCPUBase1ParallelFor
    LANGUAGES CXX
    VERSION 0.1.1
)

include(../dependencies.cmake)
include(../common.cmake)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    message(WARNING "This implementation cannot run on GPU without modifications")
endif()

add_executable(gray_scott_cpu_base_1_parallel_for)

target_sources(
    gray_scott_cpu_base_1_parallel_for
    PRIVATE
        gray_scott_cpu_base_1_parallel_for.cpp
)

target_link_libraries(
    gray_scott_cpu_base_1_parallel_for
    PRIVATE
        common
        Kokkos::kokkos
)

add_test(
    NAME gray_scott_cpu_base_1_parallel_for
    COMMAND gray_scott_cpu_base_1_parallel_for
)

set_tests_properties(
    gray_scott_cpu_base_1_parallel_for
    PROPERTIES
        SKIP_RETURN_CODE 139
)
