cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosCPUParallelReduce
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    message(WARNING "This implementation cannot run on GPU without modifications")
endif()

add_executable(gray_scott_cpu_parallel_reduce)

target_sources(
    gray_scott_cpu_parallel_reduce
    PRIVATE
        gray_scott_cpu_parallel_reduce.cpp
)

target_link_libraries(
    gray_scott_cpu_parallel_reduce
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_cpu_parallel_reduce
    COMMAND gray_scott_cpu_parallel_reduce
)

set_tests_properties(
    gray_scott_cpu_parallel_reduce
    PROPERTIES
        SKIP_RETURN_CODE 139
)
