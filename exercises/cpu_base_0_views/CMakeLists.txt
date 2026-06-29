cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosCPUBase0Views
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    message(WARNING "This implementation cannot run on GPU without modifications")
endif()

add_executable(gray_scott_cpu_base_0_views)

target_sources(
    gray_scott_cpu_base_0_views
    PRIVATE
        gray_scott_cpu_base_0_views.cpp
)

target_link_libraries(
    gray_scott_cpu_base_0_views
    PRIVATE
        common
        Kokkos::kokkos
)

add_test(
    NAME gray_scott_cpu_base_0_views
    COMMAND gray_scott_cpu_base_0_views
)

set_tests_properties(
    gray_scott_cpu_base_0_views
    PROPERTIES
        SKIP_RETURN_CODE 134
)
