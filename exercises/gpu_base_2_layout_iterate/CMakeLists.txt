cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUBase2LayoutIterate
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    message(WARNING "This implementation cannot run on GPU without modifications")
endif()

add_executable(gray_scott_gpu_base_2_layout_iterate)

target_sources(
    gray_scott_gpu_base_2_layout_iterate
    PRIVATE
        gray_scott_gpu_base_2_layout_iterate.cpp
)

target_link_libraries(
    gray_scott_gpu_base_2_layout_iterate
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_gpu_base_2_layout_iterate
    COMMAND gray_scott_gpu_base_2_layout_iterate
)
