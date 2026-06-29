cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUBase0SharedSpace
    LANGUAGES CXX
    VERSION 0.1.0
)

include(../dependencies.cmake)
include(../common.cmake)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    message(WARNING "This implementation cannot run on GPU without modifications")
endif()

add_executable(gray_scott_gpu_base_0_shared_space)

target_sources(
    gray_scott_gpu_base_0_shared_space
    PRIVATE
        gray_scott_gpu_base_0_shared_space.cpp
)

target_link_libraries(
    gray_scott_gpu_base_0_shared_space
    PRIVATE
        Kokkos::kokkos
        common
)

add_test(
    NAME gray_scott_gpu_base_0_shared_space
    COMMAND gray_scott_gpu_base_0_shared_space
)

set_tests_properties(
    gray_scott_gpu_base_0_shared_space
    PROPERTIES
        SKIP_RETURN_CODE 1
)
