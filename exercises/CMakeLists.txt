cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkos
    LANGUAGES CXX
    VERSION 0.1.0
)

option(ENABLE_TESTS "Enable to test the implementations")

if(ENABLE_TESTS)
    enable_testing()

    # TODO change to TEST_LAUNCHER for CMake >= 3.29
    set(CMAKE_CROSSCOMPILING_EMULATOR "${PROJECT_SOURCE_DIR}/scripts/check_outcome.sh")
endif()

include(dependencies.cmake)
include(common.cmake)

add_subdirectory(hello_world)
add_subdirectory(sequential)
add_subdirectory(cpu_base_0_views)
add_subdirectory(cpu_base_1_parallel_for)
add_subdirectory(cpu_base)
add_subdirectory(cpu_more_simd)
add_subdirectory(gpu_base_0_shared_space)
add_subdirectory(gpu_base_1_fences)
add_subdirectory(gpu_base_2_layout_iterate)
add_subdirectory(gpu_base)
add_subdirectory(gpu_more_async_0_write)
add_subdirectory(gpu_more_async)

message(
    WARNING
    "You are about to perform a top-level build, which is usually done by "
    "developers or by the CI."
    "\n"
    "If you are doing the exercises while following the course, you should "
    "build each implementation folder separately instead."
)
