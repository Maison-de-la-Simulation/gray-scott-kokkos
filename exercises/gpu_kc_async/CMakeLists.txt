cmake_minimum_required(VERSION 3.28)

project(
    GrayScottKokkosGPUMoreAsync
    LANGUAGES CXX
    VERSION 0.1.2
)

include(../dependencies.cmake)
include(../common.cmake)

find_package(MPI REQUIRED)
find_package(KokkosComm REQUIRED)

add_executable(gray_scott_kcomm)

target_sources(
    gray_scott_kcomm
    PRIVATE
        gray_scott_kcomm.cpp
)

target_link_libraries(
    gray_scott_kcomm
    PRIVATE
        Kokkos::kokkos
        common
        MPI::MPI_CXX
        KokkosComm::KokkosComm
)

target_compile_definitions(
    gray_scott_kcomm
    PRIVATE
        $<IF:$<BOOL:${ENABLE_DOUBLE}>,PRECISION=64,PRECISION=32>
)

add_test(
    NAME gray_scott_kcomm
    COMMAND gray_scott_kcomm
)
