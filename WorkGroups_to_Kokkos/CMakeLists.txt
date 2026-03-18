cmake_minimum_required(VERSION 3.5)

option(MODE "Mode to compile" CPU)

if (MODE STREQUAL "CPU")
 project(gray_scott_kokkos LANGUAGES CXX)
elseif(MODE STREQUAL "GPU")
 project(gray_scott_kokkos LANGUAGES CXX CUDA)
endif()

#set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

if (MODE STREQUAL "CPU")
 # Enable OpenMP
 find_package(OpenMP REQUIRED)
elseif(MODE STREQUAL "GPU")
 # Enable CUDA language
 enable_language(CUDA)
endif()

# Find Kokkos (requires KOKKOS_PATH or CMAKE_PREFIX_PATH set)
#find_package(Kokkos REQUIRED)
include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
)
FetchContent_MakeAvailable(Kokkos)

# Find HDF5 (C++ interface)
find_package(HDF5 REQUIRED COMPONENTS CXX)

add_executable(gray-scott.exe
    ../src/gray-scott.cpp
)

target_compile_options(gray-scott.exe PRIVATE -O3)

if (MODE STREQUAL "CPU")

 target_link_libraries(gray-scott.exe
     PRIVATE
     Kokkos::kokkos
     OpenMP::OpenMP_CXX
     HDF5::HDF5
 )

elseif(MODE STREQUAL "GPU")
	
 target_link_libraries(gray-scott.exe
     PRIVATE
     Kokkos::kokkos
     HDF5::HDF5
 )

 # Let Kokkos control CUDA flags
 set_target_properties(gray-scott.exe
     PROPERTIES
     CUDA_SEPARABLE_COMPILATION ON
 )

endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
