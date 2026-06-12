include_guard()

include(FetchContent)

# option to prevent to download dependencies
option(ENABLE_DOWNLOAD_FALLBACK "Enable to download dependencies if they are not found" OFF)

##
# Kokkos
#

if(NOT NO_KOKKOS)
    if(NOT ENABLE_DOWNLOAD_FALLBACK)
        find_package(Kokkos 5.1.1 REQUIRED)
    else()
        find_package(Kokkos 5.1.1 QUIET)

        # download Kokkos if not found
        if(NOT Kokkos_FOUND)
            message(STATUS "Downloading Kokkos from the repository")
            FetchContent_Declare(
                Kokkos
                URL https://github.com/kokkos/kokkos/releases/download/5.1.1/kokkos-5.1.1.zip
                URL_HASH SHA256=1de49ac660b5ccc619e22f5b3215194b1ea2ac005aa6130ab28d300ca8f9e5ff
            )
            FetchContent_MakeAvailable(Kokkos)
        endif()
    endif()
endif()

##
# HDF5
#

if(NOT NO_HDF5)
    find_package(HDF5 1.10 REQUIRED COMPONENTS CXX)
endif()

##
# CLI11
#

if(NOT NO_CLI11)
    if(NOT ENABLE_DOWNLOAD_FALLBACK)
        find_package(CLI11 2.4 REQUIRED)
    else()
        find_package(CLI11 2.4 QUIET)

        # download CLI11 if not found
        if(NOT CLI11_FOUND)
            message(STATUS "Downloading CLI11 from the repository")
            FetchContent_Declare(
                cli11
                URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.zip
                URL_HASH SHA256=43e650d5e1a3acaaf419d1e61a81f77b408d0696f472be0599ddf877d40984b0
            )
            FetchContent_MakeAvailable(cli11)
        endif()
    endif()
endif()
