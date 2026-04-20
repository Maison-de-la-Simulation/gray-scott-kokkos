include_guard()

##
# Kokkos
#

# option to prevent to download kokkos
option(ENABLE_KOKKOS_DOWNLOAD_FALLBACK "Enable to download Kokkos if it is not found" OFF)

if(NOT ENABLE_KOKKOS_DOWNLOAD_FALLBACK)
    # stop if Kokkos not found
    find_package(Kokkos 5.0 REQUIRED)
else()
    find_package(Kokkos 5.0 QUIET)
endif()

# download Kokkos if not found
if(NOT Kokkos_FOUND)
    include(FetchContent)
    message(STATUS "Downloading Kokkos from the repository")
    FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/releases/download/5.0.2/kokkos-5.0.2.zip
        URL_HASH SHA256=c4ec42b952a0c8474d370cf0aac025adeed81326c78c23ecdfc98c18818a575a
    )
    FetchContent_MakeAvailable(Kokkos)
endif()

##
# HDF5
#

find_package(HDF5 1.10 REQUIRED COMPONENTS CXX)

##
# CLI11
#

find_package(CLI11 2.4 REQUIRED)

##
# fmt
#

find_package(fmt 9.0 REQUIRED)
