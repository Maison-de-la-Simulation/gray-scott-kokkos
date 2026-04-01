include_guard()

find_package(Kokkos 5.0.0 QUIET)

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

find_package(HDF5 1.10 REQUIRED COMPONENTS CXX)

find_package(CLI11 2.4 REQUIRED)
