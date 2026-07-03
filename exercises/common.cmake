include_guard()

option(
    ENABLE_DOUBLE
    "Enable double precision for floating variables, instead of simple precision"
    ON
)

# add the common libs from top level or low level
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/common" "${CMAKE_CURRENT_BINARY_DIR}/common")
