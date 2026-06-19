#!/bin/bash

set -eu

print_usage () {
    cat <<EOF
check_outcome.sh [-h] PROGRAM

Check the Gray-Scott Kokkos implementations for the 10 × 10 case.

Positional arguments:
    PROGRAM
        Path to the program to check.

Optional arguments:
    -h
        Display this help message and exit.
EOF
}

print_success () {
    local case_name=$1
    local test_name=$2
    echo "Test $case_name $test_name $(tput setaf 2)$(tput bold)passed$(tput sgr0)"
}

print_failure () {
    local case_name=$1
    local test_name=$2
    echo "Test $case_name $test_name $(tput setaf 1)$(tput bold)failed$(tput sgr0)"
}

get_checksums () {
    local output=$1
    echo "$output" | grep "Checksum field" | grep -Eo "([0-9]+\.[0-9]+)"
}

# Source - https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ $1 = "-h" ]]
then
    print_usage
    exit 0
fi

program=$1
case_name=$(basename "$program")

# case parameters
size_n=10
size_m=10

outcome=true
output=$("$program" -n "$size_n" -m "$size_m")

# check checksums
checksums=$(get_checksums "$output")
checksums_expected="53.58
0.60"
if [[ "$checksums" = "$checksums_expected" ]]
then
    print_success "$case_name" checksums
else
    echo -e "Expected checksums:\n$checksums_expected"
    echo -e "Got checksums:\n$checksums"
    print_failure "$case_name" checksums

    outcome=false
fi

# check HDF5 file
tolerance=1e-6
file=gray_scott.h5
file_expected=$SCRIPT_DIR/data/gray_scott.h5
if h5diff -d $tolerance "$file" "$file_expected"
then
    print_success "$case_name" "HDF5 file"
else
    print_failure "$case_name" "HDF5 file"

    outcome=false
fi

# return 1 if there are errors
if ! $outcome
then
    exit 1
fi
