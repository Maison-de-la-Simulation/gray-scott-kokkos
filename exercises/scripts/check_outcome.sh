#!/bin/bash

set -eu

get_checksums () {
    local output=$1
    echo "$output" | grep "Checksum field" | grep -Eo "([0-9]+\.[0-9]+)"
}

if [[ $1 = "-h" ]]
then
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
    exit 0
fi

program=$1
test_name=$(basename "$program")

# case parameters
size_n=10
size_m=10
checksums_expected="97.58
0.60"

output=$("$program" -n "$size_n" -m "$size_m")
checksums=$(get_checksums "$output")

if [[ "$checksums" = "$checksums_expected" ]]
then
    echo "Test $test_name $(tput setaf 2)$(tput bold)passed$(tput sgr0)"
    exit 0
fi

echo "Test $test_name $(tput setaf 1)$(tput bold)failed$(tput sgr0)"
echo -e "Expected checksums:\n$checksums_expected"
echo -e "Got checksums:\n$checksums"
exit 1
