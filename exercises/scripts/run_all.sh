#!/bin/bash

set -eu

if [[ $1 = "-h" ]]
then
    cat <<EOF
run_all.sh [-h] DIRECTORY

Run all Gray-Scott Kokkos implementations for the 10 × 10 case.

Positional arguments:
    DIRECTORY
        Path to the build directory where the implementation directories are
        present.

Optional arguments:
    -h
        Display this help message and exit.
EOF
    exit 0
fi

IMPLEMENTATIONS="sequential cpu cpu_simd gpu gpu_async gpu_async_more"

build_dir=$1
for implementation in $IMPLEMENTATIONS
do
        echo "------------- $build_dir/$implementation -------------"
        "$build_dir/$implementation/gray_scott_$implementation" -n 10 -m 10
done
