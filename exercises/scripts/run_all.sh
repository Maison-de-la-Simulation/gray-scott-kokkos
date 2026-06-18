#!/bin/bash

set -eu

print_usage () {
    cat <<EOF
run_all.sh [-h] DIRECTORY [ARGS]

Run all Gray-Scott Kokkos implementation executables in their directory.

Positional arguments:
    DIRECTORY
        Path to the build directory where the implementation directories are
        present.
    ARGS
        Arguments directly passed to the implementation executables.

Optional arguments:
    -h
        Display this help message and exit.
EOF
}

if [[ $1 = "-h" ]]
then
    print_usage
    exit 0
fi

IMPLEMENTATIONS="sequential cpu cpu_simd gpu gpu_async gpu_async_more"

build_dir=$1
shift 1
arguments=$*

for implementation in $IMPLEMENTATIONS
do
        echo "------------- $build_dir/$implementation -------------"
        echo "Arguments: $arguments"
        (
            cd "$build_dir"
            # shellcheck disable=SC2086 # we want the arguments to be passed with spaces
            "$implementation/gray_scott_$implementation" $arguments || {
                echo "Error when running $build_dir/$implementation $arguments"
            }
        ) 
done
