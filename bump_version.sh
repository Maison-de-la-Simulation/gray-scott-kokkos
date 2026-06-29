#!/bin/bash

set -eu

usage () {
    cat <<EOF
bump_version.sh [-y] [-f] [-h] VERSION

Bump the project to a new version.

Positional arguments:
    VERSION
        New version to bump the project to.

Optional arguments:
    -y
        Enable dry run: do not create any commit.
    -f
        Force execution even if the repository is not clean.
    -h
        Display this help message and exit.
EOF
}

get_cmake_files () {
    find . -type f -name CMakeLists.txt
}

update_cmake_files () {
    local version=$1
    shift 1
    local files=$*
    # shellcheck disable=SC2086 # allow to pass the list of files
    sed -i \
        -e "s/^    VERSION .*$/    VERSION $version/" \
        $files
}

dry_run=false
force_run=false

# parse optional arguments
while getopts ":hyf" option
do
    case $option in
        h)
            usage
            exit 0
            ;;
        y)
            dry_run=true
            ;;
        f)
            force_run=true
            ;;
        *)
            echo "Unknown option" >&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

# check positional argument
if [[ -z ${1+x} ]]
then
    echo "Missing version" >&2
    usage
    exit 2
fi

version=$1

# check repository is clean
if ! $force_run
then
    if [[ -n "$(git status --untracked-files=no --porcelain)" ]]
    then
        echo "Repository not clean" >&2
        echo "Please commit or stash your current changes" >&2
        exit 2
    fi
fi

# update version in files
echo "Bumping CMakeLists.txt files to version $version"
cmake_files=$(get_cmake_files)
# shellcheck disable=SC2086 # allow to pass the list of files
update_cmake_files "$version" $cmake_files

if ! $dry_run
then
    echo "Creating commit and tag for version $version"
    # shellcheck disable=SC2086 # allow to pass the list of files
    git add $cmake_files
    git commit -m "Version $version"
    git tag "$version"
fi
