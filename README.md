# Gray-Scott School Kokkos

[![Exercises](https://github.com/Maison-de-la-Simulation/gray-scott-kokkos/actions/workflows/exercises.yml/badge.svg)](https://github.com/Maison-de-la-Simulation/gray-scott-kokkos/actions/workflows/exercises.yml)
[![Courses](https://github.com/Maison-de-la-Simulation/gray-scott-kokkos/actions/workflows/courses.yml/badge.svg)](https://github.com/Maison-de-la-Simulation/gray-scott-kokkos/actions/workflows/courses.yml)
[![Images](https://github.com/Maison-de-la-Simulation/gray-scott-kokkos/actions/workflows/docker.yml/badge.svg)](https://github.com/Maison-de-la-Simulation/gray-scott-kokkos/actions/workflows/docker.yml)

## Presentation

This repository contains the Kokkos courses and exercises for the Gray-Scott School 2026.

## Repository structure

- The `courses` folder contains the courses that are presented online (not present in the images):
  - The CPU course is `kokkos_cpu`;
  - The GPU course is `kokkos_gpu`.
- The `exercises` folder contains the exercises associated to the course, based on the Gray-Scott equation:
  - The `common` folder contains common files for all implementations;
  - The `hello_world` folder contains a simple test code to check the installation of Kokkos is successful;
  - The `sequential` folder contains the sequential implementation of the Gray-Scott equation;
  - The `cpu` folder contains the Kokkos CPU implementation (note it cannot be run with a GPU backend);
  - The `cpu_simd` folder contains the Kokkos CPU implementation using SIMD (note it cannot be run with a GPU backend);
  - The `gpu` folder contains the Kokkos GPU implementation;
  - The `gpu_async` folder contains the Kokkos GPU implementation with asynchronous writing of the results;
  - The `gpu_async_more` folder contains the Kokkos GPU implementation with asynchronous data synchronization and writing of the results;
  - The `scripts` folder contains some tools to run the built implementations:
    - `run_all.sh` runs all known implementations binaries given a build directory (useful for a top-level build only);
    - `check_outcome.sh` run an implementation binary for the 10 × 10 case and check the checksums;
- The `docker` folder contains the Dockerfiles for the different editors (not present in the images):
    - The `cpu` folder contains the CPU Dockerfiles;
    - The `gpu` folder contains the GPU Dockerfiles for NVIDIA GPUs.

## Courses

In order to build the sources, a LaTeX distribution is required.
You are encouraged to use the PDF files of the latest release if you do not have a workable LaTeX environment.

Courses are built as following:

```sh
cd courses
make
```

## Exercises

The exercises have a common `CMakeLists.txt` that allows to build all implementations at the same time.
You can also build each implementation independently.

### Configuration

Dependencies to build the exercises are:

- CMake 3.28 or more recent;
- Kokkos 5.1.1 or more recent;
- HDF5 1.10 with C++ bindings or more recent;
- CLI11 2.4 or more recent.

By default, you have to manage the installation of the dependencies by yourself:

```sh
cmake -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DKokkos_ROOT=${path_to_kokkos} \
          -DHDF5_ROOT=${path_to_hdf5} \
          -DCLI11_ROOT=${path_to_cli11}
```

For the Gray-Scott School, a Docker image is provided with most dependencies installed, with the exception of Kokkos, which you have to build and install by yourself.

You can also use the `ENABLE_DOWNLOAD_FALLBACK` flag to allow CMake to download and build the dependencies itself:

```sh
cmake -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_DOWNLOAD_FALLBACK=ON \
          -DHDF5_ROOT=${path_to_hdf5} \
          ${kokkos_parameters} \
          ${cli11_parameters}
```

Note that HDF5 cannot be handled this way, however.
You can still install HDF5 with your system package manager.

### Compilation

```sh
cmake --build build --parallel ${number_of_jobs}
```

### Cluster configuration

#### Jean-Zay

Here is the environment to execute the code in the Jean-Zay supercomputer:

```
module purge
module load cuda/12.6.3
module load cmake/3.31.4
module load git/2.53.0
module load libtool/2.4.6
```

When compiling Kokkos, you have to specify the GPU architecture among its flags:

```sh
cmake -B build_kokkos \
          -DKokkos_ARCH_VOLTA70=ON \
          ${other_kokkos_flags}
```

### Doing the exercises

You should duplicate the `sequential` directory to start your work.
Then rename the template CMake list file and the C++ source file:

```sh
cp -r sequential my_gs
cd my_gs
mv CMakeLists.txt{.sample,}
mv gray_scott{_sequential,}.cpp
```

## Images

The following Docker images can be used:

- CPU
  - `gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos/kokkos_cpu_interactive:latest`
  - `gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos/kokkos_cpu_interactive_jupyter:latest`
  - `gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos/kokkos_cpu_interactive_vscode:latest`
  - `gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos/kokkos_cpu_learning_platform_code_server:latest`
- GPU
  - `gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos/kokkos_gpu_interactive:latest`
  - `gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos/kokkos_gpu_interactive_jupyter:latest`
  - `gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos/kokkos_gpu_interactive_vscode:latest`
  - `gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos/kokkos_gpu_learning_platform_code_server:latest`
