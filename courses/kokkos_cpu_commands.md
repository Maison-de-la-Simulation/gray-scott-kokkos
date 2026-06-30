# Kokkos CPU commands

## Docker images

### Initialization

```sh
REPO=docker://gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos

# pick one
IMAGE=kokkos_cpu_interactive
IMAGE=kokkos_cpu_interactive_jupyter
IMAGE=kokkos_cpu_interactive_vscode
IMAGE=kokkos_cpu_learning_platform_code_server

apptainer pull $REPO/$IMAGE:latest

srun ... --pty /bin/bash

apptainer run ${IMAGE}_latest.sif

cp -r /home/jovyan/Examples $HOME/gray_scott_kokkos

cd $HOME/gray_scott_kokkos/exercises
```

## Gray-Scott sequential implementation

### From the "sequential" folder

```sh
cmake \
    -B build/serial \
    -DCMAKE_BUILD_TYPE=Release

cmake \
    --build build/serial
```

## Starting with the serial backend

### Build Kokkos for the serial backend

```sh
cmake \
    -S /src/kokkos \
    -B build/serial \
    -DCMAKE_BUILD_TYPE=Release

cmake \
    --build build/serial \
    --parallel 4

cmake \
    --install build/serial \
    --prefix $HOME/opt/kokkos/serial
```

## First steps with Kokkos

### From the "hello_world" folder

```sh
cmake \
    -B build/serial \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ROOT=$HOME/opt/kokkos/serial

cmake \
    --build build/serial
```

## Data containers

### From the "gray_scott" folder (aka the worktree)

```c++
#include <Kokkos_Core.hpp>

// ...

using View =
    Kokkos::View<double **>;

View a("A", N, M);
a(0, 1) = 1;
```

## Parallel loops

### From your worktree or the "cpu_base_0_views" folder

```c++
Kokkos::parallel_for(
    "my_loop",
    Kokkos::MDRangePolicy<
        Kokkos::Rank<2>
    >(
        {0, 0},
        {Nx, Ny}
    ),
    KOKKOS_LAMBDA (int i, int j) {
        A(i, j) = i + j;
    }
);
```

## Parallel reductions

### From your worktree or the "cpu_base_1_parallel_for" folder

```c++
double result;
Kokkos::parallel_reduce(
    "my_reduction",
    N,
    KOKKOS_LAMBDA (
        const int i,
        double& local_result
    ) {
        local_result += A(i);
    },
    result
);
```

## Moving to a parallel CPU backend

### (Re-)Initialization

```sh
# pick one
IMAGE=kokkos_cpu_interactive
IMAGE=kokkos_cpu_interactive_jupyter
IMAGE=kokkos_cpu_interactive_vscode
IMAGE=kokkos_cpu_learning_platform_code_server

srun ... --pty /bin/bash

apptainer run ${IMAGE}_latest.sif

cd $HOME/gray_scott_kokkos/exercises
```

### Build Kokkos for the OpenMP backend

```sh
cmake \
    -S /src/kokkos \
    -B build/openmp \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ARCH_NATIVE=ON

cmake \
    --build build/openmp \
    --parallel 4

cmake \
    --install build/openmp \
    --prefix $HOME/opt/kokkos/openmp
```

### From the "hello_world" folder

```sh
cmake \
    -B build/serial \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ROOT=$HOME/opt/kokkos/openmp

cmake \
    --build build/openmp
```

### From your worktree or the "cpu_base" folder

```sh
cmake \
    -B build/openmp \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ROOT=$HOME/opt/kokkos/openmp

cmake \
    --build build/openmp
```

## SIMD and vectorization

### From your worktree or the "cpu_base" folder

```c++
#include <Kokkos_SIMD.hpp>

// ...

namespace KE =
    Kokkos::Experimental;

using simd_type = KE::simd<real>;

View a("A", N, M);
simd_type a_simd(&a(0, 0));
```
