# Kokkos GPU commands

## Moving to a GPU backend

### Initialization

```sh
REPO=docker://gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos

# pick one
IMAGE=kokkos_gpu_interactive
IMAGE=kokkos_gpu_interactive_jupyter
IMAGE=kokkos_gpu_interactive_vscode
IMAGE=kokkos_gpu_learning_platform_code_server

apptainer pull $REPO/$IMAGE:latest

srun ... --pty /bin/bash

apptainer run --nv ${IMAGE}_latest.sif

# if you didn't attend last week session
cp -r /home/jovyan/Examples $HOME/gray_scott_kokkos

cd $HOME/gray_scott_kokkos/exercises
```

### Build Kokkos for a GPU backend

```sh
cmake \
    -S /src/kokkos \
    -B build/cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_XXX=ON

cmake \
    --build build/cuda \
    --parallel 4

cmake \
    --install build/cuda \
    --prefix $HOME/opt/kokkos/cuda
```

### From the "hello_world" folder

```sh
cmake \
    -B build/cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ROOT=$HOME/opt/kokkos/cuda

cmake \
    --build build/cuda
```

### From last week worktree or the "cpu_base" folder

```sh
cmake \
    -B build/cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ROOT=$HOME/opt/kokkos/cuda

cmake \
    --build build/cuda
```

## Memory spaces

### From your worktree or the "cpu_base" folder

```c++
using View =
    Kokkos::View<
        real**,
        Kokkos::SharedSpace
    >;
```

## Fencing

### From your worktree or the "gpu_base_0_shared_space" folder

```c++
Kokkos::parallel_for(
    "my_loop",
    N,
    KOKKOS_LAMBDA (int i) {
        // ...
    }
);

Kokkos::fence("my_loop fence");
```

## Layout and iteration order

### From your worktree or the "gpu_base_1_fences" folder

```c++
using View =
    Kokkos::View<
        real**,
        Kokkos::LayoutRight
    >;

// ...

Kokkos::parallel_for(
    // ...
    Kokkos::MDrangePolicy<
        Kokkos::Range<
            2,
            Kokkos::Iterate::Default,
            Kokkos::Iterate::Right
        >
    >(/* ... */),
    // ...
);
```

## Kokkos tools

### Build Kokkos-Tools

```sh
cmake \
    -S /src/kokkos-tools \
    -B build/cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ROOT=$HOME/opt/kokkos/cuda

cmake \
    --build build/cuda \
    --parallel 4

cmake \
    --install build/cuda \
    --prefix $HOME/opt/kokkos-tools/cuda

export PATH=$HOME/opt/kokkos-tools/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/opt/kokkos-tools/cuda/lib:$LD_LIBRARY_PATH
```

### From your worktree or the "gpu_base_2_layout_iterate" folder

```c++
Kokkos::Profiling::pushRegion(
    "region name"
);

// ...

Kokkos::Profiling::popRegion();
```

```sh
nsys profile \
    -o report \
    ./my_program \
        --kokkos-tools-libs=libkp_nvtx_connector.so

nsys-ui report.nsys-rep
```

## Explicit memory management

### From your worktree or the "gpu_base_2_layout_iterate" folder

```sh
auto host_view =
    Kokkos::create_mirror_view(
        device_view
    );

// ...

Kokkos::deep_copy(
    device_view,
    host_view
);
```

## Asynchronous execution

### (Re-)Initialization

```sh
# pick one
IMAGE=kokkos_gpu_interactive
IMAGE=kokkos_gpu_interactive_jupyter
IMAGE=kokkos_gpu_interactive_vscode
IMAGE=kokkos_gpu_learning_platform_code_server

srun ... --pty /bin/bash

apptainer run --nv ${IMAGE}_latest.sif

cd $HOME/gray_scott_kokkos/exercises
```
