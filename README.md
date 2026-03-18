
## Presentation

This a Kokkos adapted version of the existing demonstrator code make by David CHAMONT for the Gray Scott School [GrayScottSyclBench](https://gitlab.in2p3.fr/CodeursIntensifs/grayscott/GrayScottSyclBench). In this version we have adapted the original Sycl verion to Kokkos keeping the same code structure.  

## Repository structure

Inside of each of the benchmark version you will find the following files and folder.
- `src`      : C++ source
- `build-gpu`   : Files to build the gpu version
- `results`    : h5 output files
- `CmakeLists.txt` : CmakeList to create the make file

## Jean-Zay Enviroment

Here is the enviroment have been using to execute the code in the Jean-Zay supermachine.

```
module purge
module load cuda/12.6.3
module load cmake/3.31.4
module load git/2.53.0
module load libtool/2.4.6
module load nvidia-compilers/25.11

# Cuda path
export CUDA_HOME=/lustre/fshomisc/sys/common/nvidia/cuda/12.6.3/
export CUDA_LIB_PATH=$CUDA_HOME/lib64/stubs/
export CUDA_TOOLKIT_PATH=$CUDA_HOME
export HDF5_ROOT=/lustre/fswork/projects/idris/sos/ssos044/libraries/hdf5/install
```

##  Compilation and execution on Jean-Zay partitions

# CPU
To compile the code on CPU use the following cmake command.

```
cmake ../ -DMODE=CPU && make
```

To execute the code the following command. 

```
./gray-scott.exe CPU 1920 1080 1000 32
```

# GPU

To compile the code on GPU V100 Jean-Zay node use the following cmake command.

```
cmake ../ -DMODE=GPU -DKokkos_ENABLE_CUDA=ON -DKokkos_VOLTA100=ON && make
```

To execute the code the following command.

```
./gray-scott.exe GPU 1920 1080 1000 32
```
