#!/bin/bash

BASE_REGISTRY=gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos

docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/test_kokkos_cpu_interactive:latest cpu/interactive
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/test_kokkos_cpu_interactive_jupyter:latest cpu/interactive_jupyter
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/test_kokkos_cpu_interactive_vscode:latest cpu/interactive_vscode
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/test_kokkos_cpu_learning_platform_code_server:latest cpu/learning_platform_code_server

docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/test_kokkos_gpu_interactive:latest gpu/interactive
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/test_kokkos_gpu_interactive_jupyter:latest gpu/interactive_jupyter
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/test_kokkos_gpu_interactive_vscode:latest gpu/interactive_vscode
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/test_kokkos_gpu_learning_platform_code_server:latest gpu/learning_platform_code_server

# docker buildx build --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_cpu_interactive:latest cpu/interactive
# docker buildx build --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_cpu_interactive_jupyter:latest cpu/interactive_jupyter
# docker buildx build --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_cpu_interactive_vscode:latest cpu/interactive_vscode
# docker buildx build --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_cpu_learning_platform_code_server:latest cpu/learning_platform_code_server

# docker buildx build --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_gpu_interactive:latest gpu/interactive
# docker buildx build --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_gpu_interactive_jupyter:latest gpu/interactive_jupyter
# docker buildx build --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_gpu_interactive_vscode:latest gpu/interactive_vscode
# docker buildx build --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_gpu_learning_platform_code_server:latest gpu/learning_platform_code_server
