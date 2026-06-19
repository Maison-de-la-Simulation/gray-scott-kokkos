#!/bin/bash

BASE_REGISTRY=gitlab-registry.in2p3.fr/thomas.padioleau/gray-scott-kokkos

cd ..

docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_cpu_interactive:latest -f docker/cpu/interactive/Dockerfile .
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_cpu_interactive_jupyter:latest -f docker/cpu/interactive_jupyter/Dockerfile .
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_cpu_interactive_vscode:latest -f docker/cpu/interactive_vscode/Dockerfile .
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_cpu_learning_platform_code_server:latest -f docker/cpu/learning_platform_code_server/Dockerfile .

docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_gpu_interactive:latest -f docker/gpu/interactive/Dockerfile .
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_gpu_interactive_jupyter:latest -f docker/gpu/interactive_jupyter/Dockerfile .
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_gpu_interactive_vscode:latest -f docker/gpu/interactive_vscode/Dockerfile .
docker buildx build --push --platform linux/amd64,linux/arm64 --tag $BASE_REGISTRY/kokkos_gpu_learning_platform_code_server:latest -f docker/gpu/learning_platform_code_server/Dockerfile .
