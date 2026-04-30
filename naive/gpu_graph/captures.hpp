#pragma once

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
template <typename DstViewType, typename SrcViewType>
struct SynchronizeDeviceToDevice {
    DstViewType dst;
    SrcViewType src;

    void operator()(const Kokkos::Cuda &exec) const {
        cudaMemcpyAsync(dst.data(), src.data(),
                        src.size() * sizeof(typename SrcViewType::value_type),
                        cudaMemcpyDeviceToDevice, exec.cuda_stream());
    }
};

template <typename DstViewType, typename SrcViewType>
struct SynchronizeDeviceToHost {
    DstViewType dst;
    SrcViewType src;

    void operator()(const Kokkos::Cuda &exec) const {
        cudaMemcpyAsync(dst.data(), src.data(),
                        src.size() * sizeof(typename SrcViewType::value_type),
                        cudaMemcpyDeviceToHost, exec.cuda_stream());
    }
};
#elif defined(KOKKOS_ENABLE_HIP)
#error "not implemented"
#endif
