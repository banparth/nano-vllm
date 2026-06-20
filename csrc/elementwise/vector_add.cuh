#pragma once
#include <cstdint>

#include <cuda_runtime.h>

namespace nanovllm {

// Host-callable launcher for the vector-add kernel. Defined and explicitly
// instantiated in vector_add.cu; called from vector_add.cpp. Keeping the launch
// (and a torch-free kernel) in the .cu lets clangd parse it in native CUDA mode,
// while the torch glue lives in the .cpp (parsed as ordinary C++).
template <typename scalar_t>
void vector_add_launch(const scalar_t* a, const scalar_t* b, scalar_t* out,
                       int64_t n, cudaStream_t stream);

}  // namespace nanovllm
