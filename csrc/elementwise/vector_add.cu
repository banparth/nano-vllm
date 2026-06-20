#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "elementwise/vector_add.cuh"

namespace nanovllm {

template <typename scalar_t>
__global__ void vector_add_kernel(const scalar_t* a, const scalar_t* b,
                                  scalar_t* out, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

template <typename scalar_t>
void vector_add_launch(const scalar_t* a, const scalar_t* b, scalar_t* out,
                       int64_t n, cudaStream_t stream) {
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  vector_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(a, b, out, n);
}

// Explicit instantiations for the dtypes dispatched in vector_add.cpp.
template void vector_add_launch<float>(const float*, const float*, float*,
                                       int64_t, cudaStream_t);
template void vector_add_launch<double>(const double*, const double*, double*,
                                        int64_t, cudaStream_t);
template void vector_add_launch<c10::Half>(const c10::Half*, const c10::Half*,
                                           c10::Half*, int64_t, cudaStream_t);
template void vector_add_launch<c10::BFloat16>(const c10::BFloat16*,
                                               const c10::BFloat16*,
                                               c10::BFloat16*, int64_t,
                                               cudaStream_t);

}  // namespace nanovllm
