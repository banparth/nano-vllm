#include <ATen/cuda/CUDAContext.h>

#include "ops.h"

namespace nanovllm {

template <typename scalar_t>
__global__ void vector_add_kernel(const scalar_t* a, const scalar_t* b,
                                  scalar_t* out, int64_t n) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "dtype mismatch");
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
  a = a.contiguous();
  b = b.contiguous();
  auto out = torch::empty_like(a);
  int64_t n = a.numel();
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  // Launch on the current stream so the op is CUDA-graph capture safe.
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(),
                                  "vector_add", [&] {
    vector_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(), n);
  });
  return out;
}

}  // namespace nanovllm
