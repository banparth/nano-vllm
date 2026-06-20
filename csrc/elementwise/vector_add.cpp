#include <ATen/cuda/CUDAContext.h>

#include "elementwise/vector_add.cuh"
#include "ops.h"

namespace nanovllm {

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "dtype mismatch");
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
  a = a.contiguous();
  b = b.contiguous();
  auto out = torch::empty_like(a);
  int64_t n = a.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(),
                                  "vector_add", [&] {
    vector_add_launch<scalar_t>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                                out.data_ptr<scalar_t>(), n, stream);
  });
  return out;
}

}  // namespace nanovllm
