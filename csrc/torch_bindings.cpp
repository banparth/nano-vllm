#include <torch/library.h>

#include "core/registration.h"
#include "ops.h"

// Central registration for every custom op. Host-side declarations live in
// ops.h; implementations are defined under csrc/<category>/. Add a schema +
// impl line here for each new op.
TORCH_LIBRARY(nanovllm, m) {
  m.def("vector_add(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanovllm, CUDA, m) {
  m.impl("vector_add", &nanovllm::vector_add);
}

REGISTER_EXTENSION(_C)
