#pragma once
#include <torch/all.h>

// Central declarations for every custom kernel's host-side entry point.
// Group declarations by category; implementations live under csrc/<category>/.
// torch_bindings.cpp includes this header to register the schemas/impls.
namespace nanovllm {

// --- elementwise/ ---
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);

}  // namespace nanovllm
