"""Python entry point for nano-vllm's custom CUDA kernels.

Importing this module loads the compiled ``nanovllm._C`` extension, whose static
initializers register the C++ ops into the PyTorch dispatcher (``torch.ops.nanovllm.*``).
For each op we also register a ``fake`` (meta) implementation so the op stays
``torch.compile``-traceable: the fake returns output metadata (shape/dtype/device)
without running the kernel.
"""

import torch

# Importing the compiled extension runs its static initializers, which register
# the ops into the dispatcher. It is a built .so, so the type checker can't see it.
import nanovllm._C  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]


@torch.library.register_fake("nanovllm::vector_add")
def _vector_add_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.nanovllm.vector_add(a, b)


def has_cuda_kernels() -> bool:
    """True when the compiled ``nanovllm._C`` extension is importable."""
    return True
