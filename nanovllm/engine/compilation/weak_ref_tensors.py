import torch
from typing import Any

def weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Create a tensor that aliases the same memory as ``tensor`` without
    keeping the original storage alive. The returned tensor shares the
    underlying data pointer but owns a storage with no deleter, so it is
    safe to hold inside a captured CUDA graph (which only cares about the
    fixed address) without creating reference cycles to the real buffers.
    """
    src = tensor.untyped_storage()
    weak_storage = torch._C._construct_storage_from_data_pointer(
        src.data_ptr(), tensor.device, src.nbytes()
    )
    out = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    out.set_(weak_storage, tensor.storage_offset(), tensor.size(), tensor.stride())
    return out


def weak_ref_tensors(
    tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
) -> torch.Tensor | list[Any] | tuple[Any] | Any:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")
