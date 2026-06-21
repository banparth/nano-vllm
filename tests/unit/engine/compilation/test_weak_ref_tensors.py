import torch

from nanovllm.engine.compilation.weak_ref_tensors import weak_ref_tensor


def test_weak_ref_tensor():
    x = torch.randn(10, 10)
    y = weak_ref_tensor(x)
    assert x.data_ptr() == y.data_ptr()
    assert x.storage_offset() == y.storage_offset()
    assert x.size() == y.size()
    assert x.stride() == y.stride()


def test_is_tensor():
    x = torch.randn(10, 10)
    assert torch.is_tensor(x)
