import pytest
import torch

from nanovllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="custom CUDA kernels require a GPU"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_parity(dtype):
    a = torch.randn(4096, device="cuda", dtype=dtype)
    b = torch.randn(4096, device="cuda", dtype=dtype)
    torch.testing.assert_close(ops.vector_add(a, b), a + b)


def test_opcheck():
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    torch.library.opcheck(torch.ops.nanovllm.vector_add, (a, b))


def test_compile_fullgraph():
    # fullgraph=True fails if the custom op causes a graph break, proving the
    # TORCH_LIBRARY + register_fake registration makes it torch.compile-traceable.
    compiled = torch.compile(ops.vector_add, fullgraph=True)
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    torch.testing.assert_close(compiled(a, b), a + b)
