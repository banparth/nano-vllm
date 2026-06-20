# Custom CUDA kernels (`nanovllm._C`)

This directory holds nano-vllm's hand-written C++/CUDA kernels. They are compiled
ahead of time into an importable extension module (`nanovllm._C`) and exposed to
Python as first-class PyTorch operators under `torch.ops.nanovllm.*`.

Reach for a custom kernel only when pure PyTorch / Triton can't express what you
need (warp-level primitives, tensor-core `mma`, bespoke shared-memory layouts, or
reusing existing CUDA/CUTLASS code). For most fused elementwise/reduction ops,
Triton (already used for `store_kvcache`) is simpler and nearly as fast.

The first op, `vector_add`, is intentionally trivial: its only job is to prove the
build + registration pipeline end to end. Real kernels follow the same recipe.

## Directory layout

```
csrc/
  core/
    registration.h        # PyInit macro (shared infra; no kernels here)
    clangd_cuda_shim.h    # IDE-only: lets clangd parse .cu as C++ (see below)
  ops.h                   # central declarations of every host-side entry point
  elementwise/            # a category dir: one file per op
    vector_add.cu
  torch_bindings.cpp      # the single place that registers every op
  README.md
```

Kernels are grouped by category directory (`elementwise/`, and later
`attention/`, `gemm/`, `normalization/`, ...). `core/` holds shared
infrastructure, `ops.h` is the one header the bindings include, and CMake
auto-discovers every `.cu`/`.cpp` under `csrc/`, so new files just need to be
declared and registered.

## What changed, and why

- `pyproject.toml` - build backend switched from `setuptools` to
  `scikit-build-core` (`build-backend = "scikit_build_core.build"`). Why: we now
  need CMake to drive an ahead-of-time native (CUDA) build, which setuptools
  can't do cleanly. Added `[tool.scikit-build]` (build dir, `wheel.packages`,
  editable rebuild) and put `nano-vllm` in uv's `no-build-isolation-package`
  (the build must import the real `torch` + CUDA).
- `CMakeLists.txt` (new) - finds Python, the CUDA toolkit, and libtorch, then
  auto-discovers and compiles every `csrc/**/*.cu|*.cpp` into `nanovllm/_C*.so`.
  Why: single source of truth for the build; globbing means new kernels need no
  CMake edit.
- `csrc/elementwise/vector_add.cu` (new) - the CUDA kernel and its host launcher,
  filed under its category. Why: the worked example; launches on the current
  stream so it is CUDA-graph safe.
- `csrc/ops.h` (new) - central declarations of every kernel's host entry point.
  Why: one header for `torch_bindings.cpp` to include; keeps declarations in one
  place as the kernel set grows.
- `csrc/torch_bindings.cpp` (new) - the single place that registers all ops into
  the PyTorch dispatcher with `TORCH_LIBRARY` (schema) + `TORCH_LIBRARY_IMPL`
  (CUDA impl). Why: makes each op a real operator instead of an opaque function.
- `csrc/core/registration.h` (new) - a tiny `REGISTER_EXTENSION` macro that emits
  a minimal `PyInit_*`. Why: makes `import nanovllm._C` work (which runs the
  `TORCH_LIBRARY` registration) without pulling in pybind11.
- `nanovllm/_custom_ops.py` (new) - imports `nanovllm._C`, adds a
  `register_fake` (meta) for each op, and exposes thin Python wrappers. Why: the
  fake makes the op `torch.compile`-traceable; the wrappers are the public API.
- `tests/unit/engine/test_custom_ops.py` (new) - dtype parity vs `a + b`,
  `torch.library.opcheck`, and `torch.compile(fullgraph=True)`. Why: proves
  correctness and that the op does not cause a graph break.

## Architecture (two independent axes)

- Build (when/how source becomes a binary): CMake + scikit-build-core, ahead of
  time. The `.so` is compiled at install and tied to this Python + torch build.
- Exposure (how Python sees the op): `TORCH_LIBRARY` + `register_fake`. The op
  joins the dispatcher and behaves like a built-in (`torch.mm`), so it composes
  with autograd, `torch.compile`, and `torch.export`.

```
csrc/<category>/<op>.cu  -- kernel + host launcher (declared in csrc/ops.h)
        |
csrc/torch_bindings.cpp  -- TORCH_LIBRARY schema + CUDA impl + REGISTER_EXTENSION
        |  (CMake auto-globs + builds ->)
nanovllm/_C.so  -- import runs the registration
        |
nanovllm/_custom_ops.py  -- register_fake (meta) + Python wrapper
        |
torch.ops.nanovllm.<op>  -- usable in eager / torch.compile / CUDA graphs
```

## Build & dev workflow

Prerequisites: `cmake >= 3.26` and `ninja` available in the environment, plus the
CUDA toolkit (`nvcc`). Build dependencies (`scikit-build-core`, `torch`, `wheel`)
must already be installed because we build with isolation off.

```bash
# one-time dev tools (if missing)
uv pip install ninja scikit-build-core wheel

# build + install (editable)
uv pip install -e . --no-build-isolation

# smoke test
uv run python -c "import torch; from nanovllm import _custom_ops as o; \
a=torch.randn(8,device='cuda'); print(torch.allclose(o.vector_add(a,a), a+a))"
```

- Editable rebuilds: `editable.rebuild = true` recompiles automatically on import
  after you change `csrc/` or `CMakeLists.txt`. If it ever goes stale, re-run the
  install above.
- CUDA arch: defaults to the H100 (`sm_90`) via `TORCH_CUDA_ARCH_LIST=9.0` set in
  `CMakeLists.txt`. Override by exporting `TORCH_CUDA_ARCH_LIST` before building
  (e.g. `TORCH_CUDA_ARCH_LIST=8.0 uv pip install -e . --no-build-isolation`).

## Editor / IntelliSense (clangd)

C++/CUDA navigation uses clangd (the `vscode-clangd` extension):

- `CMakeLists.txt` emits `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS`),
  symlinked at the repo root so clangd picks up the torch/CUDA/Python include paths.
- `.clangd` strips nvcc-only flags clang can't parse, and parses `.cu`/`.cuh` as
  C++ (`-xc++`) so libtorch's API resolves for go-to-definition. clang's CUDA
  frontend mis-resolves torch 2.12's headers (`torch::empty_like`, `data_ptr`, ...);
  plain C++ parsing resolves them cleanly.
- `core/clangd_cuda_shim.h` is force-included by clangd only (guarded by
  `__CUDACC__`, so nvcc ignores it) to define the CUDA keywords/builtins
  (`__global__`, `threadIdx`, ...) used in kernel bodies.
- `.vscode/settings.json` sets `clangd.path` to the installed clangd binary.

Caveat: a launch `kernel<<<grid, block>>>(...)` isn't valid C++ grammar, so that
one diagnostic (`expected_expression`) is suppressed in `.clangd`; nvcc validates
launches at build time. After editing build flags, run "clangd: Restart language
server" if navigation looks stale.

## How to add a new kernel

1. Write `csrc/<category>/<op>.cu` (e.g. `normalization/rms_norm.cu`): the kernel
   plus a host launcher that takes/returns `torch::Tensor`, and `#include "ops.h"`.
   Launch on `at::cuda::getCurrentCUDAStream()` and do no host sync / `.item()` /
   out-of-pool allocation (keeps it CUDA-graph safe).
2. Declare the host function in `csrc/ops.h`, under its category comment.
3. In `csrc/torch_bindings.cpp`, add a `m.def("<op>(...) -> ...")` schema in the
   `TORCH_LIBRARY` block and a `m.impl("<op>", &nanovllm::<op>)` in the
   `TORCH_LIBRARY_IMPL(nanovllm, CUDA, m)` block. Mark mutated args with `Tensor(a!)`.
4. No CMake edit needed - sources are auto-discovered. (If an editable rebuild
   doesn't pick up a brand-new file, re-run the install once.)
5. In `nanovllm/_custom_ops.py`, add a `@torch.library.register_fake("nanovllm::<op>")`
   (return output metadata only, no compute) and a thin Python wrapper.
6. Add a test: parity vs a reference, `torch.library.opcheck`, and a
   `torch.compile(fullgraph=True)` check.

## Constraints: `torch.compile` + CUDA graphs

These are two independent properties:

- Capturable (works inside a CUDA graph) depends only on kernel behavior: launch
  on the current stream, no host sync, no host copy, no out-of-pool allocation.
- Traceable (no `torch.compile` graph break) depends only on registration:
  `TORCH_LIBRARY` + a correct `register_fake`.

nano-vllm uses both `torch.compile` (on some layers) and CUDA-graph capture (in
`model_runner.py`), so kernels should satisfy both. `register_fake` returns the
output's shape/dtype/device without running the kernel, which is what lets the
compiler trace through the op.

## Glossary

- AOT vs JIT: AOT compiles at install (instant first call, what we do here); JIT
  (e.g. `torch.utils.cpp_extension.load`) compiles on first use at runtime.
- API vs ABI: API is the source-level contract (signatures); ABI is the
  binary-level contract (struct layout, calling convention, symbol linkage).
- Stable ABI: a C-API subset CPython keeps stable across versions, so one wheel
  works on many Python versions. Not used here (single local environment); it
  only matters when distributing wheels.
