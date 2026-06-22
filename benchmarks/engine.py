"""Engine lifecycle + stats helpers shared by the benchmark and correctness
harnesses.

Keeps all the fiddly bits (tearing an engine down so the next one can take its
GPU memory, reading peak memory / KV capacity, popping block-manager cache
counters) in one place so the higher-level harnesses stay readable.
"""

from __future__ import annotations

import atexit
from typing import Any

import torch
import torch.distributed as dist

from nanovllm import LLM, SamplingParams


def _hard_cleanup() -> None:
    """Restore global torch/distributed state between engine builds.

    ``ModelRunner.__init__`` calls ``dist.init_process_group`` and sets the
    default device/dtype to CUDA, only restoring them at the *end* of a
    successful build. If a build fails partway (e.g. OOM during CUDA-graph
    capture under a tight KV budget) those globals are left dirty and the next
    build dies with "initialize the default process group twice". Building many
    engines in one process (the whole point of the matrix runner) makes this
    fatal, so we always reset here. Engine source is untouched.
    """
    try:
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
    except Exception:
        pass
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass


# All counters the block manager may expose. CPU-tier fields default to 0 until
# the KV-cache connector lands, so the harness reads a stable schema either way.
STAT_KEYS = (
    "gpu_hits",
    "cpu_hits",
    "misses",
    "evictions",
    "cpu_lru_drops",
    "h2d_copies",
    "d2h_copies",
    "prefill_seqs",
)


_GREEDY_INSTALLED = False


def install_greedy_sampler() -> None:
    """Monkey-patch the sampler to deterministic argmax (process-local only).

    This does NOT modify engine source -- it is the same runtime patch
    tests/correctness.py uses. Deterministic decoding makes benchmark output
    checksums comparable across branches and makes GSM8K reproducible. Safe to
    call multiple times; only affects rank-0 (the single H100 process).
    """
    global _GREEDY_INSTALLED
    if _GREEDY_INSTALLED:
        return
    from nanovllm import sampling_params as _sp_mod
    from nanovllm.layers import sampler as _sampler_mod

    _sp_mod.SamplingParams.__post_init__ = lambda self: None

    def forward(self, logits, temperatures):
        del self, temperatures
        return logits.argmax(dim=-1)

    _sampler_mod.Sampler.forward = forward
    _GREEDY_INSTALLED = True


def build_llm(model_path: str, cfg: dict[str, Any]) -> LLM:
    """Build an LLM. Single H100 -> tensor_parallel_size is always 1.

    Unknown kwargs (e.g. ``cpu_memory_utilization`` before the CPU tier lands)
    are silently dropped by ``LLMEngine`` -- this is intentional forward-compat.
    """
    try:
        return LLM(model_path, tensor_parallel_size=1, **cfg)
    except Exception:
        # A partially-constructed engine may have left the process group +
        # default device dirty; clean up so the next build can proceed.
        _hard_cleanup()
        raise


def destroy_llm(llm: LLM) -> None:
    """Free an engine so a fresh one can reuse the GPU (mirrors the careful
    teardown in tests/correctness.py: unregister atexit before exiting)."""
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    try:
        llm.exit()
    except Exception:
        pass
    del llm
    _hard_cleanup()


def warmup(llm: LLM, max_tokens: int = 4) -> None:
    """Settle CUDA graph replays + the allocator before any timed pass."""
    llm.generate(
        ["Benchmark: "],
        SamplingParams(temperature=0.6, max_tokens=max_tokens, ignore_eos=True),
        use_tqdm=False,
    )


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def engine_static_stats(llm: LLM) -> dict[str, Any]:
    """One-time facts about the built engine: KV capacity, block bytes, tiers."""
    mr = llm.model_runner
    cfg = mr.config
    num_blocks = cfg.num_kvcache_blocks
    block_size = cfg.kvcache_block_size
    out: dict[str, Any] = {
        "num_kvcache_blocks": num_blocks,
        "kvcache_block_size": block_size,
        "kv_capacity_tokens": num_blocks * block_size,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
    }
    kv = getattr(mr, "kv_cache", None)
    if kv is not None:
        shape = kv.shape  # (2, L, num_blocks, block, H, D)
        per_block = shape[3] * shape[4] * shape[5] * kv.element_size()
        out["kv_block_bytes"] = int(2 * shape[1] * per_block)
    bm = getattr(llm.scheduler, "block_manager", None)
    for attr in ("blocks_gpu", "blocks_cpu"):
        v = getattr(bm, attr, None)
        if v is not None:
            try:
                out[f"num_{attr}"] = len(v)
            except TypeError:
                pass
    return out


def pop_cache_stats(llm: LLM) -> dict[str, int]:
    """Pop (read + reset) the block-manager cache counters for the last pass.

    Returns the full STAT_KEYS schema with 0 fills, or {} if the engine has no
    instrumented block manager.
    """
    bm = getattr(llm.scheduler, "block_manager", None)
    if bm is None or not hasattr(bm, "pop_stats"):
        return {}
    s = bm.pop_stats()
    return {k: int(s.get(k, 0)) for k in STAT_KEYS}
