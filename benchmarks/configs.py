"""Named engine-config presets + per-model resolution.

A single H100 means ``tensor_parallel_size`` is always 1, so the interesting
config axes are KV budget, block size, CUDA graphs, batch limits and (forward
compat) the CPU KV tier. ``resolve_config`` starts from per-model memory hints
(so big models fit) and layers a preset on top. Presets that shrink the KV
budget (``tight_kv``/``cpu_*``) are only feasible for the smaller models; for
larger ones the engine build fails the ``num_kvcache_blocks > 0`` assert and
the runner records the cell as skipped.
"""

from __future__ import annotations

from typing import Any

from benchmarks.models import ModelSpec

# Each preset is a delta applied over the per-model defaults below.
# Note: ``cpu_memory_utilization`` is not a Config field until the CPU KV tier
# lands; LLMEngine drops unknown kwargs, so cpu_on == cpu_off today and will
# diverge automatically once the tier is implemented.
PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "eager": {"enforce_eager": True},
    "big_block": {"kvcache_block_size": 512},
    "tight_kv": {"gpu_memory_utilization": 0.30},
    "cpu_off": {"gpu_memory_utilization": 0.30, "cpu_memory_utilization": 0.0},
    "cpu_on": {"gpu_memory_utilization": 0.30, "cpu_memory_utilization": 0.4},
}

ALL_CONFIGS = list(PRESETS)


def resolve_config(spec: ModelSpec, preset: str, max_model_len: int = 4096) -> dict[str, Any]:
    if preset not in PRESETS:
        raise KeyError(f"unknown config preset '{preset}'. Known: {ALL_CONFIGS}")
    cfg: dict[str, Any] = {
        "gpu_memory_utilization": spec.gpu_mem_util,
        "max_num_batched_tokens": spec.max_num_batched_tokens,
        "max_num_seqs": spec.max_num_seqs,
        "max_model_len": max_model_len,
        "enforce_eager": False,
        "kvcache_block_size": 256,
    }
    cfg.update(PRESETS[preset])
    if spec.force_eager:
        cfg["enforce_eager"] = True  # e.g. 32B: CUDA-graph capture won't fit on 80GB
    return cfg
