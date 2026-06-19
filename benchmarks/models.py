"""Model registry for the nano-vllm benchmark/correctness suite.

nano-vllm only implements Qwen3 (dense) via ``Qwen3ForCausalLM``, so "multiple
models" here means multiple *sizes* of Qwen3. Each entry records where the
weights live locally, the HF repo to download from, and the memory hints the
config resolver needs so that the larger sizes still fit on a single 80GB H100.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

HF_ROOT = os.path.expanduser("~/huggingface")


def weights_complete(path: str) -> bool:
    """True iff ``path`` holds a fully-downloaded HF model.

    Guards against a model that is still downloading: ``config.json`` and the
    safetensors index land early, before the weight shards. For sharded models
    we require every shard named in the index to exist; for single-file models
    we require ``model.safetensors``.
    """
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index) as f:
                shards = set(json.load(f).get("weight_map", {}).values())
        except Exception:
            return False
        return bool(shards) and all(os.path.isfile(os.path.join(path, s)) for s in shards)
    return os.path.isfile(os.path.join(path, "model.safetensors"))


@dataclass(frozen=True)
class ModelSpec:
    key: str            # short CLI name, e.g. "0.6B"
    repo: str           # HuggingFace repo id
    dirname: str        # local directory name under HF_ROOT
    weight_gib: float   # approx bf16 weight footprint (for fit checks / docs)
    # Memory hints used by configs.resolve_config so big models fit on one H100.
    # gpu_mem_util is the *default* total-GPU-memory fraction the engine may use
    # (weights + activations + KV). Big models need it high because weights
    # dominate; the 'tight_kv' preset overrides it low (only feasible for the
    # smaller sizes -- larger ones are auto-skipped at build time).
    gpu_mem_util: float = 0.9
    max_num_batched_tokens: int = 16384   # caps the prefill activation peak
    max_num_seqs: int = 256
    force_eager: bool = False             # CUDA-graph capture won't fit (e.g. 32B on 80GB)
    abs_path: str | None = None           # set for ad-hoc specs from a raw path

    @property
    def path(self) -> str:
        return self.abs_path or os.path.join(HF_ROOT, self.dirname)

    @property
    def available(self) -> bool:
        return weights_complete(self.path)


# Ordered small -> large. Keys double as CLI tokens.
MODELS: dict[str, ModelSpec] = {
    "0.6B": ModelSpec("0.6B", "Qwen/Qwen3-0.6B", "Qwen3-0.6B", 1.2),
    "1.7B": ModelSpec("1.7B", "Qwen/Qwen3-1.7B", "Qwen3-1.7B", 3.4),
    "4B":   ModelSpec("4B",   "Qwen/Qwen3-4B",   "Qwen3-4B",   8.0),
    "8B":   ModelSpec("8B",   "Qwen/Qwen3-8B",   "Qwen3-8B",   16.0),
    "14B":  ModelSpec("14B",  "Qwen/Qwen3-14B",  "Qwen3-14B",  28.0,
                      gpu_mem_util=0.93),
    # 32B weights are ~64 GiB. CUDA-graph capture for a model this large does
    # not fit alongside the weights on one 80GB H100, so it runs eager-only;
    # gpu_mem_util stays moderate to leave room for the prefill activation peak.
    "32B":  ModelSpec("32B",  "Qwen/Qwen3-32B",  "Qwen3-32B",  64.0,
                      gpu_mem_util=0.90, max_num_batched_tokens=8192,
                      max_num_seqs=64, force_eager=True),
}

ALL_KEYS: list[str] = list(MODELS)


def spec_for(name: str) -> ModelSpec:
    """Resolve a CLI ``--model``/registry token to a ModelSpec.

    Accepts a registry key ('1.7B'), a directory name ('Qwen3-1.7B'), or an
    absolute/relative path to a model dir (synthesizing an ad-hoc spec).
    """
    if name in MODELS:
        return MODELS[name]
    # directory name under HF_ROOT
    for spec in MODELS.values():
        if name == spec.dirname:
            return spec
    # raw path -> ad-hoc spec keyed by basename
    expanded = os.path.expanduser(name)
    if os.path.isfile(os.path.join(expanded, "config.json")):
        base = os.path.basename(os.path.normpath(expanded))
        return ModelSpec(base, "", base, weight_gib=0.0, abs_path=expanded)
    raise KeyError(
        f"unknown model '{name}'. Known keys: {ALL_KEYS}. "
        f"Or pass a directory containing config.json."
    )


def available_keys() -> list[str]:
    """Registry keys whose weights are present locally, small -> large."""
    return [k for k in ALL_KEYS if MODELS[k].available]


def resolve_keys(selector: str | None) -> list[str]:
    """Turn a ``--models`` selector into a list of *available* registry keys.

    ``selector`` is a comma-separated list of keys, or None / 'all' / 'available'
    meaning every locally-present model.
    """
    if selector in (None, "all", "available"):
        return available_keys()
    out: list[str] = []
    for tok in selector.split(","):
        tok = tok.strip()
        if not tok:
            continue
        spec = spec_for(tok)
        out.append(spec.key)
    return out
