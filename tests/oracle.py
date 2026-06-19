"""Ground-truth oracle: nano-vllm greedy vs HuggingFace transformers greedy.

The correctness regression (tests/correctness.py) proves the engine is *stable*
(cold==warm==tight, matches its own golden). This oracle proves the engine is
*right*: that its greedy decoding agrees with the reference HF implementation.

How it works
------------
For each model we spawn two short-lived worker subprocesses (so nano-vllm and
the HF model never sit in GPU memory at the same time -- important for 14B/32B):

  1. nano-vllm worker : greedy-decode N tokens for each prompt, dump token ids.
  2. hf worker        : load AutoModelForCausalLM, greedy-decode N tokens, dump.

The parent then compares the two token streams per prompt. Because flash-attn
vs the HF attention kernel differ in floating point, greedy can legitimately
diverge late; so the oracle is tolerant:

  - FAIL  : the very first generated token differs (a real bug), or --strict and
            the streams aren't identical.
  - warn  : agreement is partial (FP-driven late divergence).
  - ok    : full agreement over N tokens.

Usage
-----
    uv run python tests/oracle.py --models 0.6B,1.7B
    uv run python tests/oracle.py --models 4B --max-tokens 48 --strict

Exits non-zero if any prompt fails for any model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Reuse the correctness prompt slice as the shared prompt set.
from tests.correctness import PROMPT_SLICES  # noqa: E402

PROMPTS = PROMPT_SLICES["core"]


# --------------------------------------------------------------------------- #
# Workers (run in their own process; write {idx: [token_ids]} to --out).
# --------------------------------------------------------------------------- #
def _worker_nanovllm(model_path: str, max_tokens: int, out_path: str) -> None:
    from benchmarks.engine import install_greedy_sampler
    from nanovllm import LLM, SamplingParams

    install_greedy_sampler()
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=2048, max_num_batched_tokens=4096, max_num_seqs=16)
    sp = SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True)
    # generate() returns dicts at runtime (engine annotates it list[str]); Any
    # keeps the type checker quiet without asserting the wrong element type.
    outs: Any = llm.generate(list(PROMPTS), sp, use_tqdm=False)
    tokens = {i: o["token_ids"] for i, o in enumerate(outs)}
    with open(out_path, "w") as f:
        json.dump(tokens, f)


def _worker_hf(model_path: str, max_tokens: int, out_path: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model: Any = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
    model = model.to("cuda").eval()
    tokens: dict[int, list[int]] = {}
    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **ids, do_sample=False, num_beams=1,
                min_new_tokens=max_tokens, max_new_tokens=max_tokens,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][ids["input_ids"].shape[1]:].tolist()
        tokens[i] = gen
    with open(out_path, "w") as f:
        json.dump(tokens, f)


# --------------------------------------------------------------------------- #
# Parent orchestration + comparison.
# --------------------------------------------------------------------------- #
def _run_worker(kind: str, model_path: str, max_tokens: int) -> dict[int, list[int]]:
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix=f"oracle_{kind}_")
    os.close(fd)
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", kind,
             "--model-path", model_path, "--max-tokens", str(max_tokens),
             "--out", out_path],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{kind} worker failed:\n{proc.stderr[-2000:]}")
        with open(out_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def _agreement(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare_model(model_key: str, model_path: str, max_tokens: int, strict: bool) -> bool:
    print(f"\n=== oracle {model_key} (greedy, {max_tokens} tokens) ===")
    print(f"  [nanovllm] decoding {len(PROMPTS)} prompts ...", flush=True)
    nano = _run_worker("nanovllm", model_path, max_tokens)
    print(f"  [hf]       decoding {len(PROMPTS)} prompts ...", flush=True)
    hf = _run_worker("hf", model_path, max_tokens)

    ok = True
    total_agree = 0
    for i in range(len(PROMPTS)):
        a, b = nano.get(i, []), hf.get(i, [])
        agree = _agreement(a, b)
        total_agree += agree
        full = agree == max_tokens
        if agree == 0:
            ok = False
            print(f"  [FAIL] prompt{i}: first token differs  nano={a[:4]} hf={b[:4]}")
        elif full:
            print(f"  [ok]   prompt{i}: full match ({agree}/{max_tokens})")
        else:
            tag = "FAIL" if strict else "warn"
            if strict:
                ok = False
            print(f"  [{tag}] prompt{i}: agree {agree}/{max_tokens} then diverge "
                  f"(nano={a[agree]} hf={b[agree]})")
    mean_frac = total_agree / (len(PROMPTS) * max_tokens)
    print(f"  mean agreement: {mean_frac*100:.1f}%")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # worker hooks (internal)
    ap.add_argument("--worker", choices=("nanovllm", "hf"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    # user-facing
    ap.add_argument("--models", default=None, help="comma list of model keys (default: all available)")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--strict", action="store_true", help="require full token-exact agreement")
    args = ap.parse_args()

    if args.worker == "nanovllm":
        _worker_nanovllm(args.model_path, args.max_tokens, args.out)
        return 0
    if args.worker == "hf":
        _worker_hf(args.model_path, args.max_tokens, args.out)
        return 0

    from benchmarks.models import MODELS, available_keys, resolve_keys
    keys = resolve_keys(args.models) if args.models else available_keys()
    keys = [k for k in keys if MODELS[k].available]
    if not keys:
        print("[error] no available models. Download with: uv run python -m benchmarks.download_models")
        return 2

    print(f"[oracle] models={keys} max_tokens={args.max_tokens} strict={args.strict}")
    all_ok = True
    for k in keys:
        if not compare_model(k, MODELS[k].path, args.max_tokens, args.strict):
            all_ok = False
    print(f"\n[{'OK' if all_ok else 'FAIL'}] oracle over {len(keys)} model(s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
