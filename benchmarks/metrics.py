"""Per-workload measurement routines + output checksum.

Each ``run_*`` returns a flat dict of JSON-serializable metrics. ``run_workload``
dispatches on ``Workload.kind``. Decoding is assumed deterministic (greedy
sampler installed by the runner), so ``checksum`` over the produced token ids is
a stable fingerprint that lets the compare tool flag accidental output changes.
"""

from __future__ import annotations

import hashlib
from time import perf_counter
from typing import Any

from benchmarks.workloads import Workload
from nanovllm import LLM


def checksum(token_lists: list[list[int]]) -> str:
    h = hashlib.sha1()
    for toks in token_lists:
        h.update(b"|")
        h.update(",".join(map(str, toks)).encode())
    return h.hexdigest()[:16]


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _input_tokens(prompts: list) -> int:
    # Prompts are token-id lists for synthetic/sharegpt workloads.
    return sum(len(p) for p in prompts)


def run_throughput(llm: LLM, wl: Workload) -> dict[str, Any]:
    t0 = perf_counter()
    outs = llm.generate(wl.prompts, wl.sampling, use_tqdm=False)
    dt = perf_counter() - t0
    in_tok = _input_tokens(wl.prompts)
    out_tok = sum(len(o["token_ids"]) for o in outs)
    return {
        "wall_s": round(dt, 4),
        "n_prompts": len(wl.prompts),
        "input_tok": in_tok,
        "output_tok": out_tok,
        "total_tok_s": round((in_tok + out_tok) / dt, 1),
        "output_tok_s": round(out_tok / dt, 1),
        "prefill_tok_s": round(in_tok / dt, 1),
        "checksum": checksum([o["token_ids"] for o in outs]),
    }


def run_prefix(llm: LLM, wl: Workload) -> dict[str, Any]:
    """Cold + warm passes. Wall speedup is the prefix-cache effectiveness signal
    (no engine instrumentation needed)."""
    passes: list[tuple[float, list]] = []
    for _ in range(max(2, wl.passes)):
        t0 = perf_counter()
        outs = llm.generate(wl.prompts, wl.sampling, use_tqdm=False)
        passes.append((perf_counter() - t0, outs))
    cold_s, cold_outs = passes[0]
    warm_s, _ = passes[-1]
    in_tok = _input_tokens(wl.prompts)
    out_tok = sum(len(o["token_ids"]) for o in cold_outs)
    return {
        "n_prompts": len(wl.prompts),
        "input_tok": in_tok,
        "output_tok": out_tok,
        "cold_s": round(cold_s, 4),
        "warm_s": round(warm_s, 4),
        "wall_speedup": round(cold_s / warm_s, 3) if warm_s > 0 else 0.0,
        "cold_total_tok_s": round((in_tok + out_tok) / cold_s, 1),
        "warm_total_tok_s": round((in_tok + out_tok) / warm_s, 1),
        "checksum": checksum([o["token_ids"] for o in cold_outs]),
    }


def run_latency(llm: LLM, wl: Workload) -> dict[str, Any]:
    """Single-stream latency by driving the engine step-by-step.

    TTFT = time of the first step (prefill -> first token); decode latency =
    each subsequent single-token step. Assumes the prompt fits in one prefill
    batch (true for the latency workload's prompt_len).
    """
    prompt = wl.prompts[0]
    llm.add_request(prompt, wl.sampling)
    t0 = perf_counter()
    llm.step()  # prefill -> first token
    ttft = perf_counter() - t0
    step_times: list[float] = []
    while not llm.is_finished():
        t = perf_counter()
        llm.step()
        step_times.append(perf_counter() - t)
    decode_tokens = len(step_times)
    decode_s = sum(step_times)
    return {
        "prompt_len": len(prompt),
        "gen_tokens": decode_tokens + 1,
        "ttft_ms": round(ttft * 1000, 3),
        "decode_tok_s": round(decode_tokens / decode_s, 1) if decode_s > 0 else 0.0,
        "inter_token_p50_ms": round(_pct(step_times, 0.50) * 1000, 3),
        "inter_token_p99_ms": round(_pct(step_times, 0.99) * 1000, 3),
    }


def run_accuracy(llm: LLM, wl: Workload) -> dict[str, Any]:
    from benchmarks.gsm8k import score_predictions

    t0 = perf_counter()
    outs = llm.generate(wl.prompts, wl.sampling, use_tqdm=False)
    dt = perf_counter() - t0
    texts = [o["text"] for o in outs]
    sc = score_predictions(texts, wl.references or [])
    out_tok = sum(len(o["token_ids"]) for o in outs)
    return {
        "n_questions": len(wl.prompts),
        "accuracy": round(sc["accuracy"], 4),
        "correct": sc["correct"],
        "total": sc["total"],
        "wall_s": round(dt, 4),
        "output_tok": out_tok,
        "output_tok_s": round(out_tok / dt, 1) if dt > 0 else 0.0,
        "checksum": checksum([o["token_ids"] for o in outs]),
    }


_DISPATCH = {
    "throughput": run_throughput,
    "prefix": run_prefix,
    "latency": run_latency,
    "accuracy": run_accuracy,
}


def run_workload(llm: LLM, wl: Workload) -> dict[str, Any]:
    return _DISPATCH[wl.kind](llm, wl)
