"""Probability-sensitive correctness regression test for nano-vllm.

Run this whenever you touch the engine, scheduler, block manager, attention,
or KV cache code. It should take ~20-40s on Qwen3-0.6B and gives you both a
strict "tokens changed" signal and a continuous "distribution drifted"
signal.

What it checks
--------------
At every sampling step it captures ``log p(chosen_token | context)`` (the
"chosen log-prob"). On comparison against the golden snapshot it reports:

  - **token mismatches** -- chosen token differs (hard regression).
  - **log-prob drift**   -- tokens agree but the model's confidence moved.
                            This catches distributional drift that isn't
                            big enough to flip argmax -- exactly the
                            failure mode you get from a KV-cache restore
                            that's slightly numerically wrong.

How it samples
--------------
Two test-only sampler modes (engine source on disk is not modified; we
monkey-patch ``Sampler.forward`` and lift the ``temperature > 0`` check
in ``SamplingParams`` only in this process):

  --sampling greedy   (default): tokens = argmax(logits).  Bit-deterministic
                                 on a fixed hardware/build.  Use this for
                                 the strictest "did I change the argmax?"
                                 regression check.
  --sampling gumbel:             tokens = argmax(log p - log e), e ~ Exp(1)
                                 with the CUDA RNG seeded. This is the
                                 Gumbel-max trick: each token is a proper
                                 sample from softmax(logits/T) but the
                                 noise sequence is reproducible run-to-run.
                                 More sensitive to distributional drift
                                 than greedy (a tiny logit shift can flip
                                 a sample near the Gumbel-perturbed
                                 argmax). May produce small spurious
                                 cross-scenario diffs from FP noise.

Scenarios (same prompt set, all three should agree token-for-token)
------------------------------------------------------------------
  cold  : fresh engine, first pass on the prompts.
  warm  : second pass on the same engine -- prompts fully in the prefix
          cache. Must match cold; a mismatch means prefix caching is
          numerically broken.
  tight : new engine with a much smaller KV budget so blocks actually get
          evicted during prefill. Must also match cold; this is the
          regression sentinel for the CPU-offload / eviction work.

Usage
-----
    # First time, after you're confident the current code is correct:
    python tests/correctness.py --bless

    # In your edit loop:
    python tests/correctness.py

    # Most sensitive distribution check (catches drift greedy misses):
    python tests/correctness.py --sampling gumbel

    # Looser log-prob tolerance if FP noise trips you up:
    python tests/correctness.py --logprob-tol 1e-2

Exits non-zero on any token mismatch or log-prob diff above tolerance.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch


GOLDEN_PATH = Path(__file__).resolve().parent / "correctness_golden.json"
DEFAULT_MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
DEFAULT_SEED = 0
DEFAULT_LOGPROB_TOL = 1e-3   # |dlogprob| above this fails the check.


# ---------------------------------------------------------------------------
# Test-only sampler. Captures log-prob of the chosen token at every step.
# ---------------------------------------------------------------------------

# Per-scheduler-call chosen log-probs. Each entry is one batch's worth of
# floats (length == number of sequences active in that call), in the
# scheduler's seq_id order. Transposed into per-prompt arrays after each
# generate() call.
_CAPTURED_BATCHES: list[list[float]] = []


def _capture_clear() -> None:
    _CAPTURED_BATCHES.clear()


def _capture_to_per_prompt(num_prompts: int) -> list[list[float]]:
    """Transpose per-call batches into per-prompt log-prob arrays.

    Relies on the scheduler always batching seqs in seq_id order (it does:
    waiting/running are FIFO deques), and on every prompt being present in
    every batch (true here because we use ``ignore_eos=True`` and a fixed
    ``max_tokens``, so all sequences finish on the same decode step).
    """
    out: list[list[float]] = [[] for _ in range(num_prompts)]
    for batch in _CAPTURED_BATCHES:
        for i, lp in enumerate(batch):
            if i < num_prompts:
                out[i].append(lp)
    return out


def install_test_sampler(mode: str) -> None:
    """Replace ``Sampler.forward`` with a deterministic capturing sampler.

    Patches the class object; subsequent ``Sampler()`` instances pick this
    up via normal attribute lookup. Only affects this Python process, not
    the engine source. Lifts the ``temperature > 0`` assertion in
    ``SamplingParams`` so we can pass ``0.0`` for greedy without changing
    engine code.

    NOTE: only applied in rank-0 (this process). If you ever extend this
    test to ``tensor_parallel_size > 1`` you also need to install the patch
    inside the spawned worker.
    """
    from nanovllm.layers import sampler as _sampler_mod
    from nanovllm import sampling_params as _sp_mod

    _sp_mod.SamplingParams.__post_init__ = lambda self: None

    if mode == "greedy":
        def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
            del self, temperatures
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            tokens = log_probs.argmax(dim=-1)
            chosen = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
            _CAPTURED_BATCHES.append(chosen.cpu().tolist())
            return tokens
    elif mode == "gumbel":
        # Gumbel-max trick in its exponential form: argmax(log p - log e),
        # e_i ~ Exp(1) iid, samples from softmax(logits/T).
        def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
            del self
            scaled = logits.float() / temperatures.unsqueeze(dim=1).clamp_min(1e-10)
            log_probs = torch.log_softmax(scaled, dim=-1)
            noise = torch.empty_like(log_probs).exponential_(1).clamp_min_(1e-10)
            tokens = (log_probs - torch.log(noise)).argmax(dim=-1)
            chosen = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
            _CAPTURED_BATCHES.append(chosen.cpu().tolist())
            return tokens
    else:
        raise ValueError(f"unknown sampling mode: {mode}")

    _sampler_mod.Sampler.forward = forward


def _seed_rng(seed: int) -> None:
    """Seed torch + CUDA RNG. Required for ``gumbel`` mode to be reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Prompt set. Strings (not token ids) so the engine's own tokenizer is on
# the hot path -- catches accidental tokenizer/config drift too.
# ---------------------------------------------------------------------------

PROMPTS: list[str] = [
    "Hello, my name is",
    "The capital of France is",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a small village nestled between two mountains,",
    "The Python programming language was designed by",
    "List three reasons why exercise is good for you:\n1.",
    # A longer one to push past one KV block (block_size=256) and exercise
    # cross-block attention / cache writes.
    (
        "In the field of machine learning, transformer models have revolutionized "
        "natural language processing by introducing the self-attention mechanism. "
        "Unlike recurrent neural networks, transformers can process entire sequences "
        "in parallel, which dramatically improves training efficiency on modern GPUs. "
        "The original transformer paper, titled 'Attention Is All You Need', was "
        "published in 2017 and proposed an architecture consisting of an encoder "
        "and a decoder, each composed of stacked self-attention and feed-forward "
        "layers. Today, decoder-only variants such as the GPT family"
    ),
]
MAX_TOKENS = 24


# ---------------------------------------------------------------------------
# Engine wrappers.
# ---------------------------------------------------------------------------

ScenarioResult = dict[str, list[list[Any]]]  # {"tokens": [[int]], "logprobs": [[float]]}


def _build_llm(model_path: str, gpu_memory_utilization: float):
    from nanovllm import LLM

    return LLM(
        model_path,
        enforce_eager=True,          # skip CUDA graphs -> more deterministic
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        max_num_seqs=16,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def _generate(llm, prompts: list[str], mode: str, seed: int) -> ScenarioResult:
    from nanovllm import SamplingParams

    _capture_clear()
    _seed_rng(seed)
    # Greedy ignores temperature; for gumbel use the engine's normal 0.6.
    temperature = 0.6 if mode == "gumbel" else 1.0
    sp = SamplingParams(temperature=temperature, max_tokens=MAX_TOKENS, ignore_eos=True)
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    tokens = [o["token_ids"] for o in outputs]
    logprobs = _capture_to_per_prompt(len(prompts))
    return {"tokens": tokens, "logprobs": logprobs}


def _destroy_llm(llm) -> None:
    """Tear an LLM engine down so a fresh one can take its GPU memory.

    ``LLMEngine.__init__`` registers ``self.exit`` with ``atexit``; we
    unregister it first so process shutdown doesn't trip an
    ``AttributeError`` on the already-destroyed runner.
    """
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass
    try:
        llm.exit()
    except Exception:
        pass
    del llm
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------

SCENARIOS = ("cold", "warm", "tight")


def run_scenarios(
    model_path: str,
    scenarios: Iterable[str],
    mode: str,
    seed: int,
) -> dict[str, ScenarioResult]:
    """Run the requested scenarios. cold/warm share an LLM; tight uses its own."""
    scenarios = set(scenarios)
    results: dict[str, ScenarioResult] = {}

    if "cold" in scenarios or "warm" in scenarios:
        print("[setup] building LLM (gpu_memory_utilization=0.5)")
        llm = _build_llm(model_path, gpu_memory_utilization=0.5)
        if "cold" in scenarios:
            print("[run] cold: first pass, empty cache")
            results["cold"] = _generate(llm, PROMPTS, mode, seed)
        if "warm" in scenarios:
            print("[run] warm: second pass, prompts cached")
            results["warm"] = _generate(llm, PROMPTS, mode, seed)
        _destroy_llm(llm)

    if "tight" in scenarios:
        print("[setup] building LLM (gpu_memory_utilization=0.25, forces eviction)")
        llm = _build_llm(model_path, gpu_memory_utilization=0.25)
        print("[run] tight: small KV budget")
        results["tight"] = _generate(llm, PROMPTS, mode, seed)
        _destroy_llm(llm)

    return results


# ---------------------------------------------------------------------------
# Diff helpers.
# ---------------------------------------------------------------------------

def _first_token_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _logprob_stats(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """Return (max_abs_diff, mean_abs_diff, argmax_position) over equal-length prefixes."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0
    diffs = [abs(a[i] - b[i]) for i in range(n)]
    max_d = max(diffs)
    mean_d = sum(diffs) / n
    arg = diffs.index(max_d)
    return max_d, mean_d, arg


def _compare(
    label: str,
    a: ScenarioResult,
    b: ScenarioResult,
    logprob_tol: float,
) -> bool:
    """Compare two scenario results. Return True iff tokens match AND
    every prompt's max abs logprob diff is within tolerance.

    Prints one summary line per call on success (with max/mean drift), and
    per-prompt diagnostics on any failure.
    """
    a_tok, a_lp = a["tokens"], a["logprobs"]
    b_tok, b_lp = b["tokens"], b["logprobs"]
    if len(a_tok) != len(b_tok):
        print(f"  [FAIL] {label}: {len(a_tok)} vs {len(b_tok)} outputs")
        return False

    ok = True
    overall_max = 0.0
    overall_sum = 0.0
    overall_n = 0
    for i in range(len(a_tok)):
        if a_tok[i] != b_tok[i]:
            ok = False
            idx = _first_token_divergence(a_tok[i], b_tok[i])
            print(f"  [FAIL] {label}/prompt{i}: TOKEN divergence at pos {idx}")
            print(f"    a: {a_tok[i]}")
            print(f"    b: {b_tok[i]}")
            continue
        max_d, mean_d, arg = _logprob_stats(a_lp[i], b_lp[i])
        n = min(len(a_lp[i]), len(b_lp[i]))
        overall_max = max(overall_max, max_d)
        overall_sum += mean_d * n
        overall_n += n
        if max_d > logprob_tol:
            ok = False
            print(
                f"  [FAIL] {label}/prompt{i}: LOGPROB drift max|d|={max_d:.3e} "
                f"(tol={logprob_tol:.0e}) at pos {arg}, mean|d|={mean_d:.3e}"
            )
            print(f"    a lp[{arg}]={a_lp[i][arg]:.6f}")
            print(f"    b lp[{arg}]={b_lp[i][arg]:.6f}")

    if ok:
        mean = overall_sum / overall_n if overall_n else 0.0
        if overall_max == 0.0:
            print(f"  [ok]   {label}: tokens + logprobs bit-exact (n={overall_n})")
        else:
            print(f"  [ok]   {label}: tokens match, logprob max|d|={overall_max:.3e} mean|d|={mean:.3e} (tol={logprob_tol:.0e})")
    return ok


# ---------------------------------------------------------------------------
# Golden I/O.
# ---------------------------------------------------------------------------

def _golden_meta(mode: str, seed: int) -> dict[str, Any]:
    return {"mode": mode, "seed": seed, "max_tokens": MAX_TOKENS}


def write_golden(results: dict[str, ScenarioResult], mode: str, seed: int) -> None:
    payload: dict[str, Any] = {
        "prompts": PROMPTS,
        "sampling": _golden_meta(mode, seed),
        "scenarios": results,
    }
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def read_golden() -> dict[str, Any] | None:
    if not GOLDEN_PATH.exists():
        return None
    return json.loads(GOLDEN_PATH.read_text())


def check_against_golden(
    results: dict[str, ScenarioResult],
    golden: dict[str, Any],
    mode: str,
    seed: int,
    logprob_tol: float,
) -> bool:
    if golden.get("prompts") != PROMPTS:
        print("  [FAIL] golden has a different prompt set. Re-bless.")
        return False
    g_meta = golden.get("sampling", {})
    if g_meta.get("max_tokens") != MAX_TOKENS:
        print("  [FAIL] golden has a different max_tokens. Re-bless.")
        return False
    if g_meta.get("mode") != mode:
        print(f"  [FAIL] golden mode='{g_meta.get('mode')}', running mode='{mode}'.")
        print(f"         Either pass --sampling {g_meta.get('mode')} or re-bless.")
        return False
    if g_meta.get("seed") != seed:
        print(f"  [FAIL] golden seed={g_meta.get('seed')}, running seed={seed}.")
        return False

    ok = True
    for name, got in results.items():
        want = golden.get("scenarios", {}).get(name)
        if want is None:
            print(f"  [skip] golden has no '{name}' scenario yet")
            continue
        if not _compare(f"golden/{name}", got, want, logprob_tol):
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF model dir (default: {DEFAULT_MODEL})")
    ap.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all",
                    help="Which scenario(s) to run (default: all)")
    ap.add_argument("--sampling", choices=("greedy", "gumbel"), default="greedy",
                    help="Sampling strategy (default: greedy). 'gumbel' samples "
                         "from softmax(logits/T) with seeded Gumbel noise -- "
                         "more sensitive to distributional drift but may flake "
                         "cross-scenario on FP noise.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="CUDA RNG seed for gumbel mode (default: 0)")
    ap.add_argument("--logprob-tol", type=float, default=DEFAULT_LOGPROB_TOL,
                    help=f"Max |dlog p| per token before failing (default: {DEFAULT_LOGPROB_TOL:.0e})")
    ap.add_argument("--bless", action="store_true",
                    help="Capture current outputs as the new golden snapshot.")
    ap.add_argument("--no-cross-check", action="store_true",
                    help="Skip cold==warm==tight invariance checks.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    install_test_sampler(args.sampling)

    if not os.path.isdir(args.model):
        print(f"[FAIL] model path does not exist: {args.model}")
        return 2

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = run_scenarios(args.model, scenarios, args.sampling, args.seed)

    if args.bless:
        write_golden(results, args.sampling, args.seed)
        n_lp = sum(len(s["logprobs"][0]) for s in results.values()) if results else 0
        print(f"\n[OK] golden snapshot written to {GOLDEN_PATH}")
        print(f"     mode={args.sampling} seed={args.seed} "
              f"scenarios={list(results.keys())} logprobs/prompt={n_lp // max(1, len(results))}")
        return 0

    print(f"\n== cross-scenario invariance (tol={args.logprob_tol:.0e}) ==")
    cross_ok = True
    others = [n for n in SCENARIOS if n in results]
    if args.no_cross_check or len(others) < 2:
        if len(others) < 2:
            print("  [skip] need >=2 scenarios for cross-checks")
        else:
            print("  [skip] disabled by --no-cross-check")
    else:
        ref_name, *rest = others
        ref = results[ref_name]
        for name in rest:
            if not _compare(f"{ref_name} vs {name}", ref, results[name], args.logprob_tol):
                cross_ok = False

    print("\n== golden snapshot ==")
    golden_ok = True
    golden = read_golden()
    if golden is None:
        print(f"  [skip] no golden at {GOLDEN_PATH}; run with --bless to create one")
    else:
        golden_ok = check_against_golden(results, golden, args.sampling, args.seed, args.logprob_tol)

    if cross_ok and golden_ok:
        print("\n[OK] correctness check passed")
        return 0
    print("\n[FAIL] correctness check failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
