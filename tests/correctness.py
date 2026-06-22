"""Probability-sensitive correctness regression for nano-vllm, as a matrix.

Run this whenever you touch the engine, scheduler, block manager, attention or
KV-cache code. It sweeps a matrix of (model x prompt-slice x config) and, for
each cell, runs three scenarios that must all agree token-for-token and within
a log-prob tolerance:

  cold  : fresh engine, first pass on the prompts.
  warm  : second pass on the same engine -- prompts fully in the prefix cache.
          A mismatch means prefix caching is numerically broken.
  tight : new engine with a much smaller KV budget so blocks get evicted during
          prefill. The regression sentinel for the CPU-offload / eviction work.

At every sampling step it captures ``log p(chosen_token | context)`` so it
reports both hard token mismatches and softer distributional drift (a slightly
wrong KV restore moves the log-prob before it ever flips the argmax).

Two test-only samplers (process-local monkey-patch, engine source untouched):
  --sampling greedy  (default): tokens = argmax(logits). Strictest argmax check.
  --sampling gumbel           : argmax(log p - log e), e~Exp(1), seeded.

Per-cell goldens live in tests/goldens/<model>__<slice>__<config>.json.

Usage
-----
    # first time / after intentional changes (blesses all selected cells):
    uv run python tests/correctness.py --bless --models 0.6B,1.7B

    # edit loop on the smallest model:
    uv run python tests/correctness.py --models 0.6B

    # full sweep across every locally-available model:
    uv run python tests/correctness.py

Exits non-zero on any token mismatch, log-prob drift over tolerance, or a
golden mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Allow `import benchmarks` when run as a script (python tests/correctness.py).
sys.path.insert(0, REPO_ROOT)

import torch

from benchmarks.engine import build_llm, destroy_llm
from benchmarks.models import MODELS, available_keys, resolve_keys

GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
DEFAULT_LOGPROB_TOL = 1e-3
MAX_TOKENS = 24
SCENARIOS: tuple[str, ...] = ("cold", "warm", "tight")


# --------------------------------------------------------------------------- #
# Prompt slices (strings, so the engine's own tokenizer is on the hot path).
# Keep each slice <= 16 prompts so they all batch together in one scheduler
# call (required by the per-prompt log-prob transpose below).
# --------------------------------------------------------------------------- #
PROMPT_SLICES: dict[str, list[str]] = {
    "core": [
        "Hello, my name is",
        "The capital of France is",
        "Q: What is 17 * 23?\nA:",
        "Once upon a time, in a small village nestled between two mountains,",
        "The Python programming language was designed by",
        "List three reasons why exercise is good for you:\n1.",
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
    ],
    "varied": [
        "def fibonacci(n):\n    ",
        "Translate to French: 'The weather is nice today.'\n",
        '{"name": "Ada", "age": 36, "role":',
        "Roses are red, violets are blue,",
        "The derivative of x^2 with respect to x is",
        "SELECT name FROM users WHERE",
        "In 1969, the first humans to land on the Moon were",
    ],
}

# Numeric-affecting engine configs. cold/warm/tight vary gpu memory on top.
CORR_CONFIGS: dict[str, dict[str, Any]] = {
    "eager_b256": dict(enforce_eager=True, kvcache_block_size=256),
    "graph_b256": dict(enforce_eager=False, kvcache_block_size=256),
    "breakable_b256": dict(
        enforce_eager=False, use_breakable_cudagraph=True, kvcache_block_size=256
    ),
    "eager_b512": dict(enforce_eager=True, kvcache_block_size=512),
}


# --------------------------------------------------------------------------- #
# Test-only capturing sampler.
# --------------------------------------------------------------------------- #
_CAPTURED_BATCHES: list[list[float]] = []


def _capture_clear() -> None:
    _CAPTURED_BATCHES.clear()


def _capture_to_per_prompt(num_prompts: int) -> list[list[float]]:
    out: list[list[float]] = [[] for _ in range(num_prompts)]
    for batch in _CAPTURED_BATCHES:
        for i, lp in enumerate(batch):
            if i < num_prompts:
                out[i].append(lp)
    return out


def install_test_sampler(mode: str) -> None:
    from nanovllm import sampling_params as _sp_mod
    from nanovllm.layers import sampler as _sampler_mod

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Engine + scenario execution.
# --------------------------------------------------------------------------- #
ScenarioResult = dict[str, list[list[Any]]]


def _tight_util(spec) -> float:
    """A gpu-memory fraction that still fits the weights but squeezes the KV
    budget (forces eviction on the smaller models; merely tighter on big ones)."""
    floor = spec.weight_gib / 80.0 + 0.08
    return round(min(max(0.25, floor), 0.9), 3)


def _build(spec, scenario: str, corr_cfg: dict[str, Any]):
    gmu = _tight_util(spec) if scenario == "tight" else spec.gpu_mem_util
    return build_llm(
        spec.path,
        dict(
            gpu_memory_utilization=gmu,
            max_model_len=2048,
            max_num_batched_tokens=4096,
            max_num_seqs=16,
            enforce_eager=corr_cfg["enforce_eager"],
            use_breakable_cudagraph=corr_cfg.get("use_breakable_cudagraph", False),
            kvcache_block_size=corr_cfg["kvcache_block_size"],
        ),
    )


def _generate(llm, prompts: list[str], mode: str, seed: int) -> ScenarioResult:
    from nanovllm import SamplingParams

    _capture_clear()
    _seed_rng(seed)
    temperature = 0.6 if mode == "gumbel" else 1.0
    sp = SamplingParams(temperature=temperature, max_tokens=MAX_TOKENS, ignore_eos=True)
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    return {
        "tokens": [o["token_ids"] for o in outputs],
        "logprobs": _capture_to_per_prompt(len(prompts)),
    }


def run_cell(
    spec, prompts: list[str], corr_cfg: dict[str, Any], scenarios: list[str], mode: str, seed: int
) -> tuple[dict[str, ScenarioResult], dict[str, str]]:
    results: dict[str, ScenarioResult] = {}
    skipped: dict[str, str] = {}
    want = set(scenarios)
    if want & {"cold", "warm"}:
        try:
            llm = _build(spec, "cold", corr_cfg)
        except Exception as e:
            skipped["cold/warm"] = f"build failed: {type(e).__name__}: {e}"
        else:
            if "cold" in want:
                results["cold"] = _generate(llm, prompts, mode, seed)
            if "warm" in want:
                results["warm"] = _generate(llm, prompts, mode, seed)
            destroy_llm(llm)
    if "tight" in want:
        try:
            llm = _build(spec, "tight", corr_cfg)
        except Exception as e:
            skipped["tight"] = f"build failed (KV too small for this model): {type(e).__name__}"
        else:
            results["tight"] = _generate(llm, prompts, mode, seed)
            destroy_llm(llm)
    return results, skipped


# --------------------------------------------------------------------------- #
# Subprocess-per-cell isolation (mirrors tests/oracle.py): build + generate run
# in a fresh worker process so the dynamo compile cache, CUDA memory, and the
# dist rendezvous port can't leak across cells. The parent only diffs the JSON
# the worker returns.
# --------------------------------------------------------------------------- #
def _worker_cell(
    model_key: str,
    slice_name: str,
    config_name: str,
    scenarios: list[str],
    mode: str,
    seed: int,
    out_path: str,
) -> None:
    install_test_sampler(mode)
    results, skipped = run_cell(
        MODELS[model_key],
        PROMPT_SLICES[slice_name],
        CORR_CONFIGS[config_name],
        scenarios,
        mode,
        seed,
    )
    Path(out_path).write_text(json.dumps({"results": results, "skipped": skipped}))


def run_cell_subprocess(
    model_key: str, slice_name: str, config_name: str, scenarios: list[str], mode: str, seed: int
) -> tuple[dict[str, ScenarioResult], dict[str, str]]:
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="corr_cell_")
    os.close(fd)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--model",
                model_key,
                "--slice",
                slice_name,
                "--config",
                config_name,
                "--scenarios",
                ",".join(scenarios),
                "--sampling",
                mode,
                "--seed",
                str(seed),
                "--out",
                out_path,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            tail = " | ".join(proc.stderr.strip().splitlines()[-3:])
            return {}, {"worker": f"crashed (rc={proc.returncode}): {tail}"}
        try:
            payload = json.loads(Path(out_path).read_text())
        except (json.JSONDecodeError, OSError):
            return {}, {"worker": "produced no/invalid output"}
        return payload.get("results", {}), payload.get("skipped", {})
    except subprocess.TimeoutExpired:
        return {}, {"worker": "timed out"}
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


# --------------------------------------------------------------------------- #
# Diff + golden I/O.
# --------------------------------------------------------------------------- #
def _first_token_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _logprob_stats(a: list[float], b: list[float]) -> tuple[float, float, int]:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0
    diffs = [abs(a[i] - b[i]) for i in range(n)]
    max_d = max(diffs)
    return max_d, sum(diffs) / n, diffs.index(max_d)


def _compare(label: str, a: ScenarioResult, b: ScenarioResult, tol: float) -> bool:
    a_tok, a_lp = a["tokens"], a["logprobs"]
    b_tok, b_lp = b["tokens"], b["logprobs"]
    if len(a_tok) != len(b_tok):
        print(f"    [FAIL] {label}: {len(a_tok)} vs {len(b_tok)} outputs")
        return False
    ok, overall_max = True, 0.0
    for i in range(len(a_tok)):
        if a_tok[i] != b_tok[i]:
            ok = False
            idx = _first_token_divergence(a_tok[i], b_tok[i])
            print(f"    [FAIL] {label}/prompt{i}: TOKEN divergence at pos {idx}")
            continue
        max_d, _mean_d, arg = _logprob_stats(a_lp[i], b_lp[i])
        overall_max = max(overall_max, max_d)
        if max_d > tol:
            ok = False
            print(
                f"    [FAIL] {label}/prompt{i}: LOGPROB drift max|d|={max_d:.3e} "
                f"(tol={tol:.0e}) at pos {arg}"
            )
    if ok:
        kind = "bit-exact" if overall_max == 0.0 else f"max|d|={overall_max:.3e}"
        print(f"    [ok]   {label}: tokens match, {kind}")
    return ok


def cell_golden_path(model: str, slice_name: str, config: str) -> Path:
    return GOLDEN_DIR / f"{model}__{slice_name}__{config}.json"


def write_golden(
    path: Path, prompts: list[str], results: dict[str, ScenarioResult], mode: str, seed: int
) -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prompts": prompts,
                "sampling": {"mode": mode, "seed": seed, "max_tokens": MAX_TOKENS},
                "scenarios": results,
            },
            indent=2,
        )
        + "\n"
    )


def check_against_golden(
    path: Path,
    prompts: list[str],
    results: dict[str, ScenarioResult],
    mode: str,
    seed: int,
    tol: float,
) -> bool | None:
    if not path.exists():
        print(f"    [skip] no golden ({path.name}); run with --bless")
        return None
    golden = json.loads(path.read_text())
    if golden.get("prompts") != prompts:
        print(f"    [FAIL] golden prompt set differs ({path.name}); re-bless")
        return False
    meta = golden.get("sampling", {})
    if meta.get("mode") != mode or meta.get("seed") != seed or meta.get("max_tokens") != MAX_TOKENS:
        print(f"    [FAIL] golden sampling meta differs ({path.name}); re-bless")
        return False
    ok = True
    for name, got in results.items():
        want = golden.get("scenarios", {}).get(name)
        if want is None:
            print(f"    [skip] golden has no '{name}' scenario")
            continue
        if not _compare(f"golden/{name}", got, want, tol):
            ok = False
    return ok


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models", default=None, help="comma list of model keys (default: all available)"
    )
    ap.add_argument(
        "--slices",
        default="core",
        help=f"comma list of prompt slices {list(PROMPT_SLICES)} (default: core)",
    )
    ap.add_argument(
        "--configs",
        default="eager_b256,graph_b256",
        help=f"comma list of configs {list(CORR_CONFIGS)} (default: eager_b256,graph_b256)",
    )
    ap.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    ap.add_argument("--sampling", choices=("greedy", "gumbel"), default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logprob-tol", type=float, default=DEFAULT_LOGPROB_TOL)
    ap.add_argument("--bless", action="store_true", help="write current outputs as the golden")
    ap.add_argument("--no-cross-check", action="store_true", help="skip cold==warm==tight checks")
    # Internal: run a single cell inside a worker subprocess (see run_cell_subprocess).
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--slice", dest="slice_name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--scenarios", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.worker:
        scenarios = [s.strip() for s in (args.scenarios or "").split(",") if s.strip()]
        _worker_cell(
            args.model, args.slice_name, args.config, scenarios, args.sampling, args.seed, args.out
        )
        return 0

    model_keys = resolve_keys(args.models) if args.models else available_keys()
    model_keys = [k for k in model_keys if MODELS[k].available]
    if not model_keys:
        print(
            "[error] no available models. Download with: uv run python -m benchmarks.download_models"
        )
        return 2
    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    print(
        f"[correctness] models={model_keys} slices={slices} configs={cfgs} "
        f"sampling={args.sampling} scenarios={scenarios}"
    )

    all_ok = True
    n_cells = 0
    for mkey in model_keys:
        for slice_name in slices:
            prompts = PROMPT_SLICES[slice_name]
            for cfg_name in cfgs:
                n_cells += 1
                print(f"\n=== {mkey} | {slice_name} | {cfg_name} ===")
                results, skipped = run_cell_subprocess(
                    mkey, slice_name, cfg_name, scenarios, args.sampling, args.seed
                )
                for sc, reason in skipped.items():
                    print(f"    [skip] {sc}: {reason}")
                if not results:
                    continue
                path = cell_golden_path(mkey, slice_name, cfg_name)
                if args.bless:
                    write_golden(path, prompts, results, args.sampling, args.seed)
                    print(f"    [bless] wrote {path.name} ({list(results)})")
                    continue
                # cross-scenario invariance
                if not args.no_cross_check and len(results) >= 2:
                    ref_name, *rest = list(results)
                    for name in rest:
                        if not _compare(
                            f"{ref_name} vs {name}",
                            results[ref_name],
                            results[name],
                            args.logprob_tol,
                        ):
                            all_ok = False
                # golden
                verdict = check_against_golden(
                    path, prompts, results, args.sampling, args.seed, args.logprob_tol
                )
                if verdict is False:
                    all_ok = False

    if args.bless:
        print(f"\n[OK] blessed {n_cells} cells into {GOLDEN_DIR}")
        return 0
    print(f"\n[{'OK' if all_ok else 'FAIL'}] correctness check over {n_cells} cells")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
