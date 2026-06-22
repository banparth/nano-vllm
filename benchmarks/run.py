"""Matrix benchmark runner.

Runs a suite of (model x workload x config) cells, each in a freshly-built
engine (fair timing + isolated peak-memory + empty cache per cell), and writes
the metrics to ``results/<label>.json``. A failed/infeasible cell (e.g. a
KV-budget too small for a big model, or a CUDA OOM) is recorded as skip/error
and never aborts the run.

Examples
--------
    # fast edit-loop check on the smallest model
    uv run python -m benchmarks.run --suite smoke

    # full baseline across every locally-available model
    uv run python -m benchmarks.run --suite full --label baseline

    # just the latency workload on two models
    uv run python -m benchmarks.run --suite full --workloads latency --models 0.6B,1.7B
"""

from __future__ import annotations

import argparse
import sys
import traceback

from transformers import AutoTokenizer

from benchmarks import models as M
from benchmarks.configs import resolve_config
from benchmarks.engine import (
    build_llm,
    destroy_llm,
    engine_static_stats,
    install_greedy_sampler,
    peak_memory_bytes,
    pop_cache_stats,
    reset_peak_memory,
    warmup,
)
from benchmarks.metrics import run_workload
from benchmarks.results import Cell, save_results
from benchmarks.workloads import ALL_WORKLOADS, Workload, build_workload

# A suite = which models + a list of cells. Each cell is
#   (workload_name, [config presets], scale params for the workload builder).
SUITES: dict[str, dict] = {
    "smoke": {
        "models": ["0.6B"],
        "cells": [
            ("random", ["default"], dict(num_seqs=32, min_input=64, max_input=256, output_len=64)),
            ("sharegpt", ["default"], dict(num_prompts=64, output_len=32)),
            ("latency", ["default"], dict(prompt_len=256, output_len=64)),
        ],
    },
    "full": {
        "models": None,  # all locally available
        "cells": [
            ("random", ["default", "big_block"], dict(num_seqs=256, output_len=256)),
            ("sharegpt", ["default", "tight_kv"], dict(num_prompts=512, output_len=64)),
            (
                "longprefix",
                ["default", "tight_kv"],
                dict(shared_prefix=2048, num_prompts=128, output_len=64),
            ),
            ("latency", ["default", "eager"], dict(prompt_len=512, output_len=128)),
            ("gsm8k", ["default"], dict(num_questions=200, output_len=512)),
        ],
    },
    # Forward-compatible CPU KV tier comparison (OFF vs ON). cpu_* presets are
    # identical today (cpu_memory_utilization is dropped until the tier lands)
    # and will diverge automatically once it is implemented.
    "cpu-tier": {
        "models": ["0.6B", "1.7B", "4B", "8B"],
        "cells": [
            ("sharegpt", ["cpu_off", "cpu_on"], dict(num_prompts=256, output_len=64)),
            (
                "longprefix",
                ["cpu_off", "cpu_on"],
                dict(shared_prefix=2048, num_prompts=256, output_len=64),
            ),
        ],
    },
}


def _resolve_models(suite: str, sel: str | None) -> list[str]:
    if sel:
        keys = M.resolve_keys(sel)
    else:
        suite_models = SUITES[suite]["models"]
        keys = M.available_keys() if suite_models is None else list(suite_models)
    missing = [k for k in keys if not M.MODELS[k].available]
    if missing:
        print(f"[warn] skipping unavailable models (not downloaded): {missing}")
    return [k for k in keys if M.MODELS[k].available]


def _filter_cells(suite: str, wl_sel: str | None, cfg_sel: str | None):
    wlset = set(wl_sel.split(",")) if wl_sel else None
    cfgset = set(cfg_sel.split(",")) if cfg_sel else None
    out = []
    for wname, cfgnames, params in SUITES[suite]["cells"]:
        if wlset and wname not in wlset:
            continue
        cfgs = [c for c in cfgnames if (not cfgset or c in cfgset)]
        if cfgs:
            out.append((wname, cfgs, params))
    return out


def _run_cell(spec: M.ModelSpec, mkey: str, wname: str, wl, cname: str, max_model_len: int) -> Cell:
    if isinstance(wl, Exception):
        return Cell(mkey, wname, cname, "error", f"workload build failed: {wl}")
    cfg = resolve_config(spec, cname, max_model_len=max_model_len)
    try:
        llm = build_llm(spec.path, cfg)
    except Exception as e:
        # Most common: KV budget too small for this model under this preset.
        return Cell(mkey, wname, cname, "skip", f"build: {type(e).__name__}: {e}")
    status, note, metrics = "ok", "", {}
    try:
        warmup(llm)
        reset_peak_memory()
        metrics = run_workload(llm, wl)
        metrics.update(engine_static_stats(llm))
        cache = pop_cache_stats(llm)
        if cache:
            metrics["cache"] = cache
        metrics["peak_gpu_gib"] = round(peak_memory_bytes() / 2**30, 2)
    except Exception as e:
        status, note = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        destroy_llm(llm)
    return Cell(mkey, wname, cname, status, note, metrics)


def _summary_line(c: Cell) -> str:
    m = c.metrics
    if c.status != "ok":
        return f"  [{c.status:>5}] {c.key:<34} {c.note}"
    if c.workload == "latency":
        body = f"ttft={m.get('ttft_ms'):.1f}ms decode={m.get('decode_tok_s'):.0f}tok/s p99={m.get('inter_token_p99_ms'):.1f}ms"
    elif c.workload == "gsm8k":
        body = f"acc={m.get('accuracy', 0) * 100:.1f}% ({m.get('correct')}/{m.get('total')}) {m.get('output_tok_s'):.0f}tok/s"
    elif c.metrics.get("wall_speedup") is not None and "cold_s" in m:
        body = f"cold={m.get('cold_s'):.2f}s warm={m.get('warm_s'):.2f}s speedup={m.get('wall_speedup'):.2f}x"
    else:
        body = f"{m.get('total_tok_s'):.0f}tok/s out={m.get('output_tok_s'):.0f}tok/s"
    return f"  [ ok  ] {c.key:<34} {body}  peak={m.get('peak_gpu_gib', '?')}GiB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--suite", choices=list(SUITES), default="smoke")
    ap.add_argument("--models", default=None, help="comma list of keys (default: suite's models)")
    ap.add_argument("--workloads", default=None, help=f"comma list to filter ({ALL_WORKLOADS})")
    ap.add_argument("--configs", default=None, help="comma list of config presets to filter")
    ap.add_argument("--label", default=None, help="results/<label>.json (default: suite name)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument(
        "--stochastic",
        action="store_true",
        help="use the engine's real sampler instead of deterministic greedy "
        "(disables stable output checksums)",
    )
    ap.add_argument("--list", action="store_true", help="list suites/models/workloads and exit")
    args = ap.parse_args()

    if args.list:
        print("suites   :", list(SUITES))
        print("workloads:", ALL_WORKLOADS)
        print("models   :", M.ALL_KEYS, "| available:", M.available_keys())
        return 0

    if not args.stochastic:
        install_greedy_sampler()

    label = args.label or args.suite
    model_keys = _resolve_models(args.suite, args.models)
    cells_spec = _filter_cells(args.suite, args.workloads, args.configs)
    if not model_keys:
        print(
            "[error] no available models to run. Download with: uv run python -m benchmarks.download_models"
        )
        return 2

    print(f"[run] suite={args.suite} label={label} models={model_keys}")
    n_cells = sum(len(c[1]) for c in cells_spec) * len(model_keys)
    print(f"[run] {n_cells} cells (greedy={'off' if args.stochastic else 'on'})")

    out_cells: list[Cell] = []
    done = 0
    for mkey in model_keys:
        spec = M.MODELS[mkey]
        print(f"\n=== model {mkey} ({spec.path}) ===")
        tok = AutoTokenizer.from_pretrained(spec.path, use_fast=True)
        built: dict[str, object] = {}
        for wname, _cfgs, params in cells_spec:
            if wname not in built:
                try:
                    built[wname] = build_workload(wname, tok, **params)
                except Exception as e:  # dataset/tokenize failure
                    built[wname] = e
        for wname, cfgs, _params in cells_spec:
            wl = built[wname]
            for cname in cfgs:
                done += 1
                print(f"[{done}/{n_cells}] running {mkey}|{wname}|{cname} ...", flush=True)
                cell = _run_cell(spec, mkey, wname, wl, cname, args.max_model_len)
                out_cells.append(cell)
                print(_summary_line(cell), flush=True)

    path = save_results(label, out_cells)
    ok = sum(1 for c in out_cells if c.status == "ok")
    skipped = sum(1 for c in out_cells if c.status == "skip")
    errored = sum(1 for c in out_cells if c.status == "error")
    print(f"\n[done] {ok} ok, {skipped} skipped, {errored} errored -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
