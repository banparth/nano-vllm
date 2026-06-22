"""Compare two result files (baseline vs candidate) and gate on regressions.

Aligns cells by ``model|workload|config`` and judges each on a single, stable
*primary* metric per workload kind (plus the output checksum and the cell
status), so the verdict is meaningful and low-noise:

- throughput (random) -> total_tok_s        (higher better)
- prefix (sharegpt/..) -> warm_total_tok_s   (higher better)
- latency              -> decode_tok_s       (higher better)
- accuracy (gsm8k)     -> accuracy           (higher better, gated in points)

A cell is a FAIL if its deterministic output changed (checksum differs) or it
regressed from ok to skip/error; a REGRESSION if the primary metric dropped
beyond tolerance. The process exits non-zero on any FAIL or REGRESSION, so it
works as a CI/edit-loop gate.

    uv run python -m benchmarks.compare results/baseline.json results/candidate.json
"""

from __future__ import annotations

import argparse

from benchmarks.results import cells_by_key, load_results

PRIMARY = {
    "random": ("total_tok_s", "up"),
    "sharegpt": ("warm_total_tok_s", "up"),
    "longprefix": ("warm_total_tok_s", "up"),
    "latency": ("decode_tok_s", "up"),
    "gsm8k": ("accuracy", "up_pts"),
}

# Extra informational metrics shown per workload (not gated).
INFO = {
    "random": ["output_tok_s", "peak_gpu_gib"],
    "sharegpt": ["wall_speedup", "cold_total_tok_s"],
    "longprefix": ["wall_speedup", "cold_total_tok_s"],
    "latency": ["ttft_ms", "inter_token_p99_ms"],
    "gsm8k": ["output_tok_s"],
}


def _fmt_delta(base: float, cand: float, kind: str) -> str:
    if kind == "up_pts":
        return f"{(cand - base) * 100:+.1f}pt"
    if base == 0:
        return "n/a"
    return f"{(cand - base) / base * 100:+.1f}%"


def _judge(base_cell, cand_cell, thr: float, acc_pts: float, allow_output: bool):
    """Return (verdict, detail). verdict in OK/IMPROVED/REGRESSION/FAIL."""
    if cand_cell is None:
        return "MISSING", "absent in candidate"
    if base_cell["status"] != "ok" or cand_cell["status"] != "ok":
        if base_cell["status"] == "ok" and cand_cell["status"] != "ok":
            return "FAIL", f"ok -> {cand_cell['status']} ({cand_cell.get('note', '')})"
        return "SKIP", f"{base_cell['status']} -> {cand_cell['status']}"

    bm, cm = base_cell["metrics"], cand_cell["metrics"]

    if (
        not allow_output
        and "checksum" in bm
        and "checksum" in cm
        and bm["checksum"] != cm["checksum"]
    ):
        return "FAIL", "output checksum changed"

    metric, direction = PRIMARY.get(base_cell["workload"], ("total_tok_s", "up"))
    if metric not in bm or metric not in cm:
        return "OK", "no primary metric"
    base, cand = bm[metric], cm[metric]

    if direction == "up_pts":
        if cand < base - acc_pts / 100:
            return "REGRESSION", f"{metric} {base * 100:.1f}% -> {cand * 100:.1f}%"
        if cand > base + acc_pts / 100:
            return "IMPROVED", f"{metric} {base * 100:.1f}% -> {cand * 100:.1f}%"
        return "OK", ""
    # higher-is-better ratio metric
    if base > 0 and cand < base * (1 - thr):
        return "REGRESSION", f"{metric} {_fmt_delta(base, cand, direction)}"
    if base > 0 and cand > base * (1 + thr):
        return "IMPROVED", f"{metric} {_fmt_delta(base, cand, direction)}"
    return "OK", ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument(
        "--tol",
        type=float,
        default=0.03,
        help="fractional tolerance for tok/s + latency (default 0.03)",
    )
    ap.add_argument(
        "--accuracy-pts", type=float, default=1.0, help="accuracy tolerance in points (default 1.0)"
    )
    ap.add_argument(
        "--allow-output-change", action="store_true", help="don't FAIL on checksum changes"
    )
    args = ap.parse_args()

    base = load_results(args.baseline)
    cand = load_results(args.candidate)
    b_cells = cells_by_key(base)
    c_cells = cells_by_key(cand)

    print(
        f"baseline : {args.baseline}  ({base['env'].get('git_commit')} on {base['env'].get('git_branch')})"
    )
    print(
        f"candidate: {args.candidate}  ({cand['env'].get('git_commit')} on {cand['env'].get('git_branch')})"
    )
    if base["env"].get("gpu") != cand["env"].get("gpu"):
        print(f"[warn] GPU differs: {base['env'].get('gpu')} vs {cand['env'].get('gpu')}")
    print()

    hdr = f"{'cell':<34} {'metric':<17} {'baseline':>11} {'candidate':>11} {'delta':>9}  verdict"
    print(hdr)
    print("-" * len(hdr))

    n_fail = n_regress = n_improve = 0
    for key in sorted(b_cells):
        bcell = b_cells[key]
        ccell = c_cells.get(key)
        verdict, detail = _judge(
            bcell, ccell, args.tol, args.accuracy_pts, args.allow_output_change
        )
        wl = bcell["workload"]
        metric, direction = PRIMARY.get(wl, ("total_tok_s", "up"))
        bm = bcell.get("metrics", {})
        cm = (ccell or {}).get("metrics", {})
        if metric in bm and metric in cm:
            bv, cv = bm[metric], cm[metric]
            if direction == "up_pts":
                bstr, cstr = f"{bv * 100:.1f}%", f"{cv * 100:.1f}%"
            else:
                bstr, cstr = f"{bv:.1f}", f"{cv:.1f}"
            delta = _fmt_delta(bv, cv, direction)
        else:
            metric, bstr, cstr, delta = "-", "-", "-", "-"
        tag = {
            "REGRESSION": "REGRESSION",
            "FAIL": "FAIL  ***",
            "IMPROVED": "improved",
            "OK": "ok",
            "SKIP": "skip",
            "MISSING": "MISSING",
        }.get(verdict, verdict)
        line = f"{key:<34} {metric:<17} {bstr:>11} {cstr:>11} {delta:>9}  {tag}"
        if detail and verdict in ("FAIL", "REGRESSION", "MISSING"):
            line += f"  [{detail}]"
        print(line)
        if verdict == "FAIL":
            n_fail += 1
        elif verdict == "REGRESSION":
            n_regress += 1
        elif verdict == "IMPROVED":
            n_improve += 1

    only_cand = sorted(set(c_cells) - set(b_cells))
    if only_cand:
        print(f"\n[new cells only in candidate] {only_cand}")

    print(f"\n{n_improve} improved, {n_regress} regressed, {n_fail} failed")
    return 1 if (n_fail or n_regress) else 0


if __name__ == "__main__":
    raise SystemExit(main())
