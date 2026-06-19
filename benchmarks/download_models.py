"""Download the registry's Qwen3 sizes into ``~/huggingface/Qwen3-<size>``.

By default downloads every registry model that is not already complete; pass
``--models`` to restrict. Uses ``huggingface_hub.snapshot_download`` so it works
with the same auth/cache as the rest of the toolchain.

    uv run python -m benchmarks.download_models                # all missing
    uv run python -m benchmarks.download_models --models 32B   # just one
"""

from __future__ import annotations

import argparse

from benchmarks.models import ALL_KEYS, MODELS, weights_complete


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="comma list of keys (default: all registry models)")
    ap.add_argument("--force", action="store_true", help="re-download even if already complete")
    args = ap.parse_args()

    keys = [k.strip() for k in args.models.split(",")] if args.models else list(ALL_KEYS)
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        print(f"[error] unknown model keys: {unknown}. Known: {ALL_KEYS}")
        return 2

    from huggingface_hub import snapshot_download

    for key in keys:
        spec = MODELS[key]
        if spec.available and not args.force:
            print(f"[skip] {key}: already complete at {spec.path}")
            continue
        print(f"[download] {key}: {spec.repo} -> {spec.path} (~{spec.weight_gib:.0f} GiB)")
        snapshot_download(repo_id=spec.repo, local_dir=spec.path)
        status = "ok" if weights_complete(spec.path) else "INCOMPLETE"
        print(f"[download] {key}: {status}")

    print("\navailable now:", [k for k in ALL_KEYS if MODELS[k].available])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
