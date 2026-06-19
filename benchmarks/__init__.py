"""Unified correctness + benchmark harness for nano-vllm.

This package turns the previously scattered, single-model scripts into one
registry-driven suite so a change to the engine (e.g. the CPU KV-cache tier)
can be both proven correct and measured against a saved baseline across
multiple Qwen3 sizes, datasets/workloads and engine configurations.

Layout
------
- ``models.py``    : registry of Qwen3 sizes + path resolution + download.
- ``configs.py``   : named engine-config presets + per-model resolution.
- ``workloads.py`` : prompt-set builders (random/sharegpt/longprefix/latency/gsm8k).
- ``engine.py``    : build/destroy/warmup helpers + engine stats readout.
- ``metrics.py``   : throughput / latency (TTFT) timing + output checksum.
- ``results.py``   : result schema, environment metadata, JSON I/O.
- ``run.py``       : matrix runner CLI  (python -m benchmarks.run).
- ``compare.py``   : baseline-vs-candidate diff + regression gate.
- ``download_models.py`` : fetch missing Qwen3 sizes.

Everything assumes a single H100 (``tensor_parallel_size=1``).
"""
