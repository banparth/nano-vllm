<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Evaluation Suite (correctness + benchmarks)

The `benchmarks/` package + `tests/` harness let you prove a change is **correct**
and measure whether it is a **meaningful improvement**, across multiple Qwen3
sizes, multiple workloads and multiple engine configs. Everything assumes a
single H100 (`tensor_parallel_size=1`) and is driven through `uv run`.

### Models

`benchmarks/models.py` is a registry of Qwen3 dense sizes (`0.6B, 1.7B, 4B, 8B,
14B, 32B`) under `~/huggingface/Qwen3-<size>`. Download the ones you don't have:

```bash
uv run python -m benchmarks.download_models            # all missing sizes
uv run python -m benchmarks.download_models --models 32B
```

The harness auto-detects which models are fully downloaded; unavailable ones are
skipped. Larger sizes (14B/32B) automatically use a memory-constrained config so
they fit on one 80GB card.

### Correctness

Two complementary layers:

1. **Self-consistency + golden regression** (`tests/correctness.py`): a matrix of
   (model x prompt-slice x config). For each cell it runs three scenarios that
   must agree token-for-token and within a log-prob tolerance:
   `cold` (fresh), `warm` (prefix cache hit), `tight` (small KV budget -> eviction).
   It also compares against a per-cell golden in `tests/goldens/`.

   ```bash
   uv run python tests/correctness.py --bless --models 0.6B,1.7B   # first time
   uv run python tests/correctness.py --models 0.6B                # edit loop
   uv run python tests/correctness.py                              # all available
   ```

2. **Ground-truth oracle** (`tests/oracle.py`): greedy-decodes the same prompts
   with both nano-vllm and HuggingFace `transformers` (in isolated subprocesses
   so they never share GPU memory) and checks they agree. Tolerant by default
   (flash-attn vs HF kernels diverge late in FP); fails only on a first-token
   mismatch or under `--strict`.

   ```bash
   uv run python tests/oracle.py --models 0.6B,1.7B
   ```

### Benchmarks

`benchmarks/run.py` runs a suite of (model x workload x config) cells, each in a
fresh engine, and writes `results/<label>.json`. Workloads: `random`
(throughput), `sharegpt` + `longprefix` (prefix-cache cold/warm, reported as a
wall-clock speedup), `latency` (single-stream TTFT + decode tok/s), `gsm8k`
(output-quality accuracy).

```bash
uv run python -m benchmarks.run --suite smoke                 # fast sanity (0.6B)
uv run python -m benchmarks.run --suite full --label baseline # full sweep
uv run python -m benchmarks.run --suite cpu-tier              # CPU KV tier OFF vs ON
# filters:
uv run python -m benchmarks.run --suite full --models 1.7B,4B --workloads latency,gsm8k
```

### The "meaningful improvement" workflow

Capture a baseline, make your change, capture a candidate, then diff:

```bash
git checkout main
uv run python -m benchmarks.run --suite full --label baseline

git checkout my-change
uv run python -m benchmarks.run --suite full --label candidate

uv run python -m benchmarks.compare results/baseline.json results/candidate.json
```

`compare.py` aligns cells by `model|workload|config` and gates on a stable
primary metric per workload: it prints per-cell deltas and **exits non-zero** on
any regression (throughput/latency beyond `--tol`, GSM8K accuracy drop) or
**FAIL** (the deterministic output checksum changed). Decoding is greedy by
default so checksums are comparable across branches.

> The older standalone scripts (`bench.py`, `bench_sharegpt.py`,
> `sweep_cpucache.py`) still work and are left as-is; the unified `benchmarks/`
> suite supersedes them. Run outputs go to `results/` (gitignored); the
> reference goldens in `tests/goldens/` are committed.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)