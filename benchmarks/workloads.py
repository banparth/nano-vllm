"""Workload builders: turn datasets / synthetic generators into prompt sets.

A ``Workload`` bundles the prompts, the sampling params, how many passes to run
(prefix workloads run cold+warm), and any references needed for scoring. The
harness dispatches on ``kind``:

- ``throughput`` : one timed pass over many prompts (random token ids).
- ``prefix``     : two passes (cold/warm) so the wall-clock speedup reveals
                   prefix-cache effectiveness without touching engine internals.
- ``latency``    : a single batch=1 stream for TTFT + per-token decode latency.
- ``accuracy``   : GSM8K exact-match (output-quality guard).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from nanovllm import SamplingParams

# Token-id ceiling for synthetic prompts. Well under the Qwen3 vocab (~151k) and
# matches the original bench.py range so numbers stay comparable.
SYNTH_VOCAB = 10000


@dataclass
class Workload:
    name: str
    kind: str  # throughput | prefix | latency | accuracy
    prompts: list  # list[list[int]] or list[str]
    sampling: SamplingParams
    passes: int = 1
    references: list[str] | None = None  # gold answers for accuracy
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Builders. Each takes a tokenizer (may be unused) + keyword scale params.
# --------------------------------------------------------------------------- #


def build_random(
    tokenizer=None,
    num_seqs: int = 256,
    min_input: int = 128,
    max_input: int = 1024,
    output_len: int = 256,
    seed: int = 0,
    **_,
) -> Workload:
    rnd = random.Random(seed)
    prompts = [
        [rnd.randint(0, SYNTH_VOCAB - 1) for _ in range(rnd.randint(min_input, max_input))]
        for _ in range(num_seqs)
    ]
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    return Workload(
        "random", "throughput", prompts, sp, meta={"num_seqs": num_seqs, "output_len": output_len}
    )


def build_sharegpt(
    tokenizer,
    num_prompts: int = 512,
    min_turns: int = 4,
    max_turns: int = 6,
    max_prompt_tokens: int = 2048,
    output_len: int = 64,
    seed: int = 0,
    dataset: str | None = None,
    **_,
) -> Workload:
    from nanovllm.utils.sharegpt import (
        DEFAULT_SHAREGPT_PATH,
        build_multiturn_prompts,
        ensure_sharegpt,
        load_conversations,
    )

    path = ensure_sharegpt(dataset or DEFAULT_SHAREGPT_PATH)
    convs = load_conversations(path, min_turns=min_turns)
    prompts = build_multiturn_prompts(
        convs,
        tokenizer,
        max_turns=max_turns,
        max_prompt_tokens=max_prompt_tokens,
        max_prompts=num_prompts,
    )
    random.Random(seed).shuffle(prompts)
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    return Workload(
        "sharegpt",
        "prefix",
        prompts,
        sp,
        passes=2,
        meta={"n_prompts": len(prompts), "output_len": output_len},
    )


def build_longprefix(
    tokenizer=None,
    shared_prefix: int = 2048,
    num_prompts: int = 128,
    suffix: int = 32,
    output_len: int = 64,
    seed: int = 0,
    **_,
) -> Workload:
    """Many prompts that share one long identical prefix + a unique short suffix.

    Maximizes prefix-cache reuse (every prompt after the first reuses the shared
    prefix blocks) and is the workload where a CPU KV tier should pay off most.
    """
    prefix = list(range(100, 100 + shared_prefix))  # constant across prompts
    rnd = random.Random(seed)
    prompts = [
        prefix + [rnd.randint(0, SYNTH_VOCAB - 1) for _ in range(suffix)]
        for _ in range(num_prompts)
    ]
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    return Workload(
        "longprefix",
        "prefix",
        prompts,
        sp,
        passes=2,
        meta={"shared_prefix": shared_prefix, "num_prompts": num_prompts},
    )


def build_latency(
    tokenizer=None, prompt_len: int = 512, output_len: int = 128, seed: int = 0, **_
) -> Workload:
    rnd = random.Random(seed)
    prompts = [[rnd.randint(0, SYNTH_VOCAB - 1) for _ in range(prompt_len)]]
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    return Workload(
        "latency", "latency", prompts, sp, meta={"prompt_len": prompt_len, "output_len": output_len}
    )


def build_gsm8k(
    tokenizer, num_questions: int = 200, output_len: int = 512, seed: int = 0, **_
) -> Workload:
    from benchmarks.gsm8k import INSTRUCTION, load_gsm8k

    items = load_gsm8k(n=num_questions, seed=seed)
    golds = [g for _, g in items]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION.format(q=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q, _ in items
    ]
    # Allow EOS so the model can stop after the answer; greedy makes it stable.
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=False)
    return Workload(
        "gsm8k",
        "accuracy",
        prompts,
        sp,
        references=golds,
        meta={"n_questions": len(golds), "output_len": output_len},
    )


BUILDERS: dict[str, Callable[..., Workload]] = {
    "random": build_random,
    "sharegpt": build_sharegpt,
    "longprefix": build_longprefix,
    "latency": build_latency,
    "gsm8k": build_gsm8k,
}

ALL_WORKLOADS = list(BUILDERS)


def build_workload(name: str, tokenizer, **params) -> Workload:
    if name not in BUILDERS:
        raise KeyError(f"unknown workload '{name}'. Known: {ALL_WORKLOADS}")
    return BUILDERS[name](tokenizer, **params)
