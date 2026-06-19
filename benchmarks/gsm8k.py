"""GSM8K loader + answer scoring for the accuracy workload.

Kept dependency-free (stdlib only, no ``nanovllm`` import) and self-downloading
in the same spirit as ``nanovllm/utils/sharegpt.py`` so the suite needs no extra
packages. GSM8K answers are integers; the gold answer is the number after the
``####`` marker in the dataset's ``answer`` field.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
DEFAULT_GSM8K_PATH = os.path.expanduser("~/datasets/gsm8k/test.jsonl")

# Instruction nudges the model to emit "#### <answer>" so extraction is robust.
INSTRUCTION = (
    "Solve the following grade-school math problem. Reason step by step, "
    "then give the final answer on its own line as '#### <number>'.\n\n{q}"
)

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def ensure_gsm8k(path: str = DEFAULT_GSM8K_PATH, url: str = GSM8K_TEST_URL) -> str:
    """Return ``path``, downloading the GSM8K test split there if missing."""
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[gsm8k] downloading test split to {path}")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def _gold_from_answer(answer_field: str) -> str:
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    nums = _NUMBER_RE.findall(answer_field)
    return nums[-1].replace(",", "") if nums else ""


def load_gsm8k(path: str = DEFAULT_GSM8K_PATH, n: int | None = 200, seed: int = 0
               ) -> list[tuple[str, str]]:
    """Load up to ``n`` (question, gold_answer) pairs, deterministically sampled."""
    path = ensure_gsm8k(path)
    items: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("question", "").strip()
            gold = _gold_from_answer(row.get("answer", ""))
            if q and gold:
                items.append((q, gold.replace(",", "")))
    random.Random(seed).shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def extract_prediction(text: str) -> str | None:
    """Pull the model's final numeric answer out of a generation.

    Prefers a number after the last ``####`` marker; otherwise falls back to
    the last number anywhere in the text.
    """
    if "####" in text:
        tail = text.split("####")[-1]
        m = _NUMBER_RE.search(tail)
        if m:
            return m.group(0).replace(",", "")
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return a == b


def score_predictions(generations: list[str], golds: list[str]) -> dict:
    """Exact-match accuracy of extracted predictions vs gold answers."""
    correct = 0
    for text, gold in zip(generations, golds):
        pred = extract_prediction(text)
        if pred is not None and _num_eq(pred, gold):
            correct += 1
    total = len(golds)
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
    }
