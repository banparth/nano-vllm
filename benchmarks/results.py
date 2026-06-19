"""Result schema + environment capture + JSON I/O for the suite.

A run is a flat list of *cells*. Each cell is one (model, workload, config)
combination plus the metrics measured for it (or a skip/error note). The
``env`` block records enough provenance (git commit, GPU, library versions)
that two result files are only compared when they came from comparable setups.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def env_metadata() -> dict[str, Any]:
    """Capture provenance for the current run (best-effort, never raises)."""
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "gpu": gpu_name,
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "flash_attn": _pkg_version("flash-attn") or _pkg_version("flash_attn"),
        "triton": _pkg_version("triton"),
    }


@dataclass
class Cell:
    model: str
    workload: str
    config: str
    status: str = "ok"                       # ok | skip | error
    note: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model}|{self.workload}|{self.config}"


def cell_key(model: str, workload: str, config: str) -> str:
    return f"{model}|{workload}|{config}"


def save_results(label: str, cells: list[Cell], path: str | None = None) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(RESULTS_DIR, f"{label}.json")
    payload = {
        "label": label,
        "env": env_metadata(),
        "cells": [asdict(c) for c in cells],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def load_results(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def cells_by_key(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {cell_key(c["model"], c["workload"], c["config"]): c for c in payload.get("cells", [])}
