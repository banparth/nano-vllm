# Simple CI gate for nano-vllm.
#
# Run `make ci` before pushing. It mirrors what an automated CI check enforces:
# formatting, import ordering, and the full test suite. It fails fast on the
# first problem, so a non-zero exit cleanly blocks the gate.
#
# `uv run` automatically syncs the locked environment (including the dev group),
# so these targets work from a fresh checkout with nothing pre-installed.

.PHONY: ci format-check lint test format

# --- CI gate -----------------------------------------------------------------
# The single target to treat as the gate.
ci: format-check lint test

# --- Individual checks (composed by `ci`) ------------------------------------
# Verify formatting is consistent without modifying files.
format-check:
	uv run ruff format --check .

# Verify import ordering / lint rules without modifying files.
lint:
	uv run ruff check .

# Run the full test suite.
test:
	uv run pytest

# --- Local convenience -------------------------------------------------------
# Auto-fix imports and formatting in place (the fix-up counterpart to the gate).
format:
	uv run ruff check --fix .
	uv run ruff format .
