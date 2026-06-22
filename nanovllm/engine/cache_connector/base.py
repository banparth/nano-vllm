"""KV cache connector abstraction, ported from vLLM's ``KVConnectorBase_V1``
(``vllm/distributed/kv_transfer/kv_connector/v1/base.py``).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence

if TYPE_CHECKING:
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.utils.context import Context


class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process.
    SCHEDULER = enum.auto()
    # Connector running in the worker (model-runner) process.
    WORKER = enum.auto()


class KVConnectorMetadata:
    """Abstract metadata used to communicate Scheduler KVConnector -> Worker
    KVConnector for one engine step. Must be picklable: under tensor parallelism
    it crosses the process boundary to the worker model runners."""

    pass


class KVConnectorBase:
    """Base class for KV connectors (mirrors vLLM's ``KVConnectorBase_V1``)."""

    def __init__(self, config: Config, role: KVConnectorRole):
        self._connector_metadata: KVConnectorMetadata | None = None
        self.config = config
        self._role = role

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler. Called by the model
        runner every time before model execution."""
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata. Called by the model runner every time
        after model execution."""
        self._connector_metadata = None

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata. Should only be called inside the connector
        while metadata is bound."""
        assert self._connector_metadata is not None
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        """Whether connector metadata is currently bound."""
        return self._connector_metadata is not None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Initialize with the KV caches: a dict of ``{layer_name: kv_cache}``,
        each value the paged buffer for one layer."""
        return

    def handle_preemptions(self, connector_metadata: KVConnectorMetadata) -> None:
        """Handle preempted requests / evicted blocks BEFORE they are
        overwritten. Needed for connectors that use async saves."""
        return

    def start_load_kv(self, forward_context: "Context", **kwargs: Any) -> None:
        """Start loading KV from the connector into nano-vllm's paged buffer,
        before the forward pass (may be async, paired with
        ``wait_for_layer_load``)."""
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until layer ``layer_name``'s KV load is complete. Called from
        within the attention layer for layer-by-layer pipelining."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """Start saving a layer of KV from the paged buffer to the connector
        (may be async). Called from within the attention layer."""
        return

    def wait_for_save(self) -> None:
        """Block until all saves are done, as the forward context exits, to
        prevent overwriting the paged buffer before saving completes."""
        return

    def get_finished(
        self, finished_req_ids: set[int]
    ) -> tuple[set[int] | None, set[int] | None]:
        """Notify ids of finished requests; return ids that finished async
        transfer as ``(sending/saving, recving/loading)``."""
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Block ids that failed to load (empty if none)."""
        return set()

    def shutdown(self) -> None:
        """Shutdown the connector: flush/cleanup any async work."""
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockManager") -> None:
        """Bind the GPU block pool for per-block status tracking (e.g. ref counts
        or iterating the prefix cache)."""
        return

    def get_num_new_matched_tokens(
        self, request: Sequence, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Number of new tokens that can be loaded from the external KV cache
        beyond ``num_computed_tokens``. Must be side-effect free.

        Returns ``(num_external_tokens, load_is_async)``; the count is ``None``
        if the connector needs to be queried again later, and ``load_is_async``
        must be ``False`` when the count is 0."""
        return 0, False

    def update_state_after_alloc(
        self, request: Sequence, blocks: list[int], num_external_tokens: int
    ) -> None:
        """Update connector state after the block manager allocates ``blocks``
        for ``num_external_tokens`` to be loaded into."""
        return

    def build_connector_meta(
        self, seqs: list[Sequence], is_prefill: bool
    ) -> KVConnectorMetadata:
        """Build (and reset) this step's connector metadata for the worker.

        nano-vllm has no ``SchedulerOutput``, so the scheduled ``seqs`` plus
        ``is_prefill`` stand in for it here."""
        return KVConnectorMetadata()

    def on_new_request(self, request: Sequence) -> None:
        """Called when a new request is added, for connector bookkeeping."""
        return

    def request_finished(
        self, request: Sequence, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called once when a request finishes, before its blocks are freed.
        Returns ``(defer_free, extra)`` -- ``True`` keeps the blocks for an async
        save."""
        return False, None
