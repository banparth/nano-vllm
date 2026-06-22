# The module-level eager_on_graph() helpers intentionally reach into the
# protected members of BreakableCUDAGraph / BreakableCUDAGraphCapture (and
# torch.cuda._POOL_HANDLE) from within this same module. Allow that here.
# pyright: reportPrivateUsage=false
import threading
from contextvars import ContextVar
from typing import Any, Callable

import torch

_current_capture_var: ContextVar["BreakableCUDAGraphCapture | None"] = ContextVar(
    "current_capture", default=None
)
_current_stream_var: ContextVar[torch.cuda.Stream | None] = ContextVar(
    "current_stream", default=None
)
_forked_streams_var: ContextVar[set[torch.cuda.Stream] | None] = ContextVar(
    "forked_streams", default=None
)


# copied from sglang
def _copy_output(dst: Any, src: Any) -> Any:
    """Copy src output into dst in-place where possible.

    Handles plain tensors, dataclass/object with tensor attributes,
    and dicts of tensors. Returns dst if in-place copy succeeded,
    otherwise returns src.
    """
    if torch.is_tensor(dst) and torch.is_tensor(src):
        return dst.copy_(src)

    if hasattr(dst, "__dict__") and hasattr(src, "__dict__"):
        for key, src_val in src.__dict__.items():
            dst_val = getattr(dst, key, None)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                setattr(dst, key, src_val)
        return dst

    if isinstance(dst, dict) and isinstance(src, dict):
        for key, src_val in src.items():
            dst_val = dst.get(key)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                dst[key] = src_val
        return dst

    return src


def _weak_ref_if_tensor(x: Any) -> Any:
    """Return a weak-ref tensor view (shared storage, no refcount) for tensors;
    pass-through for non-tensors. Weak-ref'ing captured args lets the shared
    mempool reclaim per-layer intermediates between segments — storage stays
    alive for each segment CUDAGraph's lifetime via its pool use_count.

    weak_ref_tensors is imported lazily: the module hard-raises on
    non-CUDA/NPU platforms, and we only reach this code during an active
    Breakable capture (which can't happen on CPU-only runners anyway)."""
    if torch.is_tensor(x):
        from nanovllm.engine.compilation.weak_ref_tensors import weak_ref_tensors

        return weak_ref_tensors(x)
    return x


def get_current_stream(device: torch.device | None = None) -> torch.cuda.Stream:
    stream = _current_stream_var.get()
    if stream is None:
        return torch.cuda.current_stream(device)
    return stream


def _stream_is_capturing() -> bool:
    """True if the current stream is in the middle of a CUDA graph capture."""
    try:
        _ = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    except RuntimeError:
        return False
    return True


def eager_on_graph():
    def decorator(inner: Callable[..., Any]):

        def wrapper(*args: Any, **kwargs: Any):
            stream = get_current_stream()
            capture = _current_capture_var.get()

            if capture is None:
                return inner(*args, **kwargs)

            capture._end_current_segment()

            output = inner(*args, **kwargs)

            captured_inner = inner
            captured_args = tuple(_weak_ref_if_tensor(a) for a in args)
            captured_kwargs = {k: _weak_ref_if_tensor(v) for k, v in kwargs.items()}
            captured_output = _weak_ref_if_tensor(output)

            def replay_fn():
                new_out = captured_inner(*captured_args, **captured_kwargs)
                return _copy_output(captured_output, new_out)

            capture._cuda_graph._break_fns.append(replay_fn)

            capture._begin_new_segment()

            return output

        return wrapper

    return decorator


# Hook torch.cuda.Stream.wait_stream to track side-stream forks/joins that happen
# during breakable capture. We need this because capture_end() on a torch
# CUDAGraph fails if there are still side streams participating in the capture
# — so before ending each segment we auto-join any forked-but-not-rejoined streams.
_original_wait_stream: (
    Callable[[torch.cuda.Stream, torch.cuda.Stream | torch._C.Stream], None] | None
) = None
_hook_lock = threading.Lock()
_hook_refcount = 0


def _raise(exc: BaseException) -> Any:
    raise exc


def _install_wait_stream_hook():
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        if _hook_refcount == 0:
            _original_wait_stream = torch.cuda.Stream.wait_stream
            # torch.cuda.Stream.wait_stream = _hooked_wait_stream  # type: ignore[assignment]
            torch.cuda.Stream.wait_stream = lambda self, stream: _raise(  # type: ignore[assignment]
                NotImplementedError("wait_stream hook is not implemented")
            )
        _hook_refcount += 1


def _uninstall_wait_stream_hook():
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        _hook_refcount -= 1
        if _hook_refcount == 0:
            assert _original_wait_stream is not None, "wait_stream hook not installed"
            torch.cuda.Stream.wait_stream = _original_wait_stream  # type: ignore[assignment]
            _original_wait_stream = None


class BreakableCUDAGraph:
    """Container holding one torch.cuda.CUDAGraph per segment plus an
    eager break function between consecutive segments."""

    def __init__(self) -> None:
        self._segments: list[torch.cuda.CUDAGraph] = []
        self._break_fns: list[Callable[[], Any]] = []

    def replay(self) -> None:
        for i, seg in enumerate(self._segments):
            seg.replay()
            if i < len(self._break_fns):
                self._break_fns[i]()


class BreakableCUDAGraphCapture:
    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        pool: torch.cuda._POOL_HANDLE | None = None,
        stream: torch.cuda.Stream | None = None,
        capture_error_mode: str = "global",
    ):
        self._cuda_graph = cuda_graph
        self._pool = pool if pool is not None else (0, 0)
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._stream_ctx = None
        self._current_capture_token = None
        self._forked_streams_token = None
        self._current_stream_token = None

    def __enter__(self) -> None:
        _install_wait_stream_hook()
        if self._stream is not None:
            self._stream_ctx = torch.cuda.stream(self._stream)
            self._stream_ctx.__enter__()

        self._current_capture_token = _current_capture_var.set(self)
        self._forked_streams_token = _forked_streams_var.set(set())
        self._current_stream_token = _current_stream_var.set(
            self._stream if self._stream is not None else torch.cuda.current_stream()
        )
        self._begin_new_segment()

    def __exit__(self, *args: object) -> None:
        try:
            self._cuda_graph._segments[-1].capture_end()
        finally:
            assert self._current_capture_token is not None
            assert self._forked_streams_token is not None
            assert self._current_stream_token is not None
            _current_capture_var.reset(self._current_capture_token)
            _forked_streams_var.reset(self._forked_streams_token)
            _current_stream_var.reset(self._current_stream_token)
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*args)
                self._stream_ctx = None
            _uninstall_wait_stream_hook()

    def _begin_new_segment(self) -> None:
        # capture_begin() fails if the stream is already capturing, so the
        # previous segment must have been ended via _end_current_segment().
        assert not _stream_is_capturing(), (
            "previous segment capture was not ended before starting a new one"
        )
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(pool=self._pool, capture_error_mode=self._capture_error_mode)
        self._cuda_graph._segments.append(graph)

    def _end_current_segment(self) -> None:
        self._cuda_graph._segments[-1].capture_end()


@eager_on_graph()
def break_on_graph():
    pass
