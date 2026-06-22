import pytest
import torch

from nanovllm.engine.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    break_on_graph,
)


def test_breakable_cuda_graph_capture_enter_exit():
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream()
    capture = BreakableCUDAGraphCapture(graph, stream=stream)

    with capture:
        pass
    assert len(graph._segments) == 1
    assert len(graph._break_fns) == 0


def test_breakable_cuda_graph_capture_enter_exit_with_stream():
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream()
    capture = BreakableCUDAGraphCapture(graph, stream=stream)
    with capture:
        x = torch.randn(10, device="cuda")
        y = torch.randn(10, device="cuda")
        z = x + y
    assert len(graph._segments) == 1
    assert len(graph._break_fns) == 0


def test_break_on_graph():
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream()
    capture = BreakableCUDAGraphCapture(graph, stream=stream)
    with capture:
        x = torch.randn(10, device="cuda")
        y = torch.randn(10, device="cuda")
        break_on_graph()
        z = x + y
    assert len(graph._segments) == 2
    assert len(graph._break_fns) == 1


def test_capture_with_forked_streams():
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream()
    capture = BreakableCUDAGraphCapture(graph, stream=stream)
    with capture:
        x = torch.randn(10, device="cuda")
        y = torch.randn(10, device="cuda")
        child_stream = torch.cuda.Stream()
        with pytest.raises(NotImplementedError):
            child_stream.wait_stream(stream)
    #     with torch.cuda.stream(stream=child_stream):
    #         z = x + y
    #     stream.wait_stream(child_stream)
    # assert len(graph._segments) == 1
    # assert len(graph._break_fns) == 0
    assert child_stream.wait_stream(stream) == None


def test_capture_with_forked_streams_break():
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream()
    capture = BreakableCUDAGraphCapture(graph, stream=stream)
    with capture:
        x = torch.randn(10, device="cuda")
        y = torch.randn(10, device="cuda")
        child_stream = torch.cuda.Stream()
        with pytest.raises(NotImplementedError):
            child_stream.wait_stream(stream)
    #     with torch.cuda.stream(child_stream):
    #         z = x + y
    #     break_on_graph()
    #     stream.wait_stream(child_stream)
    # assert len(graph._segments) == 1
    # assert len(graph._break_fns) == 0
    assert child_stream.wait_stream(stream) == None
