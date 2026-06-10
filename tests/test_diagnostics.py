"""Tests for the diagnostics module."""

from typing import Any

import pytest
import numpy as np
from ml_switcheroo.diagnostics import (
    TracebackReconstructor,
    debug_shapes,
    estimate_flops,
    memory_profiler,
    NumericalAnomalyDetector,
    to_graphviz,
    to_html,
)
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device, DeviceType
from typing import NoReturn


def test_traceback_reconstructor() -> None:
    """Test TracebackReconstructor formatting."""
    exc = ValueError("test error")
    formatted = TracebackReconstructor.format_traceback(exc)
    assert "TracebackReconstructor: test error" in formatted


def test_debug_shapes() -> None:
    """Test debug_shapes function."""

    def dummy_model(x: Any) -> Any:
        return x + 1.0

    res = debug_shapes(dummy_model, (2, 2))
    assert "| input | (2, 2) | float64 |" in res
    assert "| output | (2, 2) | float64 |" in res

    def dummy_model_no_shape(x: Any) -> int:
        return 5

    res = debug_shapes(dummy_model_no_shape, (2, 2))
    assert "| output | unknown | float64 |" in res

    def failing_model(x: Any) -> NoReturn:
        raise RuntimeError("fail")

    res_fail = debug_shapes(failing_model, (2, 2))
    assert "| Node | Shape | DType |" in res_fail
    assert "input" not in res_fail


def test_estimate_flops() -> None:
    """Test FLOPs estimation."""
    graph = LogicalGraph(name="test")
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="MatMul", shape_metadata=(10, 10))
    graph.nodes["n3"] = LogicalNode(id="n3", op_type="Add")  # no shape
    graph.nodes["n4"] = LogicalNode(id="n4", op_type="Foo")  # unknown op

    class BadShape:
        def __iter__(self) -> Any:
            raise TypeError("bad iterator")

    graph.nodes["n5"] = LogicalNode(id="n5", op_type="Add", shape_metadata=BadShape())

    flops = estimate_flops(graph)
    # n1: 100
    # n2: 100 (fixed rough estimate)
    # n3: 1 (fallback)
    # n4: 0
    # n5: 1 (fallback exception)
    assert flops == 202


def test_memory_profiler() -> None:
    """Test memory usage profiling."""
    graph = LogicalGraph(name="test")
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="Add")  # no shape

    class BadShape:
        def __iter__(self) -> Any:
            raise TypeError("bad iterator")

    graph.nodes["n3"] = LogicalNode(id="n3", op_type="Add", shape_metadata=BadShape())

    mem = memory_profiler(graph)
    # n1: 100 * 4 = 400
    # n2: 4 (fallback)
    # n3: 4 (fallback exception)
    assert mem == 408


def test_numerical_anomaly_detector() -> None:
    """Test NaN/Inf checking."""
    device = Device(DeviceType.CPU, 0)

    # Valid
    t1 = Tensor(
        data=np.array([1.0, 2.0]), shape=(2,), dtype=DType.Float32, device=device
    )
    NumericalAnomalyDetector.check(t1)

    # None data
    t_none = Tensor(data=None, shape=(2,), dtype=DType.Float32, device=device)
    NumericalAnomalyDetector.check(t_none)

    # NaN
    t2 = Tensor(
        data=np.array([1.0, np.nan]), shape=(2,), dtype=DType.Float32, device=device
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        NumericalAnomalyDetector.check(t2)

    # Non-array-like
    class NonArray:
        pass

    t3 = Tensor(data=NonArray(), shape=(2,), dtype=DType.Float32, device=device)
    # Should silently pass or catch TypeError
    NumericalAnomalyDetector.check(t3)


def test_to_graphviz() -> None:
    """Test Graphviz dot export."""
    graph = LogicalGraph(name="test")
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])

    dot = to_graphviz(graph)
    assert "digraph G {" in dot
    assert 'label="Input"' in dot
    assert '"n1" -> "n2"' in dot


def test_to_html() -> None:
    """Test HTML export."""
    graph = LogicalGraph(name="test")
    html = to_html(graph)
    assert "<h1>IR Graph</h1>" in html
