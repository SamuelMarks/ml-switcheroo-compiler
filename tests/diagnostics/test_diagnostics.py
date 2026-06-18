"""Unit tests for the diagnostics module of the ml_switcheroo_compiler framework.

This module contains comprehensive unit tests verifying the correctness of traceback
reconstruction, shape debugging, FLOPs estimation, memory profiling, numerical anomaly
detection, and graph visualization utilities (Graphviz and HTML exports).
"""

from typing import NoReturn

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.diagnostics import (
    check_numerical_anomaly,
    format_traceback,
    debug_shapes,
    estimate_flops,
    memory_profiler,
    to_graphviz,
    to_html,
)


def test_traceback_reconstructor() -> None:
    """Verifies that TracebackReconstructor correctly formats exception tracebacks.

    Returns:
    None
    """
    exc = ValueError("test error")
    formatted = format_traceback(exc)
    assert "TracebackReconstructor: test error" in formatted


def test_debug_shapes() -> None:
    """Verifies the shape debugging utility under various model execution scenarios.

    Returns:
    None
    """

    def dummy_model(x: object) -> object:
        """Dummy model.

        Args:
            x (object): The first input tensor.

        Returns:
            object: The resulting output.
        """
        return x + 1.0

    res = debug_shapes(dummy_model, (2, 2))
    assert "| input | (2, 2) | float64 |" in res
    assert "| output | (2, 2) | float64 |" in res

    def dummy_model_no_shape(x: object) -> int:
        """Dummy model no shape.

        Args:
            x (object): The first input tensor.

        Returns:
            int: The resulting output.
        """
        return 5

    res = debug_shapes(dummy_model_no_shape, (2, 2))
    assert "| output | unknown | float64 |" in res

    def failing_model(x: object) -> NoReturn:
        """Failing model.

        Args:
            x (object): The first input tensor.

        Returns:
            NoReturn: The resulting output.
        """
        msg = "fail"
        raise RuntimeError(msg)

    res_fail = debug_shapes(failing_model, (2, 2))
    assert "| Node | Shape | DType |" in res_fail
    assert "input" not in res_fail


def test_estimate_flops() -> None:
    """Verifies the FLOPs estimation utility for logical graphs.

    Returns:
    None
    """
    graph = LogicalGraph(name="test")
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="MatMul", shape_metadata=(10, 10))
    graph.nodes["n3"] = LogicalNode(id="n3", op_type="Add")  # no shape
    graph.nodes["n4"] = LogicalNode(id="n4", op_type="Foo")  # unknown op

    class BadShape:
        """Bad Shape class."""

        def __iter__(self) -> object:
            """Iter.

            Returns:
                object: The resulting output.
            """
            msg = "bad iterator"
            raise TypeError(msg)

    graph.nodes["n5"] = LogicalNode(id="n5", op_type="Add", shape_metadata=BadShape())

    flops = estimate_flops(graph)
    # n1: 100
    # n2: 100 (fixed rough estimate)
    # n3: 1 (fallback)
    # n4: 0
    # n5: 1 (fallback exception)
    assert flops == 202


def test_memory_profiler() -> None:
    """Verifies the memory profiling utility for logical graphs.

    Returns:
    None
    """
    graph = LogicalGraph(name="test")
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="Add")  # no shape

    class BadShape:
        """Bad Shape class."""

        def __iter__(self) -> object:
            """Iter.

            Returns:
                object: The resulting output.
            """
            msg = "bad iterator"
            raise TypeError(msg)

    graph.nodes["n3"] = LogicalNode(id="n3", op_type="Add", shape_metadata=BadShape())

    mem = memory_profiler(graph)
    # n1: 100 * 4 = 400
    # n2: 4 (fallback)
    # n3: 4 (fallback exception)
    assert mem == 408


def test_numerical_anomaly_detector() -> None:
    """Verifies the numerical anomaly detector's ability to identify NaNs and Infs.

    Returns:
    None
    """
    device = Device(DeviceType.CPU, 0)

    # Valid
    t1 = Tensor(
        data=np.array([1.0, 2.0]),
        shape=(2,),
        dtype=DType.Float32,
        device=device,
    )
    check_numerical_anomaly(t1)

    # None data
    t_none = Tensor(data=None, shape=(2,), dtype=DType.Float32, device=device)
    check_numerical_anomaly(t_none)

    # NaN
    t2 = Tensor(
        data=np.array([1.0, np.nan]),
        shape=(2,),
        dtype=DType.Float32,
        device=device,
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        check_numerical_anomaly(t2)

    # Non-array-like
    class NonArray:
        """Non Array class."""

    t3 = Tensor(data=NonArray(), shape=(2,), dtype=DType.Float32, device=device)
    # Should silently pass or catch TypeError
    check_numerical_anomaly(t3)


def test_to_graphviz() -> None:
    """Verifies the Graphviz DOT export utility for logical graphs.

    Returns:
    None
    """
    graph = LogicalGraph(name="test")
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])

    dot = to_graphviz(graph)
    assert "digraph G {" in dot
    assert 'label="Input"' in dot
    assert '"n1" -> "n2"' in dot


def test_to_html() -> None:
    """Verifies the HTML export utility for logical graphs.

    Returns:
    None
    """
    graph = LogicalGraph(name="test")
    html = to_html(graph)
    assert "<h1>IR Graph</h1>" in html
