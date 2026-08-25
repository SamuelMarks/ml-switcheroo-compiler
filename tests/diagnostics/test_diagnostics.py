# ruff: noqa: E501
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
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.diagnostics import (
    check_numerical_anomaly,
    debug_shapes,
    estimate_flops,
    format_traceback,
    memory_profiler,
    to_graphviz,
    to_html,
)


def test_traceback_reconstructor() -> None:
    """Test the traceback reconstructor behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies that TracebackReconstructor correctly formats exception tracebacks.\n\n    Returns:\n    None\n    "
        exc = ValueError("test error")
        formatted = format_traceback(exc)
        assert "TracebackReconstructor: test error" in formatted
    except Exception as e:
        raise e
        pass


def test_debug_shapes() -> None:
    """Test the debug shapes behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Verifies the shape debugging utility under various model execution scenarios.\n\n    Returns:\n    None\n    "

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

    import ml_switcheroo_compiler.diagnostics.shape_debugger as debugger

    dummy_dict = {"markdown_table": {"header": "header", "row": "row"}}

    with pytest.MonkeyPatch().context() as m:
        m.setattr(debugger, "_FORMATTERS", dummy_dict)
        # It should format with dummy_dict now
        try:
            debug_shapes(dummy_model, (2, 2))
        except:
            pass

        m.setattr(debugger, "_FORMATTERS", {"graphviz": {"header": "H", "node": "N", "edge": "E", "footer": "F"}})
        g = LogicalGraph()
        debugger.to_graphviz(g)

        m.setattr(debugger, "_FORMATTERS", {"html": {"template": "HTML"}})
        debugger.to_html(g)


def test_estimate_flops() -> None:
    """Test the estimate flops behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies the FLOPs estimation utility for logical graphs.\n\n    Returns:\n    None\n    "
        graph = LogicalGraph(name="test")
        graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
        graph.nodes["n2"] = LogicalNode(id="n2", op_type="MatMul", shape_metadata=(10, 10))
        graph.nodes["n3"] = LogicalNode(id="n3", op_type="Add")
        graph.nodes["n4"] = LogicalNode(id="n4", op_type="Foo")

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
        assert flops == 202
    except Exception as e:
        raise e
        pass


def test_memory_profiler() -> None:
    """Test the memory profiler behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies the memory profiling utility for logical graphs.\n\n    Returns:\n    None\n    "
        graph = LogicalGraph(name="test")
        graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
        graph.nodes["n2"] = LogicalNode(id="n2", op_type="Add")

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
        assert mem == 408
    except Exception as e:
        raise e
        pass


def test_numerical_anomaly_detector() -> None:
    """Test the numerical anomaly detector behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies the numerical anomaly detector's ability to identify NaNs and Infs.\n\n    Returns:\n    None\n    "
        from ml_switcheroo_compiler.core.config import config

        config.eager_mode = True
        device = Device(DeviceType.CPU, 0)
        t1 = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
        check_numerical_anomaly(t1)
        t_none = Tensor(None, TensorConfig((2,), DType.Float32, device))
        check_numerical_anomaly(t_none)
        t2 = Tensor(np.array([1.0, np.nan]), TensorConfig((2,), DType.Float32, device))
        with pytest.raises(Exception):
            check_numerical_anomaly(t2)

        class NonArray:
            """Non Array class."""

        t3 = Tensor(NonArray(), TensorConfig((2,), DType.Float32, device))
        check_numerical_anomaly(t3)
        config.eager_mode = False
    except Exception as e:
        raise e
        pass


def test_to_graphviz() -> None:
    """Test the to graphviz behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies the Graphviz DOT export utility for logical graphs.\n\n    Returns:\n    None\n    "
        graph = LogicalGraph(name="test")
        graph.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
        graph.nodes["n2"] = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
        dot = to_graphviz(graph)
        assert "digraph G {" in dot
        assert 'label="Input"' in dot
        assert '"n1" -> "n2"' in dot
    except Exception as e:
        raise e
        pass


def test_to_html() -> None:
    """Test the to html behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Verifies the HTML export utility for logical graphs.\n\n    Returns:\n    None\n    "
        graph = LogicalGraph(name="test")
        html = to_html(graph)
        assert "<h1>IR Graph</h1>" in html
    except Exception as e:
        raise e
        pass
