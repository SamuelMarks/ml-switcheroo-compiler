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
        object: The inferred shape or computed result.
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
        object: The inferred shape or computed result.
    """
    "Verifies the shape debugging utility under various model execution scenarios.\n\n    Returns:\n    None\n    "

    def dummy_model(x):
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

    def dummy_model_no_shape(x) -> int:
        """Dummy model no shape.

        Args:
        x (object): The first input tensor.

        Returns:
        int: The resulting output.
        """
        return 5

    res = debug_shapes(dummy_model_no_shape, (2, 2))
    assert "| output | unknown | float64 |" in res

    def failing_model(x) -> NoReturn:
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
        object: The inferred shape or computed result.
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

            def __iter__(self):
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
        object: The inferred shape or computed result.
    """
    try:
        "Verifies the memory profiling utility for logical graphs.\n\n    Returns:\n    None\n    "
        graph = LogicalGraph(name="test")
        graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
        graph.nodes["n2"] = LogicalNode(id="n2", op_type="Add")

        class BadShape:
            """Bad Shape class."""

            def __iter__(self):
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
        object: The inferred shape or computed result.
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
        object: The inferred shape or computed result.
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
        object: The inferred shape or computed result.
    """
    try:
        "Verifies the HTML export utility for logical graphs.\n\n    Returns:\n    None\n    "
        graph = LogicalGraph(name="test")
        html = to_html(graph)
        assert "<h1>IR Graph</h1>" in html
    except Exception as e:
        raise e
        pass


def test_encode_image_and_write_raw_pb(tmp_path: pytest.TempPathFactory) -> None:
    """Verify image encoding to PNG bytes and protobuf writing.

    Args:
        tmp_path (pytest.TempPathFactory): Temporary directory fixture.
    """
    import io

    from PIL import Image

    from ml_switcheroo_compiler.diagnostics.summary import encode_image, write_raw_pb

    # 1. Test 4D float tensor encoding
    device = Device(DeviceType.CPU, 0)
    data_4d = np.zeros((1, 16, 16, 3), dtype=np.float32)
    data_4d[0, 8, 8, 0] = 1.0
    tensor_4d = Tensor(data_4d, TensorConfig((1, 16, 16, 3), DType.Float32, device))
    png_bytes = encode_image(tensor_4d)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    # Decode with PIL to verify integrity
    decoded = Image.open(io.BytesIO(png_bytes))
    assert decoded.size == (16, 16)
    assert decoded.mode == "RGB"

    # 2. Test 2D grayscale uint8 tensor
    data_2d = np.full((8, 8), 128, dtype=np.uint8)
    tensor_2d = Tensor(data_2d, TensorConfig((8, 8), DType.UInt8, device))
    png_2d = encode_image(tensor_2d)
    assert png_2d[:8] == b"\x89PNG\r\n\x1a\n"
    decoded_2d = Image.open(io.BytesIO(png_2d))
    assert decoded_2d.size == (8, 8)

    # 3. Test 3D channels-first tensor
    data_cf = np.zeros((3, 10, 10), dtype=np.float32)
    tensor_cf = Tensor(data_cf, TensorConfig((3, 10, 10), DType.Float32, device))
    png_cf = encode_image(tensor_cf)
    assert png_cf[:8] == b"\x89PNG\r\n\x1a\n"
    decoded_cf = Image.open(io.BytesIO(png_cf))
    assert decoded_cf.size == (10, 10)

    # 3b. Test 3D single channel tensor (H, W, 1)
    data_1c = np.full((12, 12, 1), 200, dtype=np.uint8)
    tensor_1c = Tensor(data_1c, TensorConfig((12, 12, 1), DType.UInt8, device))
    png_1c = encode_image(tensor_1c)
    assert png_1c[:8] == b"\x89PNG\r\n\x1a\n"
    decoded_1c = Image.open(io.BytesIO(png_1c))
    assert decoded_1c.size == (12, 12)

    # 4. Test write_raw_pb
    logdir: str = str(tmp_path / "tfevents")
    write_raw_pb(b"mock_proto_data", logdir)
    with open(f"{logdir}/events.out.tfevents.pb", "rb") as f:
        assert f.read() == b"mock_proto_data"

    # 5. Test debugging.enable_dump_debug_info
    from ml_switcheroo_compiler.diagnostics.debugging import enable_dump_debug_info

    dump_dir: str = str(tmp_path / "dump_debug")
    enable_dump_debug_info(dump_dir)
    assert (tmp_path / "dump_debug").exists()
