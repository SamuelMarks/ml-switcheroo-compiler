# ruff: noqa: E501
"""Unit tests for the diagnostics module of the ml_switcheroo_compiler framework.

This module contains comprehensive unit tests verifying the correctness of traceback
reconstruction, shape debugging, FLOPs estimation, memory profiling, numerical anomaly
detection, and graph visualization utilities (Graphviz and HTML exports).
"""

from __future__ import annotations

import typing

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
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_traceback_reconstructor() -> None:
    """Verify that TracebackReconstructor formats exception tracebacks."""
    exc = ValueError("test error")
    formatted = format_traceback(exc)
    assert "TracebackReconstructor: test error" in formatted


def test_debug_shapes() -> None:
    """Verify the shape debugging utility under various model execution scenarios."""

    def dummy_model(x: EagerValue) -> EagerValue:
        """Dummy model.

        Args:
            x (EagerValue): The first input tensor.

        Returns:
            EagerValue: The resulting output.
        """
        return ensure_eager_tensor(x) + ensure_eager_tensor(1.0)

    res = ensure_callable(debug_shapes)(dummy_model, (2, 2))
    assert "| input | (2, 2) | float64 |" in res
    assert "| output | (2, 2) | float64 |" in res

    def dummy_model_no_shape(x: EagerValue) -> int:
        """Dummy model no shape.

        Args:
            x (EagerValue): The first input tensor.

        Returns:
            int: The resulting output.
        """
        return 5

    res = ensure_callable(debug_shapes)(dummy_model_no_shape, (2, 2))
    assert "| output | unknown | float64 |" in res

    def failing_model(x: EagerValue) -> typing.NoReturn:
        """Failing model.

        Args:
            x (EagerValue): The first input tensor.

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
        res_custom = ensure_callable(debug_shapes)(dummy_model, (2, 2))
        assert "header" in str(res_custom)

        m.setattr(debugger, "_FORMATTERS", {"graphviz": {"header": "H", "node": "N", "edge": "E", "footer": "F"}})
        g = LogicalGraph()
        assert "H" in debugger.to_graphviz(g)

        m.setattr(debugger, "_FORMATTERS", {"html": {"template": "HTML"}})
        assert "HTML" in debugger.to_html(g)


def test_estimate_flops() -> None:
    """Verify the FLOPs estimation utility for logical graphs."""
    from ml_switcheroo_compiler.diagnostics.flop_counter import _estimate_node_flops

    graph = LogicalGraph(name="test")
    graph.nodes["in_a"] = LogicalNode(id="in_a", op_type="Input", shape_metadata=(4, 8))
    graph.nodes["in_b"] = LogicalNode(id="in_b", op_type="Input", shape_metadata=(8, 5))
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="MatMul", inputs=["in_a", "in_b"], shape_metadata=(4, 5))
    graph.nodes["n2_no_out_shape"] = LogicalNode(id="n2_no_out_shape", op_type="MatMul", inputs=["in_a", "in_b"])
    graph.nodes["in_3d_a"] = LogicalNode(id="in_3d_a", op_type="Input", shape_metadata=(2, 4, 8))
    graph.nodes["in_3d_b"] = LogicalNode(id="in_3d_b", op_type="Input", shape_metadata=(2, 8, 5))
    graph.nodes["n2_batched_no_out"] = LogicalNode(id="n2_batched_no_out", op_type="MatMul", inputs=["in_3d_a", "in_3d_b"])
    graph.nodes["n2_fallback_2d"] = LogicalNode(id="n2_fallback_2d", op_type="MatMul", shape_metadata=(10, 10))
    graph.nodes["n2_fallback_1d"] = LogicalNode(id="n2_fallback_1d", op_type="MatMul", shape_metadata=(10,))
    graph.nodes["n2_no_shape"] = LogicalNode(id="n2_no_shape", op_type="MatMul")
    graph.nodes["in_1d"] = LogicalNode(id="in_1d", op_type="Input", shape_metadata=(8,))
    graph.nodes["n2_1d_in"] = LogicalNode(id="n2_1d_in", op_type="MatMul", inputs=["in_1d", "in_1d"], shape_metadata=(1,))
    graph.nodes["conv_in"] = LogicalNode(id="conv_in", op_type="Input", shape_metadata=(1, 3, 8, 8))
    graph.nodes["conv_w_oihw"] = LogicalNode(id="conv_w_oihw", op_type="Input", shape_metadata=(16, 3, 3, 3))
    graph.nodes["conv_w_hwio"] = LogicalNode(id="conv_w_hwio", op_type="Input", shape_metadata=(3, 3, 3, 16))
    graph.nodes["conv1"] = LogicalNode(id="conv1", op_type="Conv2D", inputs=["conv_in", "conv_w_oihw"], shape_metadata=(1, 16, 6, 6))
    graph.nodes["conv2"] = LogicalNode(id="conv2", op_type="Conv2D", inputs=["conv_in", "conv_w_hwio"], shape_metadata=(1, 6, 6, 8))
    graph.nodes["conv_bad_w"] = LogicalNode(id="conv_bad_w", op_type="Conv2D", inputs=["conv_in", "in_1d"], shape_metadata=(1, 6, 6, 16))
    graph.nodes["conv_fallback"] = LogicalNode(id="conv_fallback", op_type="Conv2D", shape_metadata=(1, 6, 6, 16))
    graph.nodes["conv_no_shape"] = LogicalNode(id="conv_no_shape", op_type="Conv2D")
    graph.nodes["n3"] = LogicalNode(id="n3", op_type="Add")

    class BadShape:
        """Bad Shape class."""

        def __iter__(self) -> None:
            """Raise TypeError on iteration."""
            msg = "bad iterator"
            raise TypeError(msg)

    n5 = LogicalNode(id="n5", op_type="Add")
    ensure_mutable(n5).shape_metadata = ensure_eager_tensor(BadShape())
    graph.nodes["n5"] = n5
    flops = estimate_flops(graph)
    assert flops > 2000
    assert _estimate_node_flops(LogicalNode(id="x", op_type="Input")) == 0


def test_memory_profiler() -> None:
    """Verify the memory profiling utility for logical graphs."""
    from ml_switcheroo_ir import DType as IRDType

    graph = LogicalGraph(name="test")
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10), dtype=IRDType.float32)
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="Add", attributes={"dtype": "DType.float64"})
    graph.nodes["n_f16"] = LogicalNode(id="n_f16", op_type="Add", shape_metadata=(10,), dtype=IRDType.float16)

    class BadShape:
        """Bad Shape class."""

        def __iter__(self) -> None:
            """Raise TypeError on iteration."""
            msg = "bad iterator"
            raise TypeError(msg)

    n3 = LogicalNode(id="n3", op_type="Add")
    ensure_mutable(n3).shape_metadata = ensure_eager_tensor(BadShape())
    graph.nodes["n3"] = n3
    mem = memory_profiler(graph)
    assert mem == 400 + 8 + 20 + 4


def test_numerical_anomaly_detector() -> None:
    """Verify the numerical anomaly detector's ability to identify NaNs and Infs."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    device = Device(DeviceType.CPU, 0)
    t1 = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
    check_numerical_anomaly(t1)
    t_none = Tensor(None, TensorConfig((2,), DType.Float32, device))
    check_numerical_anomaly(t_none)
    t2 = Tensor(np.array([1.0, np.nan]), TensorConfig((2,), DType.Float32, device))
    with pytest.raises(ValueError, match="Tensor contains NaN or Inf"):
        check_numerical_anomaly(t2)

    class NonArray:
        """Non Array class."""

    t3 = ensure_callable(Tensor)(ensure_callable(NonArray)(), ensure_callable(TensorConfig)((2,), DType.Float32, device))
    ensure_callable(check_numerical_anomaly)(t3)
    config.eager_mode = False


def test_to_graphviz() -> None:
    """Verify the Graphviz DOT export utility for logical graphs."""
    graph = LogicalGraph(name="test")
    graph.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    graph.nodes["n2"] = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
    dot = to_graphviz(graph)
    assert "digraph G {" in dot
    assert 'label="Input"' in dot
    assert '"n1" -> "n2"' in dot


def test_to_html() -> None:
    """Verify the HTML export utility for logical graphs."""
    graph = LogicalGraph(name="test")
    html = to_html(graph)
    assert "<h1>IR Graph</h1>" in html


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
    logdir: str = str(ensure_eager_tensor(tmp_path) / ensure_eager_tensor("tfevents"))
    write_raw_pb(b"mock_proto_data", logdir)
    with open(f"{logdir}/events.out.tfevents.pb", "rb") as f:
        assert f.read() == b"mock_proto_data"

    # 5. Test debugging.enable_dump_debug_info
    from ml_switcheroo_compiler.diagnostics.debugging import enable_dump_debug_info, get_debug_dump_config

    dump_dir: str = str(ensure_eager_tensor(tmp_path) / ensure_eager_tensor("dump_debug"))
    enable_dump_debug_info(dump_dir, tensor_debug_mode="FULL_TENSOR", circular_buffer_size=128)
    assert ensure_callable((ensure_eager_tensor(tmp_path) / ensure_eager_tensor("dump_debug")).exists)()
    dbg_cfg = get_debug_dump_config()
    assert dbg_cfg["enabled"] is True
    assert dbg_cfg["dump_root"] == dump_dir
    assert dbg_cfg["tensor_debug_mode"] == "FULL_TENSOR"
    assert dbg_cfg["circular_buffer_size"] == 128
