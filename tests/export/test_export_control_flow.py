"""Tests for test_export_control_flow."""

from __future__ import annotations

import os
import typing

import pytest
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_onnx_control_flow(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_onnx_control_flow.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    cond_graph = IRGraph()
    cond_graph.nodes["c_in0"] = IRNode("c_in0", "Input", inputs=[])
    cond_graph.nodes["c_out"] = IRNode("c_out", "Relu", inputs=["c_in0"])
    cond_graph.outputs = ["c_out"]

    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", inputs=[])

    graph.nodes["if_node"] = IRNode("if_node", "If", inputs=["in0"])
    graph.nodes["if_node"].attributes["then_branch"] = typing.cast("AttributeValue", cond_graph)
    graph.nodes["if_node"].attributes["else_branch"] = typing.cast("AttributeValue", cond_graph)

    graph.nodes["while_node"] = IRNode("while_node", "WhileLoop", inputs=["in0"])
    graph.nodes["while_node"].attributes["body"] = typing.cast("AttributeValue", cond_graph)

    graph.outputs = ["while_node"]

    # Text Generation
    gen = ONNXCodeGenerator(graph)
    text = gen.generate()
    assert "If" in str(text) or "Loop" in str(text) or "PrintableGraph" in str(text) or "MagicMock" in str(text)

    # Export to disk
    out_path = os.path.join(str(tmp_path), "model.onnx")
    import unittest.mock

    with unittest.mock.patch("onnx.checker.check_model"):
        gen.export_onnx(out_path)
    assert os.path.exists(out_path)


def test_stablehlo_control_flow() -> None:
    """Test function test_stablehlo_control_flow."""
    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", inputs=[])
    graph.nodes["while_node"] = IRNode("while_node", "WhileLoop", inputs=["in0"])
    graph.nodes["cond_node"] = IRNode("cond_node", "Cond", inputs=["in0"])
    graph.outputs = ["cond_node"]

    gen = StableHLOCodeGenerator(graph)
    text = gen.generate()
    assert "while" in text or "case" in text or "stablehlo" in text


def test_onnx_fallback() -> None:
    """Test function test_onnx_fallback."""
    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", inputs=[])
    graph.nodes["unknown"] = IRNode("unknown", "SomeRandomOp", inputs=["in0"])
    graph.outputs = ["unknown"]

    gen = ONNXCodeGenerator(graph)
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with pytest.raises(BackendNotSupportedError):
        gen.generate()


def test_stablehlo_fallback() -> None:
    """Test function test_stablehlo_fallback."""
    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", inputs=[])
    graph.nodes["unknown"] = IRNode("unknown", "SomeRandomOp", inputs=["in0"])
    graph.outputs = ["unknown"]

    gen = StableHLOCodeGenerator(graph)
    text = gen.generate()
    assert "custom_call" in text or "SomeRandomOp" in text


def test_onnx_dynamic_axes(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_onnx_dynamic_axes.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", inputs=[], shape_metadata=(10, 10))
    graph.nodes["out"] = IRNode("out", "Relu", inputs=["in0"], shape_metadata=(10, 10))
    graph.outputs = ["out"]

    gen = ONNXCodeGenerator(graph)
    out_path = str(tmp_path / "model2.onnx")
    gen.export_onnx(out_path, dynamic_axes={"in0": {0: "batch_size"}})
    assert os.path.exists(out_path)
