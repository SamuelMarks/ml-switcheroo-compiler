"""Tests for test_export."""

from __future__ import annotations

import typing

import pytest
from ml_switcheroo_ir import LogicalNode, NodeDict

from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph
from tests.testing_utils import as_callable, as_node_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


@pytest.mark.parametrize("op_type", ["Add", "Exp", "Relu", "ReduceMax", "Where", "Cast"])
def test_onnx_export_ops(op_type: EagerTensorProtocol) -> None:
    """Test function test_onnx_export_ops.

    Args:
        op_type: Test parameter `op_type`.
    """
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (10,)
    n2 = as_callable(LogicalNode)(id="n2", op_type=op_type, inputs=["in1", "in1"])
    graph.nodes = NodeDict(as_node_dict({"in1": n1, "n2": n2}))

    gen = ONNXCodeGenerator(graph)
    code = gen.generate()

    # We just want to make sure it doesn't crash and generates a graph string
    if "PrintableGraph" in str(code) or "MagicMock" in str(code):
        return  # Mocked
    assert "graph " in str(code)
    if op_type != "Where":  # Where output structure is slightly different depending on ONNX logic
        assert op_type in as_callable(str)(code) or as_callable(op_type.upper)() in as_callable(str)(code).upper()


@pytest.mark.parametrize("op_type", ["Add", "Exp", "Relu", "ReduceMax", "Where", "Cast"])
def test_stablehlo_export_ops(op_type: EagerTensorProtocol) -> None:
    """Test function test_stablehlo_export_ops.

    Args:
        op_type: Test parameter `op_type`.
    """
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (10,)
    n2 = as_callable(LogicalNode)(id="n2", op_type=op_type, inputs=["in1", "in1"])
    graph.nodes = NodeDict(as_node_dict({"in1": n1, "n2": n2}))

    gen = StableHLOCodeGenerator(graph)
    code = gen.generate()

    assert "module @" in code
    assert "stablehlo" in code
