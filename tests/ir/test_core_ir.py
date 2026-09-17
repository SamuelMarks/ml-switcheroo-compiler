"""Tests for core intermediate representation (IR) schemas and utilities."""

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.types import DType, TensorSpec

from ml_switcheroo_compiler.ir.core import IRBlock, IRGraph, IRNode, NoTangent, ZeroTangent, clone_logical_node


def test_clone_logical_node() -> None:
    """Verify clone_logical_node preserves and overrides fields without mutating original."""
    subgraph = LogicalGraph(name="sub1")
    spec = TensorSpec(shape=(2, 3), dtype=DType.float32)
    node = LogicalNode(
        id="id1",
        op_type="op1",
        domain="ai.onnx",
        version=1,
        attributes={"attr1": "val1"},
        inputs=["in1"],
        outputs=["out1"],
        dtype=DType.float32,
        output_specs=[spec],
        subgraphs={"body": subgraph},
        device="cuda:0",
        stream="stream_0",
    )
    node2 = clone_logical_node(node, op_type="op2", device="cpu")
    assert node2.op_type == "op2"
    assert node2.id == "id1"
    assert node2.device == "cpu"
    assert node2.stream == "stream_0"
    assert node2.dtype == DType.float32
    assert len(node2.output_specs) == 1
    assert "body" in node2.subgraphs

    # Verify deep-copy independence
    node2.attributes["attr1"] = "mutated"
    assert node.attributes["attr1"] == "val1"

    # Test clone with minimal node (covering None defaults)
    node_min = LogicalNode(id="min1", op_type="min_op")
    node_min.outputs = None  # type: ignore[assignment]
    node_min.output_specs = None  # type: ignore[assignment]
    node_min.subgraphs = None  # type: ignore[assignment]
    cloned_min = clone_logical_node(node_min)
    assert cloned_min.id == "min1"


def test_irnode() -> None:
    """Verify IRNode alias and dynamic/static shape inference properties."""
    assert IRNode is LogicalNode
    assert IRGraph is LogicalGraph

    n = IRNode("id1", "op", shape_metadata=(1, 2))
    assert n.is_dynamic_shape is False
    assert n.static_shape == (1, 2)
    assert n.rank == 2

    n_dyn = IRNode("id2", "op", shape_metadata=(1, "x"))
    assert n_dyn.is_dynamic_shape is True
    with pytest.raises(ValueError, match="contains dynamic or symbolic dimensions"):
        _ = n_dyn.static_shape
    assert n_dyn.rank == 2

    n_none = IRNode("id3", "op", shape_metadata=None)
    assert n_none.is_dynamic_shape is False
    with pytest.raises(ValueError, match="has no shape metadata"):
        _ = n_none.static_shape
    assert n_none.rank == 0


def test_tensorspec() -> None:
    """Verify canonical TensorSpec properties, static shape coercion, and dynamic checks."""
    s = TensorSpec((1, 2), DType.float32)
    assert s.is_dynamic is False
    assert s.static_shape == (1, 2)
    assert s.rank == 2

    s_dyn = TensorSpec((1, "x"), DType.float32)
    assert s_dyn.is_dynamic is True
    with pytest.raises(ValueError, match="has dynamic dimensions"):
        _ = s_dyn.static_shape


def test_irblock() -> None:
    """Verify legacy IRBlock container properties."""
    with pytest.deprecated_call():
        b = IRBlock("b1")
    assert b.id == "b1"
    assert isinstance(b, LogicalGraph)
    assert b.nodes == {}
    assert b.inputs == []
    assert b.outputs == []

    node = LogicalNode(id="n1", op_type="Add")
    with pytest.deprecated_call():
        b2 = IRBlock("b2", nodes=[node], inputs=["in1"], outputs=["out1"])
    assert b2.nodes["n1"] == node
    assert b2.inputs == ["in1"]
    assert b2.outputs == ["out1"]


def test_tangent_nodes() -> None:
    """Verify ZeroTangent and NoTangent initialization and LogicalNode inheritance."""
    zt = ZeroTangent("t1", shape_metadata=(2, 2))
    assert zt.op_type == "ZeroTangent"
    assert isinstance(zt, LogicalNode)
    assert zt.static_shape == (2, 2)

    nt = NoTangent("t2")
    assert nt.op_type == "NoTangent"
    assert isinstance(nt, LogicalNode)
