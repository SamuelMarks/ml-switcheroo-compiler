"""Tests for Constant Folding."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo.transforms.passes.constant_folding import constant_folding_pass
import numpy as np


def test_constant_folding() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array([2.0])}
    )
    g.nodes["c2"] = LogicalNode(
        id="c2", op_type="Constant", attributes={"value": np.array([3.0])}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])
    constant_folding_pass(g)
    assert g.nodes["n1"].op_type == "Constant"
    val = g.nodes["n1"].attributes["value"]
    np.testing.assert_allclose(val, np.array([5.0]))


def test_constant_folding_unsupported_op() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array([2.0])}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="UnknownOp", inputs=["c1"])
    constant_folding_pass(g)
    assert g.nodes["n1"].op_type == "UnknownOp"


def test_constant_folding_scalar_unwrap() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array([2])}
    )
    g.nodes["c2"] = LogicalNode(
        id="c2", op_type="Constant", attributes={"value": np.array([3])}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])
    constant_folding_pass(g)
    assert g.nodes["n1"].op_type == "Constant"
    assert not isinstance(g.nodes["n1"].attributes["value"], np.ndarray)
