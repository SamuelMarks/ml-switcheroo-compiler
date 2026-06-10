"""Tests for optimization scenarios."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo.optimization import dce, cse, constant_folding
import numpy as np


def test_dce() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1", "n1"])
    g.nodes["n3"] = LogicalNode(
        id="n3", op_type="Mul", inputs=["n2", "n1"]
    )  # Dead node

    opt_g = dce(g)
    assert "n3" not in opt_g.nodes
    assert "n1" in opt_g.nodes
    assert "n2" in opt_g.nodes
    assert opt_g.outputs == ["n2"]


def test_cse() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n4", "n5"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1", "n1"])
    g.nodes["n3"] = LogicalNode(
        id="n3", op_type="Add", inputs=["n1", "n1"]
    )  # Identical to n2
    g.nodes["n4"] = LogicalNode(id="n4", op_type="Mul", inputs=["n2", "n1"])
    g.nodes["n5"] = LogicalNode(
        id="n5", op_type="Mul", inputs=["n3", "n1"]
    )  # Should become identical to n4 after mapping

    opt_g = cse(g)
    assert "n3" not in opt_g.nodes
    assert "n5" not in opt_g.nodes

    # Outputs should be remapped
    assert opt_g.outputs == ["n4", "n4"]
    assert opt_g.nodes["n4"].inputs == ["n2", "n1"]


def test_constant_folding() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n5"])
    g.nodes["c1"] = LogicalNode(id="c1", op_type="Constant", attributes={"value": 2})
    g.nodes["c2"] = LogicalNode(id="c2", op_type="Constant", attributes={"value": 3})
    g.nodes["n1"] = LogicalNode(
        id="n1", op_type="Add", inputs=["c1", "c2"]
    )  # Folds to 5
    g.nodes["n2"] = LogicalNode(
        id="n2", op_type="Sub", inputs=["n1", "c1"]
    )  # Folds to 3
    g.nodes["n3"] = LogicalNode(
        id="n3", op_type="Mul", inputs=["n2", "c2"]
    )  # Folds to 9
    g.nodes["n4"] = LogicalNode(
        id="n4", op_type="Div", inputs=["n3", "c2"]
    )  # Folds to 3.0
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["n5"] = LogicalNode(
        id="n5", op_type="Add", inputs=["n4", "in"]
    )  # Does not fold

    opt_g = constant_folding(g)
    assert opt_g.nodes["n1"].op_type == "Constant"
    assert opt_g.nodes["n1"].attributes["value"] == 5
    assert opt_g.nodes["n2"].op_type == "Constant"
    assert opt_g.nodes["n2"].attributes["value"] == 3
    assert opt_g.nodes["n3"].op_type == "Constant"
    assert opt_g.nodes["n3"].attributes["value"] == 9
    assert opt_g.nodes["n4"].op_type == "Constant"
    assert opt_g.nodes["n4"].attributes["value"] == 3.0
    assert opt_g.nodes["n5"].op_type == "Add"


def test_constant_folding_unsupported_exception() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(id="c1", op_type="Constant", attributes={"value": 2})
    g.nodes["n1"] = LogicalNode(
        id="n1", op_type="NonExistentOp", inputs=["c1"]
    )  # Error thrown by evaluator

    opt_g = constant_folding(g)
    assert opt_g.nodes["n1"].op_type == "NonExistentOp"  # Unfolded due to error


def test_constant_folding_not_scalar() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array([2, 3])}
    )
    g.nodes["c2"] = LogicalNode(
        id="c2", op_type="Constant", attributes={"value": np.array([1, 1])}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])

    opt_g = constant_folding(g)
    assert opt_g.nodes["n1"].op_type == "Constant"
    assert isinstance(opt_g.nodes["n1"].attributes["value"], np.ndarray)


def test_constant_folding_numpy_scalar() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    # 0d array
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array(2)}
    )
    g.nodes["c2"] = LogicalNode(
        id="c2", op_type="Constant", attributes={"value": np.array(3)}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])

    opt_g = constant_folding(g)
    assert opt_g.nodes["n1"].op_type == "Constant"
    # Should be unwrapped using .item()
    assert not isinstance(opt_g.nodes["n1"].attributes["value"], np.ndarray)


def test_constant_folding_numpy_single_element_array() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    # 1d array with size 1
    g.nodes["c1"] = LogicalNode(
        id="c1", op_type="Constant", attributes={"value": np.array([2])}
    )
    g.nodes["c2"] = LogicalNode(
        id="c2", op_type="Constant", attributes={"value": np.array([3])}
    )
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])

    opt_g = constant_folding(g)
    assert opt_g.nodes["n1"].op_type == "Constant"
    # Should be unwrapped using .item()
    assert not isinstance(opt_g.nodes["n1"].attributes["value"], np.ndarray)
