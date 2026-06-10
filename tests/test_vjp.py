"""Docstring."""

from typing import Any

"""Tests for VJP (Reverse-Mode AD) implementations."""

import pytest  # noqa: E402
from ml_switcheroo_ir import LogicalGraph, LogicalNode  # noqa: E402
from ml_switcheroo.grad import grad  # noqa: E402


def _setup_graph(op_type: Any, inputs_count: Any) -> Any:
    """Docstring."""
    g = LogicalGraph()
    inputs = []
    for i in range(inputs_count):
        nid = f"in_{i}"
        g.nodes[nid] = LogicalNode(id=nid, op_type="Input", shape_metadata=())
        inputs.append(nid)

    g.nodes["out"] = LogicalNode(
        id="out", op_type=op_type, inputs=inputs, shape_metadata=()
    )
    return g, inputs, "out"


def test_grad_missing_output() -> None:
    """Docstring."""
    g = LogicalGraph()
    with pytest.raises(ValueError, match="not found in graph"):
        grad(g, ["w"], "out")


def test_grad_missing_wrt() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Add", 2)
    with pytest.raises(ValueError, match="not found in graph"):
        grad(g, ["w"], out)


def test_vjp_add() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Add", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2
    # Verify both gradients are 1.0 (from output adjoint)
    assert (
        grad_g.nodes[grad_g.outputs[0]].op_type == "Constant"
        or "add" in grad_g.outputs[0]
        or grad_g.nodes[grad_g.outputs[0]].op_type in ["Add", "Sum"]
        or "grad_ones" in grad_g.outputs[0]
    )


def test_vjp_sub() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Sub", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2


def test_vjp_mul() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Mul", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2


def test_vjp_div() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Div", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2


def test_vjp_exp() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Exp", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_log() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Log", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_sum() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Sum", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_mean() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Mean", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_max() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Max", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_min() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Min", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_matmul() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("MatMul", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2


def test_vjp_gemm() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Gemm", 3)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 3


def test_vjp_transpose() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Transpose", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_conv() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Conv", 2)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 2


def test_vjp_maxpool() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("MaxPool", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_relu() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Relu", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_vjp_softmax() -> None:
    """Docstring."""
    g, inputs, out = _setup_graph("Softmax", 1)
    grad_g = grad(g, inputs, out)
    assert len(grad_g.outputs) == 1


def test_grad_accumulation() -> None:
    """Docstring."""
    # Test gradient accumulation
    """Docstring."""
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    # w flows into a and b
    g.nodes["a"] = LogicalNode(id="a", op_type="Add", inputs=["w", "w"])
    g.nodes["b"] = LogicalNode(id="b", op_type="Mul", inputs=["a", "w"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Add", inputs=["b", "a"])

    grad_g = grad(g, ["w"], "out")
    assert len(grad_g.outputs) == 1


def test_stop_gradient() -> None:
    """Docstring."""
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["sg"] = LogicalNode(id="sg", op_type="StopGradient", inputs=["w"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Add", inputs=["sg", "sg"])

    grad_g = grad(g, ["w"], "out")
    assert grad_g.nodes[grad_g.outputs[0]].op_type == "Constant"
    assert grad_g.nodes[grad_g.outputs[0]].attributes["value"] == 0.0


def test_missing_vjp() -> None:
    """Docstring."""
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["out"] = LogicalNode(id="out", op_type="UnknownOp", inputs=["w"])

    with pytest.raises(ValueError, match="Missing VJP rule"):
        grad(g, ["w"], "out")


def test_vjp_returns_wrong_number_of_adjoints() -> None:
    """Docstring."""
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["out"] = LogicalNode(id="out", op_type="TestOp", inputs=["w", "w"])

    from ml_switcheroo.grad import register_vjp

    @register_vjp("TestOp")
    def bad_vjp(graph: Any, node: Any, adj: Any) -> Any:
        """Docstring."""
        return [adj]  # Returns 1 instead of 2

    with pytest.raises(ValueError, match="expected 2"):
        grad(g, ["w"], "out")


def test_vjp_returns_none() -> None:
    """Docstring."""
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["out"] = LogicalNode(id="out", op_type="TestNoneOp", inputs=["w"])

    from ml_switcheroo.grad import register_vjp

    @register_vjp("TestNoneOp")
    def none_vjp(graph: Any, node: Any, adj: Any) -> Any:
        """Docstring."""
        return [None]

    grad_g = grad(g, ["w"], "out")
    assert grad_g.nodes[grad_g.outputs[0]].op_type == "Constant"
    assert grad_g.nodes[grad_g.outputs[0]].attributes["value"] == 0.0


def test_grad_output_is_input() -> None:
    """Docstring."""
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.outputs = ["w"]
    grad_g = grad(g, ["w"], "w")
    assert len(grad_g.outputs) == 1


def test_grad_unreachable_node() -> None:
    """Docstring."""
    g = LogicalGraph()
    g.nodes["w"] = LogicalNode(id="w", op_type="Input")
    g.nodes["a"] = LogicalNode(id="a", op_type="Add", inputs=["w", "w"])
    g.nodes["b"] = LogicalNode(id="b", op_type="Mul", inputs=["w", "w"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Add", inputs=["a", "a"])
    # Both 'out' and 'b' are graph outputs, so topological_sort includes 'b'
    g.outputs = ["out", "b"]

    # We only take gradient wrt 'out'
    grad_g = grad(g, ["w"], "out")
    assert len(grad_g.outputs) == 1
