"""Behavioral unit tests for autodiff common."""

from __future__ import annotations

from ml_switcheroo_ir import IRGraph, IRNode, LogicalGraph, LogicalNode

import ml_switcheroo_compiler.transforms.autodiff_rules.common as ad_common
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients, make_zero_jvp, make_zero_vjp


def test_make_zero_vjp() -> None:
    """Test function test_make_zero_vjp."""
    vjp_fn = make_zero_vjp("TestOp")
    graph = IRGraph()
    node = IRNode(id="test_node", op_type="TestOp", inputs=["a", "b"])
    res = vjp_fn(graph, node, "cot")
    assert len(res) == 2
    assert res[0] == UnconnectedGradients.ZERO
    assert res[1] == UnconnectedGradients.ZERO


def test_make_zero_jvp() -> None:
    """Test function test_make_zero_jvp."""
    jvp_fn = make_zero_jvp("TestOp")
    graph = IRGraph()
    node = IRNode(id="test_node", op_type="TestOp", inputs=["a", "b"])
    res = jvp_fn(graph, node, ("t_a", "t_b"))
    assert res == ""


class _DummyBadShape:
    """Shape object whose iterator raises TypeError."""

    def __iter__(self) -> _DummyBadShape:
        """Return self as iterator.

        Returns:
            _DummyBadShape: Self.
        """
        return self

    def __next__(self) -> int:
        """Raise TypeError on iteration.

        Raises:
            TypeError: Simulated invalid dimension.

        Returns:
            int: Computed output value from `__next__`.
        """
        raise TypeError("Not iterable dimension")


def test_autodiff_rules_common_exhaustive() -> None:
    """Verify make_zero_vjp and make_zero_jvp functions."""
    assert ad_common.UnconnectedGradients.NONE.value == "none"
    assert ad_common.UnconnectedGradients.ZERO.value == "zero"

    graph = LogicalGraph(name="test_ad", outputs=["out"])
    node = LogicalNode("n0", "CustomZeroOp", inputs=["in_0", "in_1"])

    vjp_fn = ad_common.make_zero_vjp("CustomZeroOp")
    vjp_res = vjp_fn(graph, node, "cotangent_id")
    assert vjp_res == (ad_common.UnconnectedGradients.ZERO, ad_common.UnconnectedGradients.ZERO)

    node_empty = LogicalNode("n_empty", "CustomZeroOp", inputs=[])
    vjp_res_empty = vjp_fn(graph, node_empty, "cotangent_id")
    assert vjp_res_empty == ()

    jvp_fn = ad_common.make_zero_jvp("CustomZeroOp")
    jvp_res = jvp_fn(graph, node, ("tan_0", "tan_1"))
    assert jvp_res == ""
