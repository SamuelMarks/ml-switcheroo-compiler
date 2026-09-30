"""Tests for test_autodiff_common_rules_coverage."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.transforms.autodiff_rules.common as ad_common


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
