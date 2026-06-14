"""Provides required module functionality."""

import pytest
from ml_switcheroo_compiler.transforms.autodiff import grad
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import (
    register_vjp,
    _VJP_REGISTRY,
)


def test_autodiff_coverage_brute() -> None:
    """Execute the requested function."""
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    n3 = IRNode(id="n3", op_type="FakeOp", inputs=["n1", "n2"], attributes={}, shape_metadata=None)

    g.nodes = {"n1": n1, "n2": n2, "n3": n3}

    if "FakeOp" in _VJP_REGISTRY:
        del _VJP_REGISTRY["FakeOp"]

    @register_vjp("FakeOp")
    def fake_op_vjp(graph: object, node: object, adj_id: str) -> list[str]:
        """Docstring."""
        raise NotImplementedError("Missing VJP")

    with pytest.raises(ValueError, match="Missing VJP rule for operation"):
        grad(g, ["n1"], "n3")

    if "FakeOp" in _VJP_REGISTRY:
        del _VJP_REGISTRY["FakeOp"]

    @register_vjp("FakeOp")
    def fake_op_vjp2(graph: object, node: object, adj_id: str) -> list[str]:
        """Docstring."""
        return ["adj_1"]  # Return 1 instead of 2 expected

    with pytest.raises(ValueError, match="VJP for FakeOp returned 1 adjoints, expected 2."):
        grad(g, ["n1"], "n3")

    if "FakeOp" in _VJP_REGISTRY:
        del _VJP_REGISTRY["FakeOp"]
