"""Provides required module functionality."""

from ml_switcheroo_compiler.transforms.passes.lift_state import (
    flatten_state_dict,
    unflatten_state_dict,
    lift_state_pass,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_lift_state_coverage_brute() -> None:
    """Execute the requested function."""
    state_dict = {"a": {"b": 1}, "c": 2}
    flat = flatten_state_dict(state_dict)
    assert flat == {"a.b": 1, "c": 2}

    nested = unflatten_state_dict(flat)
    assert nested == state_dict

    assert flatten_state_dict({}) == {}
    assert flatten_state_dict({"a": 1}, prefix="p") == {"p.a": 1}

    # Let's add multiple layers to unflatten.
    flat2 = {"a.b.x": 1, "a.b.y": 2, "a.c": 3}
    nested = unflatten_state_dict(flat2)
    assert nested == {"a": {"b": {"x": 1, "y": 2}, "c": 3}}

    g = IRGraph()
    n2 = IRNode(
        id="n2", op_type="AssignVariable", inputs=["n1"], attributes={}, shape_metadata=None
    )
    g.nodes = {"n2": n2}
    g.outputs = ["n2", "n3"]

    lift_state_pass(g)
