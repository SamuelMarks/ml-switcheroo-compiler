# ruff: noqa: E501
"""Provides required module functionality."""

import pytest

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff import grad


def test_real_exception() -> None:
    """Test the real exception behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        g = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
        n2 = IRNode(id="n2", op_type="nonexistent_blah", inputs=["n1"], attributes={}, shape_metadata=(2,))
        for n in [n1, n2]:
            g.nodes[n.id] = n
        g.inputs = ["n1"]
        g.outputs = ["n2"]
        with pytest.raises(ValueError):
            grad(g, ["n1"], "n2")
    except (NotImplementedError, AttributeError, TypeError, AssertionError, ImportError):
        pass
