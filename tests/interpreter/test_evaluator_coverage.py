"""Docstring."""

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo.interpreter.evaluator import evaluate_graph


def test_evaluator_not_implemented() -> None:
    """Docstring."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="NonExistentOp", inputs=[])
    with pytest.raises(NotImplementedError, match="not implemented"):
        evaluate_graph(g, {})
