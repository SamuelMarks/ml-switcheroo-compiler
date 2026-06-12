"""Unit tests for the graph evaluator interpreter, verifying correct execution of.

supported

operators and error handling for unsupported ones.
"""

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo.interpreter.evaluator import evaluate_graph


def test_evaluator_not_implemented() -> None:
    """Verifies that the evaluator raises a NotImplementedError when encountering a non-.

    existent operator type

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="NonExistentOp", inputs=[])
    with pytest.raises(NotImplementedError, match="not implemented"):
        evaluate_graph(g, {})


def test_evaluator_greater() -> None:
    """Verifies that the evaluator correctly evaluates the 'Greater' comparison operator.

    using NumPy arrays

    Returns:
    None
    """
    import numpy as np
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo.interpreter.evaluator import evaluate_graph

    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input")
    g.nodes["b"] = LogicalNode(id="b", op_type="Input")
    g.nodes["c"] = LogicalNode(id="c", op_type="Greater", inputs=["a", "b"])
    g.outputs = ["c"]

    res = evaluate_graph(g, inputs={"a": np.array([2.0]), "b": np.array([1.0])})
    assert res["c"][0]


def test_evaluator_unimplemented() -> None:
    """Verifies that the evaluator raises a NotImplementedError when evaluating a graph.

    containing an unknown operator type

    Returns:
    None
    """
    import pytest
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo.interpreter.evaluator import evaluate_graph

    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input")
    g.nodes["b"] = LogicalNode(id="b", op_type="UnknownOp", inputs=["a"])
    g.outputs = ["b"]

    with pytest.raises(NotImplementedError):
        evaluate_graph(g, inputs={"a": 1})
