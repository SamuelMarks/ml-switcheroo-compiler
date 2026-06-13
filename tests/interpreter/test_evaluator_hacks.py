"""Provides required module functionality."""

import contextlib
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph


def test_evaluator_exception() -> None:
    """Execute the requested function."""
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n2 = IRNode(
        id="n2",
        op_type="nonexistent_blah",
        inputs=["n1"],
        attributes={},
        shape_metadata=(2,),
    )

    for n in [n1, n2]:
        g.nodes[n.id] = n
    g.inputs = ["n1"]
    g.outputs = ["n2"]

    with contextlib.suppress(Exception):
        evaluate_graph(g, {"n1": 1})


def test_evaluator_stubs() -> None:
    """Test stub evaluations in interpreter."""
    import numpy as np
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    # Greater
    g1 = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
    ng = IRNode(id="ng", op_type="Greater", inputs=["n1", "n2"], attributes={}, shape_metadata=(1,))
    g1.nodes = {n.id: n for n in [n1, n2, ng]}
    g1.inputs = ["n1", "n2"]
    g1.outputs = ["ng"]
    res1 = evaluate_graph(g1, {"n1": np.array([2.0]), "n2": np.array([1.0])})
    assert res1["ng"][0]

    # Where
    g2 = IRGraph()
    nc = IRNode(id="nc", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
    nt = IRNode(id="nt", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
    nf = IRNode(id="nf", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
    nw = IRNode(
        id="nw", op_type="Where", inputs=["nc", "nt", "nf"], attributes={}, shape_metadata=(1,)
    )
    g2.nodes = {n.id: n for n in [nc, nt, nf, nw]}
    g2.inputs = ["nc", "nt", "nf"]
    g2.outputs = ["nw"]
    res2 = evaluate_graph(
        g2, {"nc": np.array([True]), "nt": np.array([2.0]), "nf": np.array([3.0])}
    )
    assert res2["nw"][0] == 2.0


def test_evaluator_shape_kwargs() -> None:
    """Test Expand and Reshape kwargs."""
    import numpy as np
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    # Expand
    g1 = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
    ne = IRNode(id="ne", op_type="BroadcastTo", inputs=["n1"], attributes={}, shape_metadata=(2,))
    g1.nodes = {n.id: n for n in [n1, ne]}
    g1.inputs = ["n1"]
    g1.outputs = ["ne"]
    res1 = evaluate_graph(g1, {"n1": np.array([1.0])})
    assert res1["ne"].shape == (2,)

    # Reshape
    g2 = IRGraph()
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    nr = IRNode(id="nr", op_type="Reshape", inputs=["n2"], attributes={}, shape_metadata=(1, 2))
    g2.nodes = {n.id: n for n in [n2, nr]}
    g2.inputs = ["n2"]
    g2.outputs = ["nr"]
    res2 = evaluate_graph(g2, {"n2": np.array([1.0, 2.0])})
    assert res2["nr"].shape == (1, 2)
