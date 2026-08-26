# ruff: noqa: E501
import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

"Unit tests for the graph evaluator interpreter, verifying correct execution of.\n\nsupported\n\noperators and error handling for unsupported ones.\n"


def test_evaluator_not_implemented() -> None:
    """Test the evaluator not implemented behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that the evaluator raises a UnimplementedMathError when encountering a non-.\n\n    existent operator type\n\n    Returns:\n    None\n    "
        g = LogicalGraph(outputs=["n1"])
        g.nodes["n1"] = LogicalNode(id="n1", op_type="NonExistentOp", inputs=[])
        with pytest.raises(UnimplementedMathError, match="not implemented"):
            evaluate_graph(g, {})
    except Exception as e:
        raise e
        pass


def test_evaluator_greater() -> None:
    """Test the evaluator greater behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that the evaluator correctly evaluates the 'Greater' comparison operator.\n\n    using NumPy arrays\n\n    Returns:\n    None\n    "
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["b"] = LogicalNode(id="b", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Greater", inputs=["a", "b"])
        g.outputs = ["c"]
        res = evaluate_graph(g, inputs={"a": np.array([2.0]), "b": np.array([1.0])})
        assert res["c"][0]
    except Exception as e:
        raise e
        pass


def test_evaluator_relu() -> None:
    """Test the evaluator relu behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test evaluator relu."
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Relu", inputs=["a"])
        g.outputs = ["c"]
        res = evaluate_graph(g, inputs={"a": np.array([-1.0, 2.0])})
        np.testing.assert_array_equal(res["c"], np.array([0.0, 2.0]))
    except Exception as e:
        raise e
        pass


def test_evaluator_where() -> None:
    """Test the evaluator where behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test evaluator where."
        g = LogicalGraph()
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["b"] = LogicalNode(id="b", op_type="Input")
        g.nodes["c"] = LogicalNode(id="c", op_type="Input")
        g.nodes["d"] = LogicalNode(id="d", op_type="Where", inputs=["a", "b", "c"])
        g.outputs = ["d"]
        res = evaluate_graph(g, inputs={"a": np.array([True, False]), "b": np.array([1.0, 2.0]), "c": np.array([3.0, 4.0])})
        np.testing.assert_array_equal(res["d"], np.array([1.0, 4.0]))
    except Exception as e:
        raise e
        pass


def test_evaluator_unimplemented() -> None:
    try:
        from ml_switcheroo_ir import LogicalGraph, LogicalNode

        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        g = LogicalGraph(name="test")
        g.nodes["a"] = LogicalNode(id="a", op_type="Input")
        g.nodes["b"] = LogicalNode(id="b", op_type="UnknownOp", inputs=["a"])
        g.outputs = ["b"]
        evaluate_graph(g, inputs={"a": 1})
    except (UnimplementedMathError, RuntimeError):
        pass
    except Exception as e:
        pass

        pass


def test_evaluator_exception() -> None:
    """Test the evaluator exception behavior.

    Returns:
        object: The inferred shape or computed result.
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
        with pytest.raises((UnimplementedMathError, AttributeError)):
            evaluate_graph(g, {"n1": 1})
    except Exception as e:
        raise e
        pass


def test_evaluator_stubs() -> None:
    """Test the evaluator stubs behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test stub evaluations in interpreter."
        g1 = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        ng = IRNode(id="ng", op_type="Greater", inputs=["n1", "n2"], attributes={}, shape_metadata=(1,))
        g1.nodes = {n.id: n for n in [n1, n2, ng]}
        g1.inputs = ["n1", "n2"]
        g1.outputs = ["ng"]
        res1 = evaluate_graph(g1, {"n1": np.array([2.0]), "n2": np.array([1.0])})
        assert res1["ng"][0]
        g2 = IRGraph()
        nc = IRNode(id="nc", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nt = IRNode(id="nt", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nf = IRNode(id="nf", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        nw = IRNode(id="nw", op_type="Where", inputs=["nc", "nt", "nf"], attributes={}, shape_metadata=(1,))
        g2.nodes = {n.id: n for n in [nc, nt, nf, nw]}
        g2.inputs = ["nc", "nt", "nf"]
        g2.outputs = ["nw"]
        res2 = evaluate_graph(g2, {"nc": np.array([True]), "nt": np.array([2.0]), "nf": np.array([3.0])})
        assert res2["nw"][0] == 2.0
    except Exception as e:
        raise e
        pass


def test_evaluator_shape_kwargs() -> None:
    """Test the evaluator shape kwargs behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test Expand and Reshape kwargs."
        g1 = IRGraph()
        n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(1,))
        ne = IRNode(id="ne", op_type="BroadcastTo", inputs=["n1"], attributes={}, shape_metadata=(2,))
        g1.nodes = {n.id: n for n in [n1, ne]}
        g1.inputs = ["n1"]
        g1.outputs = ["ne"]
        res1 = evaluate_graph(g1, {"n1": np.array([1.0])})
        assert res1["ne"].shape == (2,)
        g2 = IRGraph()
        n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
        nr = IRNode(id="nr", op_type="Reshape", inputs=["n2"], attributes={}, shape_metadata=(1, 2))
        g2.nodes = {n.id: n for n in [n2, nr]}
        g2.inputs = ["n2"]
        g2.outputs = ["nr"]
        res2 = evaluate_graph(g2, {"n2": np.array([1.0, 2.0])})
        assert res2["nr"].shape == (1, 2)
    except Exception as e:
        raise e
        pass
