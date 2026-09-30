"""Unit tests for Common Subexpression Elimination (CSE) pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.cse import cse_pass


def test_cse_basic():
    """Test CSE on a simple graph with redundant nodes."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"]),  # identical to n2
        "n4": IRNode(id="n4", op_type="Mul", inputs=["n2", "n3"]),
    }
    graph.outputs = ["n4"]

    assert cse_pass(graph) is True

    # n3 should be eliminated
    assert "n3" not in graph.nodes

    # n4's inputs should be updated to point to n2 instead of n3
    assert graph.nodes["n4"].inputs == ["n2", "n2"]


def test_cse_different_attributes():
    """Test CSE respects node attributes."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"], attributes={"alpha": 1.0}),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"], attributes={"alpha": 2.0}),  # different attr
    }
    graph.outputs = ["n2", "n3"]

    assert cse_pass(graph) is False
    assert "n2" in graph.nodes
    assert "n3" in graph.nodes


def test_cse_chained():
    """Test CSE can eliminate chained redundant expressions."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"]),  # == n2
        "n4": IRNode(id="n4", op_type="Mul", inputs=["n2", "n2"]),
        "n5": IRNode(id="n5", op_type="Mul", inputs=["n3", "n3"]),  # == n4 (since n3 == n2)
        "n6": IRNode(id="n6", op_type="Sub", inputs=["n4", "n5"]),
    }
    graph.outputs = ["n6"]

    assert cse_pass(graph) is True
    assert "n3" not in graph.nodes
    assert "n5" not in graph.nodes
    assert graph.nodes["n6"].inputs == ["n4", "n4"]


def test_cse_no_op():
    """Test CSE does nothing on graph with no common subexpressions."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Input", inputs=[]),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n2"]),
    }
    graph.outputs = ["n3"]
    assert cse_pass(graph) is False


def test_cse_output_update() -> None:
    """Test CSE correctly updates graph outputs."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"]),  # identical to n2
    }
    graph.outputs = ["n3"]

    assert cse_pass(graph) is True
    assert graph.outputs == ["n2"]


def test_cse_commutative_reordering() -> None:
    """Test CSE normalizes and recognizes commutative operation inputs."""
    graph = IRGraph()
    graph.nodes = {
        "a": IRNode(id="a", op_type="Input", inputs=[]),
        "b": IRNode(id="b", op_type="Input", inputs=[]),
        "add1": IRNode(id="add1", op_type="Add", inputs=["a", "b"]),
        "add2": IRNode(id="add2", op_type="Add", inputs=["b", "a"]),
        "sub1": IRNode(id="sub1", op_type="Sub", inputs=["a", "b"]),
        "sub2": IRNode(id="sub2", op_type="Sub", inputs=["b", "a"]),
    }
    graph.outputs = ["add1", "add2", "sub1", "sub2"]

    assert cse_pass(graph) is True
    assert "add2" not in graph.nodes
    assert "add1" in graph.nodes
    assert "sub1" in graph.nodes
    assert "sub2" in graph.nodes
    assert graph.outputs == ["add1", "add1", "sub1", "sub2"]


def test_cse_attributes_subgraph_and_collections() -> None:
    """Test CSE structural hashing with nested IRGraphs, dictionaries, and lists in attributes."""
    subgraph_a = IRGraph()
    subgraph_a.nodes = {
        "in0": IRNode(id="in0", op_type="Input", inputs=[]),
        "mul": IRNode(id="mul", op_type="Mul", inputs=["in0", "in0"]),
    }
    subgraph_a.outputs = ["mul"]

    subgraph_b = IRGraph()
    subgraph_b.nodes = {
        "in0": IRNode(id="in0", op_type="Input", inputs=[]),
        "mul": IRNode(id="mul", op_type="Mul", inputs=["in0", "in0"]),
    }
    subgraph_b.outputs = ["mul"]

    subgraph_c = IRGraph()
    subgraph_c.nodes = {
        "in0": IRNode(id="in0", op_type="Input", inputs=[]),
        "add": IRNode(id="add", op_type="Add", inputs=["in0", "in0"]),
    }
    subgraph_c.outputs = ["add"]

    graph = IRGraph()
    graph.nodes = {
        "x": IRNode(id="x", op_type="Input", inputs=[]),
        "op1": IRNode(
            id="op1",
            op_type="CustomOp",
            inputs=["x"],
            attributes={
                "subgraph": subgraph_a,
                "config": {"gamma": 0.5, "tags": ["relu", "norm"]},
                "params": [1, 2, 3],
            },
        ),
        "op2": IRNode(
            id="op2",
            op_type="CustomOp",
            inputs=["x"],
            attributes={
                "subgraph": subgraph_b,
                "config": {"gamma": 0.5, "tags": ["relu", "norm"]},
                "params": [1, 2, 3],
            },
        ),
        "op3": IRNode(
            id="op3",
            op_type="CustomOp",
            inputs=["x"],
            attributes={
                "subgraph": subgraph_c,
                "config": {"gamma": 0.5, "tags": ["relu", "norm"]},
                "params": [1, 2, 3],
            },
        ),
    }
    graph.outputs = ["op1", "op2", "op3"]

    assert cse_pass(graph) is True
    assert "op2" not in graph.nodes
    assert "op1" in graph.nodes
    assert "op3" in graph.nodes
    assert graph.outputs == ["op1", "op1", "op3"]


def test_cse_legacy_signature_and_input_rewrite() -> None:
    """Test legacy _compute_node_signature function and node input updating without output changes."""
    from ml_switcheroo_compiler.transforms.passes.cse import _compute_node_signature

    node = IRNode(id="sample", op_type="Add", inputs=["a", "b"], attributes={"dtype": "float32"})
    sig = _compute_node_signature(node, ["a", "b"])
    assert isinstance(sig, str) and len(sig) == 64

    # Test case where a subsequent node has its inputs rewritten,
    # but the graph outputs are unchanged (already canonical).
    graph = IRGraph()
    graph.nodes = {
        "i1": IRNode(id="i1", op_type="Input", inputs=[]),
        "a1": IRNode(id="a1", op_type="Add", inputs=["i1", "i1"]),
        "a2": IRNode(id="a2", op_type="Add", inputs=["i1", "i1"]),  # eliminated, maps to a1
        "m1": IRNode(id="m1", op_type="Mul", inputs=["a2", "i1"]),  # input rewritten to [a1, i1]
    }
    graph.outputs = ["m1"]  # outputs are already pointing to m1

    assert cse_pass(graph) is True
    assert "a2" not in graph.nodes
    assert graph.nodes["m1"].inputs == ["a1", "i1"]
    assert graph.outputs == ["m1"]
