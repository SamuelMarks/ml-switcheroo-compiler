"""Unit tests for Dead Code Elimination (DCE) pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.dce import _find_side_effect_nodes, dce_pass


def test_dce_basic():
    """Test DCE on a simple graph with dead nodes."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),
        "n3": IRNode(id="n3", op_type="Mul", inputs=["n2", "n1"]),  # dead
        "n4": IRNode(id="n4", op_type="Sub", inputs=["n2", "n2"]),
    }
    graph.outputs = ["n4"]

    assert dce_pass(graph) is True
    assert "n3" not in graph.nodes
    assert "n4" in graph.nodes
    assert "n2" in graph.nodes
    assert "n1" in graph.nodes


def test_dce_chained_dead_nodes():
    """Test DCE on a chain of dead nodes."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"]),  # dead
        "n3": IRNode(id="n3", op_type="Add", inputs=["n2", "n2"]),  # dead
    }
    graph.outputs = ["n1"]

    assert dce_pass(graph) is True
    assert "n2" not in graph.nodes
    assert "n3" not in graph.nodes


def test_dce_side_effect_preservation():
    """Test that DCE preserves nodes with side effects."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input", inputs=[]),
        "n2": IRNode(id="n2", op_type="Seed", inputs=["n1"]),  # Side effect, preserve
        "n3": IRNode(id="n3", op_type="Add", inputs=["n1", "n1"]),  # dead
    }
    graph.outputs = ["n1"]

    assert _find_side_effect_nodes(graph) == {"n2"}
    assert dce_pass(graph) is True
    assert "n3" not in graph.nodes
    assert "n2" in graph.nodes  # Preserved
    assert "n1" in graph.nodes


def test_dce_no_op():
    """Test DCE on a graph where all nodes are reachable."""
    graph = IRGraph()
    graph.nodes = {"n1": IRNode(id="n1", op_type="Input", inputs=[]), "n2": IRNode(id="n2", op_type="Add", inputs=["n1", "n1"])}
    graph.outputs = ["n2"]

    assert dce_pass(graph) is False
    assert len(graph.nodes) == 2


def test_dce_pass_no_outputs_attribute() -> None:
    """Test DCE pass on a graph object without an outputs attribute."""
    from ml_switcheroo_ir import LogicalNode

    class MockGraphWithoutOutputs:
        """Mock graph missing outputs attribute."""

        def __init__(self) -> None:
            """Initialize mock graph."""
            self.nodes: dict[str, LogicalNode] = {}

    mock_graph = MockGraphWithoutOutputs()
    res = dce_pass(mock_graph)
    assert res is False


def test_dce_nested_subgraphs() -> None:
    """Test DCE pass with nested subgraphs."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    subgraph = LogicalGraph(name="sub", outputs=["used_in_sub"])
    sub_used = LogicalNode(id="used_in_sub", op_type="Constant", attributes={"value": 1.0})
    sub_unused = LogicalNode(id="unused_in_sub", op_type="Constant", attributes={"value": 2.0})
    subgraph.nodes = {"used_in_sub": sub_used, "unused_in_sub": sub_unused}
    parent = LogicalGraph(name="parent", outputs=["cond_node"])
    cond_node = LogicalNode(id="cond_node", op_type="Cond", attributes={"subgraph": subgraph})
    parent.nodes = {"cond_node": cond_node}
    res = dce_pass(parent)
    assert res is True
    assert "unused_in_sub" not in subgraph.nodes


def test_dce_outputs_branches() -> None:
    """Test DCE pass when outputs are unchanged and when dead outputs are pruned."""
    from ml_switcheroo_ir import LogicalNode

    node = LogicalNode(id="in_0", op_type="Input")
    graph1 = IRGraph(name="dce_unmodified", nodes={"in_0": node}, outputs=["in_0"])
    res1 = dce_pass(graph1)
    assert res1 is False
    graph2 = IRGraph(name="dce_modified", nodes={"in_0": node}, outputs=["in_0", "dead_out"])
    res2 = dce_pass(graph2)
    assert res2 is True
    assert graph2.outputs == ["in_0"]


def test_dce_node_subgraphs() -> None:
    """Test DCE pass recursively cleans up dead code inside node.subgraphs."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    subgraph = LogicalGraph(name="then_branch", outputs=["live"])
    subgraph.nodes["live"] = LogicalNode(id="live", op_type="Constant", attributes={"value": 1.0})
    subgraph.nodes["dead"] = LogicalNode(id="dead", op_type="Constant", attributes={"value": 2.0})

    parent = LogicalGraph(name="parent", outputs=["if_node"])
    parent.nodes["if_node"] = LogicalNode(
        id="if_node",
        op_type="If",
        subgraphs={"then_branch": subgraph},
    )

    assert dce_pass(parent) is True
    assert "dead" not in subgraph.nodes
    assert "live" in subgraph.nodes


def test_dce_subgraphs_with_side_effects() -> None:
    """Test that DCE detects side effects inside node.subgraphs and node.attributes."""
    # 0. Non-IRGraph in node.subgraphs
    node_non_graph = IRNode(id="n_non_graph", op_type="CustomOp", subgraphs={"dummy": "not_a_graph"})
    g0 = IRGraph(name="g0", nodes={"n_non_graph": node_non_graph}, outputs=["n_non_graph"])
    assert _find_side_effect_nodes(g0) == set()
    assert dce_pass(g0) is False

    # 1. Side effect inside node.subgraphs
    sub_se1 = IRGraph(name="sub1", outputs=["s1"])
    sub_se1.nodes["s1"] = IRNode(id="s1", op_type="Seed", inputs=[])
    node_with_subgraphs = IRNode(id="n_sub", op_type="CustomOp", subgraphs={"body": sub_se1})
    g1 = IRGraph(name="g1", nodes={"n_sub": node_with_subgraphs}, outputs=[])
    assert _find_side_effect_nodes(g1) == {"n_sub"}

    # 2. Side effect inside node.attributes["body"]
    sub_se2 = IRGraph(name="sub2", outputs=["s2"])
    sub_se2.nodes["s2"] = IRNode(id="s2", op_type="Seed", inputs=[])
    node_with_attrs = IRNode(id="n_attr", op_type="CustomOp", attributes={"body": sub_se2})
    g2 = IRGraph(name="g2", nodes={"n_attr": node_with_attrs}, outputs=[])
    assert _find_side_effect_nodes(g2) == {"n_attr"}

    # 3. Subgraph without side effects inside attributes
    sub_clean = IRGraph(name="sub_clean", outputs=["c1"])
    sub_clean.nodes["c1"] = IRNode(id="c1", op_type="Constant", inputs=[])
    node_clean = IRNode(id="n_clean", op_type="CustomOp", attributes={"body": sub_clean})
    g3 = IRGraph(name="g3", nodes={"n_clean": node_clean}, outputs=[])
    assert _find_side_effect_nodes(g3) == set()


def test_dce_prune_unmodified_subgraphs() -> None:
    """Test DCE pass when subgraphs in both attributes and subgraphs are already minimal (unmodified)."""
    sub1 = IRGraph(name="sub1", outputs=["c1"])
    sub1.nodes["c1"] = IRNode(id="c1", op_type="Constant", inputs=[])
    sub2 = IRGraph(name="sub2", outputs=["c2"])
    sub2.nodes["c2"] = IRNode(id="c2", op_type="Constant", inputs=[])

    parent = IRGraph(name="parent", outputs=["top_node"])
    parent.nodes["top_node"] = IRNode(
        id="top_node",
        op_type="If",
        inputs=[],
        subgraphs={"sub1": sub1},
        attributes={"body": sub2},
    )

    # Neither sub1 nor sub2 has dead code, parent node is the output
    res = dce_pass(parent)
    assert res is False
