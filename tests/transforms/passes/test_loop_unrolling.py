"""Test loop unrolling pass."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import AttributeValue, NodeDict

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.loop_unrolling import loop_unrolling_pass

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_loop_unrolling_pass() -> None:
    """Test unrolling loops with statically analyzable bound."""
    # Create body graph
    body_graph = IRGraph()
    body_graph.inputs = ["in_val"]

    n1 = IRNode(id="n1", op_type="Add", inputs=["in_val", "const_1"])
    body_graph.nodes["n1"] = n1
    body_graph.outputs = ["n1"]

    # Create main graph
    graph = IRGraph()
    loop_node = IRNode(id="loop", op_type="WhileLoop", inputs=["init_val"], attributes=typing.cast("dict[str, AttributeValue]", {"max_iterations": 2, "body": body_graph}))
    graph.nodes["loop"] = loop_node

    # Run unrolling pass
    optimized = loop_unrolling_pass(graph)

    # The loop should be unrolled twice (n1 and n1_iter0, etc.)
    # We expect 3 nodes: Identity, loop_iter0_n1, loop_iter1_n1
    assert "loop" in optimized.nodes
    assert optimized.nodes["loop"].op_type == "Identity"
    assert optimized.nodes["loop"].inputs == ["loop_iter1_n1"]

    assert "loop_iter0_n1" in optimized.nodes
    assert optimized.nodes["loop_iter0_n1"].inputs == ["init_val", "const_1"]

    assert "loop_iter1_n1" in optimized.nodes
    assert optimized.nodes["loop_iter1_n1"].inputs == ["loop_iter0_n1", "const_1"]


def test_loop_unrolling_no_unroll() -> None:
    """Test when loop should not be unrolled."""
    graph = IRGraph()
    loop_node = IRNode(
        id="loop",
        op_type="WhileLoop",
        inputs=["init_val"],
        attributes={"max_iterations": 100},  # Exceeds default limit (10)
    )
    graph.nodes["loop"] = loop_node

    optimized = loop_unrolling_pass(graph)
    assert "loop" in optimized.nodes
    assert optimized.nodes["loop"].op_type == "WhileLoop"


def test_detect_static_bound() -> None:
    """Test heuristics boundary detection."""
    from ml_switcheroo_compiler.transforms.passes.loop_unrolling import detect_static_bound

    node = IRNode(id="n", op_type="WhileLoop")
    # Empty
    assert detect_static_bound(node, []) is None

    # Steps attribute
    node_steps = IRNode(id="n_steps", op_type="WhileLoop", attributes={"steps": 4})
    assert detect_static_bound(node_steps, []) == 4

    # ForiLoop bounds (line 86)
    node_fori = IRNode(id="n_fori", op_type="ForiLoop", attributes={"lower": 2, "upper": 6})
    assert detect_static_bound(node_fori, []) == 4
    node_fori_neg = IRNode(id="n_fori_neg", op_type="ForiLoop", attributes={"lower": 10, "upper": 5})
    assert detect_static_bound(node_fori_neg, []) == 0

    # Heuristics
    heuristics: list[dict[str, AttributeValue]] = [{"op_type": "WhileLoop", "max_iterations": 5}]
    assert detect_static_bound(node, heuristics) == 5


def test_unroll_fori_loop_node() -> None:
    """Test unrolling a ForiLoop with index and state inputs."""
    body = IRGraph()
    body.inputs = ["idx", "x"]
    body.nodes = NodeDict({"add": IRNode(id="add", op_type="Add", inputs=["idx", "x"])})
    body.outputs = ["add"]

    g = IRGraph()
    fori_node = IRNode(
        id="fori",
        op_type="ForiLoop",
        inputs=["init_x"],
        attributes=typing.cast("dict[str, AttributeValue]", {"lower": 0, "upper": 3, "body": body}),
    )
    g.nodes = NodeDict({"init_x": IRNode(id="init_x", op_type="Input"), "fori": fori_node})
    res = loop_unrolling_pass(g)
    assert "fori" in res.nodes
    assert res.nodes["fori"].op_type == "Identity"


def test_loop_unrolling_extra_coverage() -> None:
    """Test function test_loop_unrolling_extra_coverage."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, LogicalNode
    from ml_switcheroo_compiler.transforms.passes.loop_unrolling import _load_config, detect_static_bound, loop_unrolling_pass

    # test yaml missing
    with patch("os.path.exists", return_value=False):
        assert _load_config() == {}

    # detect_static_bound branches
    node = IRNode(id="n1", op_type="Add")
    assert detect_static_bound(node, []) is None

    node = IRNode(id="n1", op_type="WhileLoop")
    heuristics = [{"op_type": "Add"}, {"op_type": "WhileLoop", "other": 1}, {"op_type": "WhileLoop", "max_iterations": "string"}, {"op_type": "WhileLoop", "max_iterations": 5}]
    assert typing.cast("EagerCallableProtocol", detect_static_bound)(node, heuristics) == 5

    # graph loops unrolling branches
    g = IRGraph()
    ln = LogicalNode(id="ln1", op_type="Input")
    g.nodes["ln1"] = ln

    n2 = IRNode(id="n2", op_type="WhileLoop")
    n2.attributes["max_iterations"] = 3
    # No body_graph, should skip
    g.nodes["n2"] = n2

    # Body graph with logical nodes
    n3 = IRNode(id="n3", op_type="WhileLoop", inputs=["ln1"])
    n3.attributes["max_iterations"] = 1

    body = IRGraph()
    body.inputs = ["body_in"]
    body.outputs = ["body_out"]
    bln = LogicalNode(id="body_node", op_type="Add", inputs=["body_in"])
    body.nodes["body_node"] = bln
    n3.attributes["body"] = typing.cast("AttributeValue", body)

    g.nodes["n3"] = n3

    # Add stream and device missing on IRNode
    n4 = IRNode(id="n4", op_type="WhileLoop", inputs=["ln1"])
    n4.attributes["max_iterations"] = 1

    body2 = IRGraph()
    body2.inputs = ["body_in"]
    body2.outputs = ["body_out"]
    n4_inner = IRNode(id="n4_inner", op_type="Add", inputs=["body_in"], device="cpu", stream="s0")
    body2.nodes["n4_inner"] = n4_inner
    n4.attributes["body"] = typing.cast("AttributeValue", body2)
    g.nodes["n4"] = n4

    loop_unrolling_pass(g)


def test_loop_unrolling_clone_logical_node_and_non_irnode() -> None:
    """Test clone_subgraph with LogicalNode and unroll_loops with Identity nodes."""
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
    from ml_switcheroo_compiler.transforms.passes.loop_unrolling import clone_subgraph, unroll_loops

    # 1. clone_subgraph with LogicalNode
    pure_node = LogicalNode(id="ln1", op_type="Add", inputs=[])
    sub_g = IRGraph(name="sub_g")
    sub_g.nodes["ln1"] = pure_node
    id_map: dict[str, str] = {}
    cloned = clone_subgraph(sub_g, prefix="step_0", id_map=id_map)
    assert len(cloned) == 1
    assert cloned[0].id == "step_0_ln1"

    # 2. unroll_loops with non-loop node
    g = IRGraph(name="test_g")
    g.nodes["plain_node"] = LogicalNode(id="plain_node", op_type="Identity", inputs=[])
    res = unroll_loops(g)
    assert "plain_node" in res.nodes
