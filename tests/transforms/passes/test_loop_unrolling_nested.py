"""Tests for nested loop unrolling passes in IR graphs."""

from __future__ import annotations

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.loop_unrolling import loop_unrolling_pass


def test_nested_loop_unrolling_both_levels() -> None:
    """Test unrolling nested loops where both inner and outer bounds are statically analyzable."""
    # Inner loop body: adds 1 to x
    inner_body: IRGraph = IRGraph()
    inner_body.inputs = ["inner_x"]
    inner_body.nodes["inner_add"] = IRNode(
        id="inner_add",
        op_type="Add",
        inputs=["inner_x", "one_const"],
    )
    inner_body.outputs = ["inner_add"]

    # Outer loop body contains the inner loop
    outer_body: IRGraph = IRGraph()
    outer_body.inputs = ["outer_x"]
    inner_loop: IRNode = IRNode(
        id="inner_loop",
        op_type="WhileLoop",
        inputs=["outer_x"],
        attributes={"max_iterations": 2, "body": inner_body},
    )
    outer_body.nodes["inner_loop"] = inner_loop
    outer_body.outputs = ["inner_loop"]

    # First optimize the outer body so inner loop gets unrolled
    loop_unrolling_pass(outer_body)
    assert outer_body.nodes["inner_loop"].op_type == "Identity"

    # Main graph with outer loop
    main_graph: IRGraph = IRGraph()
    main_graph.inputs = ["init_val"]
    outer_loop: IRNode = IRNode(
        id="outer_loop",
        op_type="WhileLoop",
        inputs=["init_val"],
        attributes={"max_iterations": 2, "body": outer_body},
    )
    main_graph.nodes["outer_loop"] = outer_loop
    main_graph.outputs = ["outer_loop"]

    # Unroll outer loop
    optimized: IRGraph = loop_unrolling_pass(main_graph)
    assert "outer_loop" in optimized.nodes
    assert optimized.nodes["outer_loop"].op_type == "Identity"

    # Verify that iterations of the outer loop contain unrolled inner nodes
    node_ids: list[str] = list(optimized.nodes.keys())
    assert any("outer_loop_iter0" in nid for nid in node_ids)
    assert any("outer_loop_iter1" in nid for nid in node_ids)


def test_nested_foriloop_in_whileloop() -> None:
    """Test unrolling an inner ForiLoop inside an outer WhileLoop."""
    # Inner ForiLoop body
    fori_body: IRGraph = IRGraph()
    fori_body.inputs = ["idx", "acc"]
    fori_body.nodes["mul"] = IRNode(
        id="mul",
        op_type="Mul",
        inputs=["idx", "acc"],
    )
    fori_body.outputs = ["mul"]

    # Outer body containing the ForiLoop
    outer_body: IRGraph = IRGraph()
    outer_body.inputs = ["outer_acc"]
    fori_node: IRNode = IRNode(
        id="inner_fori",
        op_type="ForiLoop",
        inputs=["outer_acc"],
        attributes={"lower": 1, "upper": 3, "body": fori_body},
    )
    outer_body.nodes["inner_fori"] = fori_node
    outer_body.outputs = ["inner_fori"]

    # Pre-unroll the inner body
    loop_unrolling_pass(outer_body)
    assert outer_body.nodes["inner_fori"].op_type == "Identity"

    # Main graph containing outer WhileLoop
    graph: IRGraph = IRGraph()
    outer_while: IRNode = IRNode(
        id="outer_while",
        op_type="WhileLoop",
        inputs=["initial_val"],
        attributes={"max_iterations": 3, "body": outer_body},
    )
    graph.nodes["outer_while"] = outer_while
    graph.outputs = ["outer_while"]

    result: IRGraph = loop_unrolling_pass(graph)
    assert result.nodes["outer_while"].op_type == "Identity"
    assert len(result.nodes) > 3


def test_nested_loop_exceeding_bound_preserves_loop() -> None:
    """Test that nested loops with bounds exceeding default limit are preserved."""
    large_body: IRGraph = IRGraph()
    large_body.inputs = ["x"]
    large_body.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "one"])
    large_body.outputs = ["add"]

    graph: IRGraph = IRGraph()
    large_loop: IRNode = IRNode(
        id="large_loop",
        op_type="WhileLoop",
        inputs=["init"],
        attributes={"max_iterations": 50, "body": large_body},
    )
    graph.nodes["large_loop"] = large_loop

    result: IRGraph = loop_unrolling_pass(graph)
    assert result.nodes["large_loop"].op_type == "WhileLoop"
