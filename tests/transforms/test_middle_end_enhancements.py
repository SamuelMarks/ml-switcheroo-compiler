"""Unit and integration tests for middle-end optimization passes enhancements (DCE, CSE, Constant Folding, Operator Fusion)."""

from __future__ import annotations

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass
from ml_switcheroo_compiler.transforms.passes.cse import (
    compute_node_structural_hash,
    cse_pass,
)
from ml_switcheroo_compiler.transforms.passes.dce import dce_pass
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    operator_fusion_pass,
)


def test_dce_nested_subgraph_and_side_effect_free_loop() -> None:
    """Test DCE eliminates side-effect-free loops and prunes dead nodes in subgraphs."""
    graph = IRGraph()
    x = IRNode(id="x", op_type="Input")
    y = IRNode(id="y", op_type="Input")

    # Construct nested loop body without side effects
    body = IRGraph()
    b_in = IRNode(id="b_in", op_type="Input")
    b_dead = IRNode(id="b_dead", op_type="Constant", attributes={"value": 42})
    b_add = IRNode(id="b_add", op_type="Add", inputs=["b_in", "b_in"])
    body.nodes = {"b_in": b_in, "b_dead": b_dead, "b_add": b_add}
    body.outputs = ["b_add"]

    # Loop node whose output is never consumed and has no side effects
    dead_loop = IRNode(id="dead_loop", op_type="WhileLoop", inputs=["x"], attributes={"body": body})
    used_add = IRNode(id="used_add", op_type="Add", inputs=["x", "y"])

    graph.nodes = {"x": x, "y": y, "dead_loop": dead_loop, "used_add": used_add}
    graph.outputs = ["used_add"]

    assert dce_pass(graph) is True
    assert "dead_loop" not in graph.nodes
    assert "used_add" in graph.nodes
    assert "x" in graph.nodes
    assert "y" in graph.nodes


def test_dce_preserves_subgraph_with_side_effects() -> None:
    """Test DCE preserves loops containing side-effecting operations."""
    graph = IRGraph()
    x = IRNode(id="x", op_type="Input")

    # Body containing Print side-effect
    body = IRGraph()
    b_in = IRNode(id="b_in", op_type="Input")
    b_print = IRNode(id="b_print", op_type="Print", inputs=["b_in"])
    body.nodes = {"b_in": b_in, "b_print": b_print}
    body.outputs = ["b_in"]

    side_effect_loop = IRNode(id="se_loop", op_type="WhileLoop", inputs=["x"], attributes={"body": body})
    graph.nodes = {"x": x, "se_loop": side_effect_loop}
    graph.outputs = []  # No graph outputs, but loop has side effect

    assert dce_pass(graph) is False
    assert "se_loop" in graph.nodes


def test_cse_commutative_structural_hashing() -> None:
    """Test CSE structural hashing deduplicates commutative operations regardless of input order."""
    graph = IRGraph()
    a = IRNode(id="a", op_type="Input")
    b = IRNode(id="b", op_type="Input")
    # Add is commutative, so Add(a, b) and Add(b, a) must hash identically
    add1 = IRNode(id="add1", op_type="Add", inputs=["a", "b"])
    add2 = IRNode(id="add2", op_type="Add", inputs=["b", "a"])
    mul = IRNode(id="mul", op_type="Mul", inputs=["add1", "add2"])

    graph.nodes = {"a": a, "b": b, "add1": add1, "add2": add2, "mul": mul}
    graph.outputs = ["mul"]

    assert cse_pass(graph) is True
    assert "add2" not in graph.nodes
    assert graph.nodes["mul"].inputs == ["add1", "add1"]


def test_cse_nested_subgraph_structural_hashing() -> None:
    """Test CSE structural hashing hashes nested IRGraph attributes."""
    g1 = IRGraph()
    g1.nodes = {"in1": IRNode(id="in1", op_type="Input"), "c1": IRNode(id="c1", op_type="Constant", attributes={"value": 1})}
    g1.outputs = ["c1"]

    g2 = IRGraph()
    g2.nodes = {"in2": IRNode(id="in2", op_type="Input"), "c2": IRNode(id="c2", op_type="Constant", attributes={"value": 1})}
    g2.outputs = ["c2"]

    n1 = IRNode(id="n1", op_type="CustomOp", inputs=["in_common"], attributes={"sub": g1})
    n2 = IRNode(id="n2", op_type="CustomOp", inputs=["in_common"], attributes={"sub": g2})

    hash1 = compute_node_structural_hash(n1, ["in_common"])
    hash2 = compute_node_structural_hash(n2, ["in_common"])
    assert hash1 == hash2


def test_constant_folding_multi_level_fixpoint() -> None:
    """Test constant folding propagates through multiple dependent constant operations until fixpoint."""
    graph = IRGraph()
    c1 = IRNode(id="c1", op_type="Constant", attributes={"value": 10})
    c2 = IRNode(id="c2", op_type="Constant", attributes={"value": 5})
    add = IRNode(id="add", op_type="Add", inputs=["c1", "c2"])
    c3 = IRNode(id="c3", op_type="Constant", attributes={"value": 2})
    mul = IRNode(id="mul", op_type="Mul", inputs=["add", "c3"])

    graph.nodes = {"c1": c1, "c2": c2, "add": add, "c3": c3, "mul": mul}
    graph.outputs = ["mul"]

    assert constant_folding_pass(graph) is True
    assert graph.nodes["add"].op_type == "Constant"
    assert graph.nodes["add"].attributes["value"] == 15
    assert graph.nodes["mul"].op_type == "Constant"
    assert graph.nodes["mul"].attributes["value"] == 30


def test_operator_fusion_linear_relu_and_swiglu() -> None:
    """Test newly added declarative fusion patterns for LinearRelu and SwiGLU."""
    # Test LinearRelu fusion
    graph_lr = IRGraph()
    x = IRNode(id="x", op_type="Input")
    w = IRNode(id="w", op_type="Input")
    b = IRNode(id="b", op_type="Input")
    linear = IRNode(id="linear", op_type="Linear", inputs=["x", "w", "b"])
    relu = IRNode(id="relu", op_type="Relu", inputs=["linear"])
    graph_lr.nodes = {"x": x, "w": w, "b": b, "linear": linear, "relu": relu}
    graph_lr.outputs = ["relu"]

    assert operator_fusion_pass(graph_lr) is True
    assert any(n.op_type == "LinearRelu" for n in graph_lr.nodes.values())

    # Test SwiGLU fusion
    graph_swiglu = IRGraph()
    s_in = IRNode(id="s_in", op_type="Input")
    y_in = IRNode(id="y_in", op_type="Input")
    silu = IRNode(id="silu", op_type="SiLU", inputs=["s_in"])
    mul = IRNode(id="mul", op_type="Multiply", inputs=["silu", "y_in"])
    graph_swiglu.nodes = {"s_in": s_in, "y_in": y_in, "silu": silu, "mul": mul}
    graph_swiglu.outputs = ["mul"]

    assert operator_fusion_pass(graph_swiglu) is True
    assert any(n.op_type == "SwiGLU" for n in graph_swiglu.nodes.values())


def test_dce_prunes_dead_output_ids() -> None:
    """Test DCE prunes non-existent or unreferenced nodes from graph outputs."""
    graph = IRGraph()
    x = IRNode(id="x", op_type="Input")
    graph.nodes = {"x": x}
    # Output list contains a dangling node ID that was removed or doesn't exist
    graph.outputs = ["x", "non_existent_output"]

    assert dce_pass(graph) is True
    assert graph.outputs == ["x"]


def test_cse_dict_and_list_attributes_and_legacy_signature() -> None:
    """Test CSE structural hashing of dict and list attributes and legacy signature compatibility."""
    from ml_switcheroo_compiler.transforms.passes.cse import _compute_node_signature

    n_dict = IRNode(id="nd", op_type="Custom", attributes={"nested": {"k": "v"}, "seq": [1, 2, 3]})
    hash_val = compute_node_structural_hash(n_dict, [])
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64

    # Test legacy signature wrapper
    assert _compute_node_signature(n_dict, []) == hash_val


def test_constant_folding_backend_item_and_exception_handling() -> None:
    """Test constant folding with backend item unwrapping and exception tolerance."""
    from unittest.mock import MagicMock, patch

    graph = IRGraph()
    c1 = IRNode(id="c1", op_type="Constant", attributes={"value": 1})
    c2 = IRNode(id="c2", op_type="Constant", attributes={"value": 2})
    op_fail = IRNode(id="op_fail", op_type="FaultyOp", inputs=["c1", "c2"])

    graph.nodes = {"c1": c1, "c2": c2, "op_fail": op_fail}
    graph.outputs = ["op_fail"]

    # Test ValueError/RuntimeError during constant folding is gracefully handled
    with patch("ml_switcheroo_compiler.transforms.passes.constant_folding.evaluate_graph", side_effect=ValueError("Invalid op")):
        assert constant_folding_pass(graph) is False

    # Test backend.item branch and fallback when get_active_backend is None
    graph_ok = IRGraph()
    c1 = IRNode(id="c1", op_type="Constant", attributes={"value": 1})
    c2 = IRNode(id="c2", op_type="Constant", attributes={"value": 2})
    add = IRNode(id="add", op_type="Add", inputs=["c1", "c2"])
    graph_ok.nodes = {"c1": c1, "c2": c2, "add": add}
    graph_ok.outputs = ["add"]

    import numpy as np

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.array([42])
    mock_backend.item.return_value = 3

    # Test _evaluate_constant_node with backend.item
    from ml_switcheroo_compiler.transforms.passes.constant_folding import _evaluate_constant_node

    val_res = _evaluate_constant_node(add, ["c1", "c2"], graph_ok, mock_backend)
    assert val_res == 3

    # Test _evaluate_constant_node without backend.item (falling back to val.item())
    mock_backend_no_item = MagicMock()
    del mock_backend_no_item.item
    mock_backend_no_item.execute_op.return_value = np.array([42])
    val_res2 = _evaluate_constant_node(add, ["c1", "c2"], graph_ok, mock_backend_no_item)
    assert val_res2 == 42

    # Test BackendRegistry.get exception fallback in constant_folding_pass
    with patch("ml_switcheroo_compiler.transforms.passes.constant_folding.get_active_backend", return_value=None), patch("ml_switcheroo_compiler.transforms.passes.constant_folding.BackendRegistry.get", side_effect=RuntimeError("Registry error")):
        assert constant_folding_pass(graph) is False


def test_spmd_all_gather_and_all_to_all_injection() -> None:
    """Test spmd_partitioning_pass injecting AllGather and AllToAll."""
    from ml_switcheroo_compiler.transforms.passes.spmd import (
        SPMDShardingAnnotation,
        spmd_partitioning_pass,
    )

    # AllGather injection
    g_ag = IRGraph()
    x = IRNode(id="x", op_type="Input")
    ag_node = IRNode(id="ag_in", op_type="Identity", inputs=["x"], attributes={"inject_collective": "AllGather"})
    ag_node.sharding = SPMDShardingAnnotation(mesh="grid", mesh_mapping=["tp"])
    g_ag.nodes = {"x": x, "ag_in": ag_node}
    g_ag.outputs = ["ag_in"]

    assert spmd_partitioning_pass(g_ag) is True
    assert "ag_in_allgather" in g_ag.nodes
    assert g_ag.nodes["ag_in_allgather"].op_type == "AllGather"

    # AllToAll injection
    g_a2a = IRGraph()
    x2 = IRNode(id="x2", op_type="Input")
    a2a_node = IRNode(id="a2a_in", op_type="Identity", inputs=["x2"], attributes={"inject_collective": "AllToAll"})
    a2a_node.sharding = SPMDShardingAnnotation(mesh="grid", mesh_mapping=["tp"])
    g_a2a.nodes = {"x2": x2, "a2a_in": a2a_node}
    g_a2a.outputs = ["a2a_in"]

    assert spmd_partitioning_pass(g_a2a) is True
    assert "a2a_in_alltoall" in g_a2a.nodes
    assert g_a2a.nodes["a2a_in_alltoall"].op_type == "AllToAll"
