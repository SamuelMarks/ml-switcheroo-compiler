# mypy: disable-error-code="untyped-decorator"
"""Tests for knapsack memory scheduler and declarative custom gradient subgraphs."""

from unittest.mock import patch

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.grad.checkpointing import (
    RematerializationPolicyModel,
    RematerializationRulesConfig,
    _evaluate_node_costs,
    _solve_01_knapsack_dp,
    get_rematerialization_policy,
    knapsack_memory_scheduler,
    load_cost_models,
    load_rematerialization_rules,
    solve_memory_budget_and_insert_checkpoints,
)
from ml_switcheroo_compiler.grad.custom_vjp_ops import custom_vjp
from ml_switcheroo_compiler.grad.jvp_vjp import custom_jvp


def test_load_cost_models() -> None:
    """Test loading cost models from cost_models.yaml.

    Returns:
        None
    """
    cost_data = load_cost_models()
    assert isinstance(cost_data, dict)
    assert "memory_sizes" in cost_data
    assert "compute_costs" in cost_data


def test_knapsack_memory_scheduler_within_budget() -> None:
    """Test scheduler returns empty list when graph activation memory fits budget.

    Returns:
        None
    """
    g = LogicalGraph(name="test_graph")
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
    # 100 floats = 400 bytes
    res = knapsack_memory_scheduler(g, memory_budget_bytes=1000)
    assert res == []


def test_knapsack_memory_scheduler_exceeds_budget() -> None:
    """Test scheduler designates nodes for checkpointing when memory exceeds budget.

    Returns:
        None
    """
    g = LogicalGraph(name="test_graph")
    # n1: 100 elements (400 bytes), light op (Add)
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
    # n2: 100 elements (400 bytes), heavy op (MatMul) -> high FLOPs
    g.nodes["n2"] = LogicalNode(id="n2", op_type="MatMul", shape_metadata=(10, 10))

    # Total memory = 800 bytes. Budget = 400 bytes.
    # Optimal knapsack should retain n2 (heavy compute) and checkpoint/rematerialize n1 (light compute).
    res = solve_memory_budget_and_insert_checkpoints(g, memory_budget_bytes=400)
    assert "n1" in res
    assert g.nodes["n1"].attributes.get("checkpoint") is True
    assert g.nodes["n1"].attributes.get("rematerialize") is True


def test_custom_vjp_declarative_subgraph() -> None:
    """Test custom_vjp allows attaching declarative backward IR subgraphs.

    Returns:
        None
    """

    @custom_vjp
    def dummy_fn(x: Tensor) -> Tensor:
        """Primal pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Scaled tensor.
        """
        return x * 2.0

    bwd_g = LogicalGraph(name="custom_bwd")
    bwd_g.inputs = ["cot", "x"]
    bwd_g.nodes["cot"] = LogicalNode(id="cot", op_type="Input")
    bwd_g.nodes["x"] = LogicalNode(id="x", op_type="Input")
    bwd_g.nodes["scaled"] = LogicalNode(id="scaled", op_type="Mul", inputs=["cot", "x"])
    bwd_g.outputs = ["scaled"]

    dummy_fn.defvjp_subgraph(bwd_g)
    assert dummy_fn.bwd_subgraph == bwd_g


def test_custom_jvp_declarative_subgraph() -> None:
    """Test custom_jvp allows attaching declarative forward/tangent IR subgraphs.

    Returns:
        None
    """

    @custom_jvp
    def dummy_fn(x: Tensor) -> Tensor:
        """Primal pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Scaled tensor.
        """
        return x * 3.0

    jvp_g = LogicalGraph(name="custom_jvp")
    jvp_g.inputs = ["x", "tan"]
    jvp_g.nodes["x"] = LogicalNode(id="x", op_type="Input")
    jvp_g.nodes["tan"] = LogicalNode(id="tan", op_type="Input")
    jvp_g.nodes["scaled_tan"] = LogicalNode(id="scaled_tan", op_type="Mul", inputs=["tan", "x"])
    jvp_g.outputs = ["scaled_tan"]

    dummy_fn.defjvp_subgraph(jvp_g)
    assert dummy_fn.jvp_subgraph == jvp_g


def test_load_cost_models_and_remat_rules_nonexistent() -> None:
    """Test loading models from nonexistent paths falls back properly.

    Returns:
        None
    """
    cm = load_cost_models(path="/nonexistent/cost_models.yaml")
    assert cm == {}

    with patch("os.path.exists", return_value=False):
        remat_cfg = load_rematerialization_rules(path="/nonexistent/rematerialization_rules.yaml")
        assert remat_cfg.default_policy == "recompute_all"


def test_evaluate_node_costs_edge_cases() -> None:
    """Test _evaluate_node_costs with non-integer shapes, dtype overrides, and non-dict sizes.

    Returns:
        None
    """
    # Test node with non-integer and negative dims, float64 dtype, and non-dict memory_sizes
    node = LogicalNode(
        id="edge_node",
        op_type="UnknownOp",
        shape_metadata=(-1, "str_dim", 4),
        attributes={"dtype": "float64"},
    )
    footprint, flop = _evaluate_node_costs(
        node=node,
        memory_sizes=None,  # type: ignore[arg-type]
        heavy_ops={"Conv"},
        light_ops={"Add"},
        heavy_cost=100,
        light_cost=10,
        default_cost=50,
    )
    assert footprint == 16
    assert flop == 200


def test_knapsack_memory_scheduler_retains_item() -> None:
    """Test knapsack scheduler retains higher-value item while evicting lower-value item.

    Returns:
        None
    """
    g = LogicalGraph(name="test_retain_graph")
    g.nodes["in_node"] = LogicalNode(id="in_node", op_type="Input")
    g.nodes["const_node"] = LogicalNode(id="const_node", op_type="Constant")
    g.nodes["cp_node"] = LogicalNode(id="cp_node", op_type="Checkpoint")
    # n1: Add (light op, flop 10 * 100 = 1000, 400 bytes)
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", shape_metadata=(10, 10))
    # n2: Multiply (default cost op, flop 50 * 100 = 5000, 400 bytes)
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Multiply", shape_metadata=(10, 10))

    # Total memory = 800 bytes. Budget = 500 bytes.
    # Knapsack can fit one item (400 bytes <= 500). It retains n2 and evicts n1.
    res = knapsack_memory_scheduler(g, memory_budget_bytes=500)
    assert res == ["n1"]
    assert g.nodes["n1"].attributes.get("checkpoint") is True
    assert g.nodes["n2"].attributes.get("checkpoint") is None


def test_solve_01_knapsack_dp_scaled_and_unselected() -> None:
    """Test _solve_01_knapsack_dp with capacity exceeding max_cells and unselected item branch.

    Returns:
        None
    """
    retained = _solve_01_knapsack_dp(5000, [1000, 2000], [10, 20])
    assert 0 in retained
    assert 1 in retained

    no_fit = _solve_01_knapsack_dp(10, [20], [100])
    assert len(no_fit) == 0

    # Tests branch where dp[c - w] + v <= dp[c]
    retained_lower = _solve_01_knapsack_dp(cap=2, weights=[2, 2], values=[10, 5])
    assert retained_lower == {0}


def test_get_rematerialization_policy_branches() -> None:
    """Test get_rematerialization_policy model input, fallback, and error handling.

    Returns:
        None
    """
    policy = RematerializationPolicyModel(description="custom_policy")
    assert get_rematerialization_policy(policy) is policy

    mock_cfg = RematerializationRulesConfig(
        default_policy="fallback",
        policies={},
        target_ops=["Add"],
        high_cost_ops=[],
    )
    with patch("ml_switcheroo_compiler.grad.checkpointing.load_rematerialization_rules", return_value=mock_cfg):
        resolved = get_rematerialization_policy(None)
        assert resolved.target_ops == ["Add"]

    with pytest.raises(ValueError, match="not found in configuration"):
        with patch("ml_switcheroo_compiler.grad.checkpointing.load_rematerialization_rules", return_value=mock_cfg):
            get_rematerialization_policy("unknown_policy")
