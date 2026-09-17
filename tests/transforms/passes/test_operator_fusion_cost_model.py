"""Unit tests for operator fusion cost model, commutative pattern matching, and rule discovery."""

from __future__ import annotations

from unittest.mock import patch

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    DeclarativeTargetFusionRule,
    MemoryAwareCostModel,
    NodePattern,
    _discover_fusion_patterns,
    match_pattern,
)


def test_operator_fusion_memory_aware() -> None:
    """Verify memory-aware cost model validation with thresholds and symbolic shapes."""
    config: dict[str, object] = {"max_fusion_memory_bytes": 1024, "memory_sizes": {"float32": 4}}
    cost = MemoryAwareCostModel(config)

    node = IRNode("mul1", "Multiply", ["a", "b"], {}, shape_metadata=[1024])
    node.attributes["dtype"] = "float32"
    assert cost.is_fusion_valid({"mul1": node}) is False

    node2 = IRNode("mul2", "Multiply", ["a", "b"], {}, shape_metadata=[10])
    node2.attributes["dtype"] = "float32"
    assert cost.is_fusion_valid({"mul2": node2}) is True

    node3 = IRNode("mul3", "Multiply", ["a", "b"], {}, shape_metadata=["symbolic"])
    assert cost.is_fusion_valid({"mul3": node3}) is True

    cost2 = MemoryAwareCostModel(None)
    assert cost2.is_fusion_valid({"mul1": node}) is True


def test_commutative_pattern_matching() -> None:
    """Verify commutative matching behavior across direct, swapped, and mismatched operands."""
    graph = IRGraph()
    graph.nodes["in_a"] = IRNode(id="in_a", op_type="TypeA")
    graph.nodes["in_b"] = IRNode(id="in_b", op_type="TypeB")
    graph.nodes["add1"] = IRNode(id="add1", op_type="Add", inputs=["in_a", "in_b"])

    # Direct match: 1st branch of commutative check succeeds
    pat_direct = NodePattern(
        op_type="Add",
        inputs=[NodePattern(op_type="TypeA"), NodePattern(op_type="TypeB")],
        commutative=True,
    )
    assert match_pattern(graph, "add1", pat_direct, {}) is True

    # Swapped match: 1st branch fails, 2nd branch succeeds
    pat_swapped = NodePattern(
        op_type="Add",
        inputs=[NodePattern(op_type="TypeB"), NodePattern(op_type="TypeA")],
        commutative=True,
    )
    assert match_pattern(graph, "add1", pat_swapped, {}) is True

    # Neither matches
    pat_neither = NodePattern(
        op_type="Add",
        inputs=[NodePattern(op_type="TypeX"), NodePattern(op_type="TypeY")],
        commutative=True,
    )
    assert match_pattern(graph, "add1", pat_neither, {}) is False


def test_declarative_target_fusion_rule_and_discovery() -> None:
    """Verify rule application and discovery from YAML declarations."""
    rule = DeclarativeTargetFusionRule("test_rule", NodePattern(op_type="Relu"), "FusedRelu")
    g = IRGraph()
    assert rule.apply(g, {"root": "string_not_ir_node"}) is None

    node = IRNode(id="r1", op_type="Relu")
    res = rule.apply(g, {"root": node})
    assert res is not None
    assert "r1" in res
    assert res["r1"].op_type == "FusedRelu"

    mock_yaml_data = {
        "patterns": [
            {"name": "empty_target_pat", "target": []},
            {"name": "valid_target_pat", "target": ["Conv2D", "Relu"], "fused_op": "FusedConvRelu"},
        ]
    }
    with patch("ml_switcheroo_compiler.transforms.passes.operator_fusion.yaml.safe_load", return_value=mock_yaml_data):
        rules = _discover_fusion_patterns(patterns_dir=None)
        assert any(isinstance(r, DeclarativeTargetFusionRule) and r.name == "valid_target_pat" for r in rules)

    with patch("ml_switcheroo_compiler.transforms.passes.operator_fusion.yaml.safe_load", side_effect=ValueError("corrupt")):
        rules_exc = _discover_fusion_patterns(patterns_dir=None)
        assert rules_exc == []
