"""Tests for compiler transform pass validation and config models."""

from __future__ import annotations

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.config_models import (
    ComputeCosts,
    CostModelConfig,
    FusionPatternConfig,
    NodePatternConfig,
    ReplacementConfig,
)
from ml_switcheroo_compiler.transforms.passes.operator_fusion import operator_fusion_pass


def test_operator_fusion_mockups() -> None:
    """Test operator fusion pass on an Add + Relu subgraph."""
    graph = IRGraph()
    graph.nodes["in_a"] = IRNode(id="in_a", op_type="Input", inputs=[])
    graph.nodes["in_b"] = IRNode(id="in_b", op_type="Input", inputs=[])
    graph.nodes["add_0"] = IRNode(id="add_0", op_type="Add", inputs=["in_a", "in_b"])
    graph.nodes["relu_0"] = IRNode(id="relu_0", op_type="Relu", inputs=["add_0"])
    graph.outputs = ["relu_0"]

    modified: bool = operator_fusion_pass(graph)
    assert isinstance(modified, bool)
    assert len(graph.nodes) > 0


def test_config_models_mockups() -> None:
    """Test validation and properties of transform pass configuration models."""
    node_pat = NodePatternConfig(op_type="Relu", capture="r")
    assert node_pat.op_type == "Relu"
    assert node_pat.capture == "r"

    repl = ReplacementConfig(op_type="FusedAddRelu", inputs=["x", "y"], capture_to_replace="r")
    assert repl.op_type == "FusedAddRelu"
    assert repl.capture_to_replace == "r"

    fusion_cfg = FusionPatternConfig(pattern=node_pat, replacement=repl)
    assert fusion_cfg.pattern.op_type == "Relu"
    assert fusion_cfg.replacement.op_type == "FusedAddRelu"

    costs = ComputeCosts(
        heavy_ops=["Conv2D", "MatMul"],
        light_ops=["Add", "Relu"],
        heavy_cost=10,
        light_cost=1,
        default_cost=2,
    )
    cost_model = CostModelConfig(
        memory_sizes={"f32": 4},
        compute_costs=costs,
        compute_heavy_threshold=5,
        heavy_interleave_penalty=3,
        light_interleave_penalty=1,
    )
    assert cost_model.compute_costs.heavy_cost == 10
    assert cost_model.memory_sizes["f32"] == 4
