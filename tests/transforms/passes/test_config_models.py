from ml_switcheroo_compiler.transforms.passes.config_models import CostModelConfig, NodePatternConfig, PassConfig


def test_pass_config_model():
    data = {
        "execution_order": ["test_pass"],
        "cost_model": {"memory_costs": {"MatMul": 10}, "compute_costs": {"MatMul": 20}, "default_memory_cost": 1, "default_compute_cost": 2},
        "fusion_patterns": {"p1": {"pattern": {"op_type": "Add", "capture": "c"}, "replacement": {"op_type": "Sub", "inputs": ["c"], "capture_to_replace": "c"}}},
    }
    config = PassConfig(**data)
    assert config.execution_order == ["test_pass"]
    assert config.cost_model.memory_costs["MatMul"] == 10

    assert config.fusion_patterns["p1"].pattern.op_type == "Add"
    assert config.fusion_patterns["p1"].replacement.inputs == ["c"]

    # Coverage for optional fields
    n1 = NodePatternConfig()
    assert n1.op_type is None
    n2 = NodePatternConfig(op_type="A", capture="a", inputs=[])
    assert n2.inputs == []
    c1 = CostModelConfig(memory_costs={}, compute_costs={}, default_memory_cost=0, default_compute_cost=0)
    assert c1.default_compute_cost == 0
