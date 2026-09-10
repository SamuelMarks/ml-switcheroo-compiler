from ml_switcheroo_compiler.transforms.passes.config_models import (
    ConvergenceCriteria,
    CostModelConfig,
    NodePatternConfig,
    PassConfig,
    PassPipelineConfig,
)


def test_pass_config_model():
    data = {
        "execution_order": ["test_pass"],
        "cost_model": {"memory_sizes": {"MatMul": 10}, "compute_costs": {"heavy_ops": ["MatMul"], "light_ops": [], "heavy_cost": 20, "light_cost": 1, "default_cost": 2}, "compute_heavy_threshold": 1, "heavy_interleave_penalty": 1, "light_interleave_penalty": 1},
        "fusion_patterns": {"p1": {"pattern": {"op_type": "Add", "capture": "c"}, "replacement": {"op_type": "Sub", "inputs": ["c"], "capture_to_replace": "c"}}},
    }
    config = PassConfig(**data)
    assert config.execution_order == ["test_pass"]
    assert config.cost_model.memory_sizes["MatMul"] == 10

    assert config.fusion_patterns["p1"].pattern.op_type == "Add"
    assert config.fusion_patterns["p1"].replacement.inputs == ["c"]

    # Coverage for optional fields
    n1 = NodePatternConfig()
    assert n1.op_type is None
    n2 = NodePatternConfig(op_type="A", capture="a", inputs=[])
    assert n2.inputs == []
    c1 = CostModelConfig(memory_sizes={}, compute_costs={"heavy_ops": [], "light_ops": [], "heavy_cost": 1, "light_cost": 1, "default_cost": 1}, compute_heavy_threshold=1, heavy_interleave_penalty=1, light_interleave_penalty=1)


def test_pass_pipeline_config_model():
    criteria = ConvergenceCriteria(max_iterations=15, detect_cyclic_oscillation=True, require_fixpoint=True)
    assert criteria.max_iterations == 15
    pipeline = PassPipelineConfig(
        execution_order=["pass_a", "pass_b"],
        convergence_criteria=criteria,
        prerequisites={"pass_b": ["pass_a"]},
    )
    assert pipeline.execution_order == ["pass_a", "pass_b"]
    assert pipeline.prerequisites["pass_b"] == ["pass_a"]


def test_load_shape_learning_protocol():
    """Test load_shape_learning_protocol loader function."""
    import os

    from ml_switcheroo_compiler.transforms.passes.config_models import load_shape_learning_protocol

    cfg = load_shape_learning_protocol()
    assert cfg is not None

    yaml_path = os.path.join(os.path.dirname(__file__), "../../../src/ml_switcheroo_compiler/transforms/passes/shape_learning_protocol.yaml")
    cfg_custom = load_shape_learning_protocol(path=yaml_path)
    assert cfg_custom is not None
