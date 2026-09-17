"""Tests for compiler transform pass configuration models and pipelines."""

import os

import yaml

from ml_switcheroo_compiler.transforms.passes.config_models import (
    ConvergenceCriteria,
    CostModelConfig,
    NodePatternConfig,
    PassConfig,
    PassConfigModel,
    PassPipelineConfig,
    PipelineStageModel,
    load_shape_learning_protocol,
)


def test_pass_config_model() -> None:
    """Verify PassConfig and cost model construction."""
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
    assert c1.compute_heavy_threshold == 1


def test_pass_pipeline_config_model() -> None:
    """Verify PassPipelineConfig, PipelineStageModel, and PassConfigModel validation."""
    criteria = ConvergenceCriteria(max_iterations=15, detect_cyclic_oscillation=True, require_fixpoint=True)
    assert criteria.max_iterations == 15

    stage = PipelineStageModel(
        stage_name="canonicalization",
        passes=["pass_a", "pass_b"],
        fixpoint_iteration=True,
        max_iterations=5,
    )
    assert stage.stage_name == "canonicalization"
    assert stage.fixpoint_iteration is True

    pass_cfg = PassConfigModel(
        pass_name="pass_a",
        enabled=True,
        prerequisites=[],
        preserves=["invariants"],
        options={"cluster_size": 4},
    )
    assert pass_cfg.pass_name == "pass_a"

    pipeline = PassPipelineConfig(
        execution_order=["pass_a", "pass_b"],
        convergence_criteria=criteria,
        prerequisites={"pass_b": ["pass_a"]},
        stages=[stage],
        pass_configs={"pass_a": pass_cfg},
    )
    assert pipeline.execution_order == ["pass_a", "pass_b"]
    assert pipeline.prerequisites["pass_b"] == ["pass_a"]
    assert len(pipeline.stages) == 1
    assert "pass_a" in pipeline.pass_configs


def test_load_bundled_pass_pipeline_yaml() -> None:
    """Verify that the bundled pass_pipeline.yaml validates cleanly against PassPipelineConfig."""
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "../../../src/ml_switcheroo_compiler/transforms/passes/pass_pipeline.yaml",
    )
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pipeline_cfg = PassPipelineConfig.model_validate(data)
    assert len(pipeline_cfg.stages) >= 3
    assert len(pipeline_cfg.execution_order) >= 10
    assert pipeline_cfg.convergence_criteria.require_fixpoint is True
    assert "constant_folding" in pipeline_cfg.pass_configs


def test_load_shape_learning_protocol() -> None:
    """Test load_shape_learning_protocol loader function."""
    cfg = load_shape_learning_protocol()
    assert cfg is not None

    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "../../../src/ml_switcheroo_compiler/transforms/passes/shape_learning_protocol.yaml",
    )
    cfg_custom = load_shape_learning_protocol(path=yaml_path)
    assert cfg_custom is not None


def test_device_mesh_partitioning_config_custom_path() -> None:
    """Test load_mesh_partitioning with explicit path argument."""
    import ml_switcheroo_compiler.transforms.passes.config_models as cm
    from ml_switcheroo_compiler.transforms.passes.config_models import load_mesh_partitioning

    real_path = os.path.join(os.path.dirname(cm.__file__), "spmd_mappings", "mesh_partitioning.yaml")
    cfg = load_mesh_partitioning(path=real_path)
    assert cfg is not None
