"""Tests for autodiff configuration and manifest Pydantic models."""

from __future__ import annotations

import os

import yaml

from ml_switcheroo_compiler.grad.config_models import (
    AutodiffRuleModel,
    AutodiffRulesManifestModel,
    FiniteDifferenceConfig,
    FiniteDifferenceDtypeConfig,
    HigherOrderAutodiffModel,
)


def test_finite_difference_config() -> None:
    """Test finite difference dtype configuration models."""
    f32_cfg = FiniteDifferenceDtypeConfig(epsilon=1e-5)
    f64_cfg = FiniteDifferenceDtypeConfig(epsilon=1e-9)
    cfg = FiniteDifferenceConfig(float32=f32_cfg, float64=f64_cfg)
    assert cfg.float32.epsilon == 1e-5
    assert cfg.float64.epsilon == 1e-9


def test_autodiff_rule_model() -> None:
    """Test AutodiffRuleModel instantiation and field defaults."""
    rule = AutodiffRuleModel(
        opcode="Add",
        vjp=["$cotangent", "$cotangent"],
        jvp="Add($tangent[0], $tangent[1])",
        description="Elementwise addition gradient rule.",
    )
    assert rule.opcode == "Add"
    assert rule.cotangent_inputs == ["$cotangent"]
    assert rule.primal_outputs == ["$output"]
    assert rule.vjp == ["$cotangent", "$cotangent"]
    assert rule.jvp == "Add($tangent[0], $tangent[1])"
    assert rule.description == "Elementwise addition gradient rule."


def test_higher_order_autodiff_model() -> None:
    """Test HigherOrderAutodiffModel configuration and fields."""
    hvp_rule = HigherOrderAutodiffModel(
        opcode="Exp",
        hvp="Mul($tangent[0], Mul($cotangent, Exp($input[0])))",
        jvp_order=2,
        vjp_order=2,
        rewrite_rules=[{"pattern": "Grad(Exp($x))", "replacement": "Exp($x)"}],
    )
    assert hvp_rule.opcode == "Exp"
    assert hvp_rule.hvp is not None
    assert hvp_rule.jvp_order == 2
    assert len(hvp_rule.rewrite_rules) == 1


def test_autodiff_rules_manifest_validation() -> None:
    """Validate transforms/autodiff_rules.yaml against AutodiffRulesManifestModel."""
    rules_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "ml_switcheroo_compiler",
            "transforms",
            "autodiff_rules.yaml",
        )
    )
    assert os.path.exists(rules_path)

    with open(rules_path, encoding="utf-8") as f:
        raw_manifest = yaml.safe_load(f)

    manifest = AutodiffRulesManifestModel.model_validate(raw_manifest)
    assert len(manifest.jvp_rules) > 0
    assert len(manifest.vjp_rules) > 0
    assert len(manifest.higher_order_rules) > 0
    assert "Add" in manifest.jvp_rules
    assert "Add" in manifest.vjp_rules
    assert "Add" in manifest.higher_order_rules
