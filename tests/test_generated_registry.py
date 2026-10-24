"""Tests for test_generated_registry."""

from __future__ import annotations


def test_generated_registry() -> None:
    """Test function test_generated_registry."""
    from unittest.mock import mock_open, patch

    import ml_switcheroo_compiler.ops.generated_registry as gr

    assert "OPS_REGISTRY" in dir(gr)
    assert len(gr.OPS_REGISTRY) > 0

    with (
        patch("ml_switcheroo_compiler.ops.generated_registry.load_backend_mappings", return_value={}),
        patch("os.listdir", return_value=["single_alias.yaml", "single_no_alias.yaml", "multi.yaml"]),
        patch("builtins.open", mock_open()),
        patch(
            "yaml.load",
            side_effect=[
                {"operation": "OpWithAlias", "aliases": ["Alias1"]},
                {"operation": "OpWithoutAlias"},
                {"MultiOp": {}},
            ],
        ),
    ):
        built = gr._build_ops_registry()
        assert "OpWithAlias" in built
        assert "Alias1" in built
        assert "OpWithoutAlias" in built
        assert "MultiOp" in built
