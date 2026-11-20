"""Tests for generated operation registry completeness and lookup behavior."""

from __future__ import annotations

import os
from unittest.mock import mock_open, patch

import yaml

import ml_switcheroo_compiler.ops.generated_registry as gr
from ml_switcheroo_compiler.diagnostics.types_registry import is_non_math_type
from ml_switcheroo_compiler.ops import get_op
from tests.test_models import AllOpsManifest

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifests", "all_ops.yaml")
with open(_MANIFEST_PATH, encoding="utf-8") as _f:
    _manifest = AllOpsManifest(**yaml.safe_load(_f))
ALL_OPS: list[str] = _manifest.all_ops


def test_generated_registry_builders_and_aliases() -> None:
    """Verify `OPS_REGISTRY` population and `_build_ops_registry` alias/multi-op loading."""
    assert len(gr.OPS_REGISTRY) > 0
    assert "Add" in gr.OPS_REGISTRY

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


def test_all_manifest_math_ops_registered() -> None:
    """Verify that every mathematical operation in `all_ops.yaml` resolves via `get_op`."""
    missing: list[str] = []
    resolved_count = 0
    for op_name in ALL_OPS:
        if is_non_math_type(op_name):
            continue
        try:
            op_cls = get_op(op_name)
            if op_cls is not None:
                resolved_count += 1
        except KeyError:
            missing.append(op_name)
    assert not missing, f"Missing ops: {missing}"
    assert resolved_count > 0
