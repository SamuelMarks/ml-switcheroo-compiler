"""Validation test suite asserting static grounding of backend mappings against framework snapshots."""

from __future__ import annotations

import glob
import os

import pytest
import yaml

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    GroundingValidationError,
    SnapshotGroundingEngine,
)
from scripts.validate_backend_mappings import (
    BUILTIN_OPS,
    extract_endpoints_from_ast,
    resolve_api_endpoint,
)


def test_all_backend_mappings_snapshot_grounded() -> None:
    """Assert that declared backend mapping YAML files resolve against framework snapshots."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends"))

    mapping_files: list[str] = glob.glob(os.path.join(base_dir, "**", "mappings", "*.yaml"), recursive=True)
    assert len(mapping_files) > 100, f"Expected >100 mapping files, found {len(mapping_files)}"

    unverified_endpoints: list[str] = []

    for filepath in mapping_files:
        rel_path: str = os.path.relpath(filepath, base_dir)
        backend_name: str = rel_path.split(os.sep)[0]
        if backend_name not in engine.config.targets:
            continue

        target = engine.config.targets[backend_name]
        if target.status == "missing_upstream":
            continue

        with open(filepath, encoding="utf-8") as f:
            data: dict[str, object] | None = yaml.safe_load(f)

        if not isinstance(data, dict):
            continue

        ops: dict[str, object] = data.get("operations", data)
        if not isinstance(ops, dict):
            continue

        for op_name, spec in ops.items():
            if not isinstance(spec, dict):
                continue

            target_api: str = str(spec.get("target_api") or "")
            if target_api and target_api != "custom_op" and not target_api.startswith("{"):
                clean: str = target_api.strip("'\"").split("(")[0].strip()
                if not resolve_api_endpoint(clean, backend_name, engine):
                    unverified_endpoints.append(f"{backend_name}::{op_name} -> {clean} ({filepath})")

            ast_template: str = str(spec.get("ast_template") or "")
            if ast_template and ast_template != "custom_op":
                calls = extract_endpoints_from_ast(ast_template, backend_name)
                for ep, _, _ in calls:
                    clean = ep.split("(")[0].strip()
                    if clean and clean not in BUILTIN_OPS and not clean.startswith("_"):
                        if not resolve_api_endpoint(clean, backend_name, engine):
                            unverified_endpoints.append(f"{backend_name}::{op_name} [template] -> {clean} ({filepath})")

    err_msg: str = "\n".join(unverified_endpoints[:20])
    assert not unverified_endpoints, f"Found {len(unverified_endpoints)} unverified endpoints:\n{err_msg}"


def test_validate_parameter_contract_positional_and_kwargs() -> None:
    """Test parameter positions, kinds, and keywords validation in SnapshotGroundingEngine."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # Valid call to torch.matmul(input, other)
    errors = engine.validate_parameter_contract("pytorch", "torch.matmul", 2, [])
    assert not errors

    # Valid call to torch.matmul with out keyword
    errors = engine.validate_parameter_contract("pytorch", "torch.matmul", 2, ["out"])
    assert not errors

    # Too many positional arguments to torch.matmul (max 2)
    errors = engine.validate_parameter_contract("pytorch", "torch.matmul", 5, [])
    assert len(errors) == 1
    assert "Too many positional arguments" in errors[0]

    # Unknown keyword argument to torch.matmul
    errors = engine.validate_parameter_contract("pytorch", "torch.matmul", 2, ["nonexistent_kwarg_123"])
    assert len(errors) == 1
    assert "Unknown keyword argument" in errors[0]


def test_validate_parameter_contract_discrepancy_detection() -> None:
    """Test detection of framework discrepancy pairs (dim vs axis)."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    # PyTorch sum expects 'dim', not 'axis'
    errors = engine.validate_parameter_contract("pytorch", "torch.sum", 1, ["axis"])
    assert any("does not accept keyword 'axis'. Did you mean 'dim'?" in e for e in errors)

    # NumPy sum expects 'axis', not 'dim'
    np_errors = engine.validate_parameter_contract("numpy", "numpy.sum", 1, ["dim"])
    assert any("does not accept keyword 'dim'. Did you mean 'axis'?" in e for e in np_errors)


def test_validate_parameter_contract_raise_on_error() -> None:
    """Test GroundingValidationError is raised when raise_on_error is True."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    with pytest.raises(GroundingValidationError) as exc_info:
        engine.validate_parameter_contract("pytorch", "torch.matmul", 10, ["fake_kw"], raise_on_error=True)

    assert "Too many positional arguments" in str(exc_info.value)
