"""Validation test suite asserting static grounding of backend mappings against framework snapshots."""

from __future__ import annotations

import glob
import os
import typing

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

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def _check_op_spec_grounding(
    backend_name: str,
    op_name: str,
    spec: dict[str, EagerValue],
    filepath: str,
    engine: SnapshotGroundingEngine,
) -> list[str]:
    """Validate target_api and ast_template in a single operation mapping spec.

    Args:
        backend_name: Target backend name.
        op_name: Canonical operation name.
        spec: Operation mapping specification dictionary.
        filepath: Source YAML mapping file path.
        engine: Snapshot grounding engine instance.

    Returns:
        List of unverified endpoint error strings.
    """
    unverified: list[str] = []
    target_api: str = str(spec.get("target_api") or "")
    if target_api and target_api != "custom_op" and not target_api.startswith("{"):
        clean: str = target_api.strip("'\"").split("(")[0].strip()
        if not resolve_api_endpoint(clean, backend_name, engine):
            unverified.append(f"{backend_name}::{op_name} -> {clean} ({filepath})")

    ast_template: str = str(spec.get("ast_template") or "")
    if ast_template and ast_template != "custom_op":
        calls = extract_endpoints_from_ast(ast_template, backend_name)
        for ep, _, _ in calls:
            clean = ep.split("(")[0].strip()
            if clean and clean not in BUILTIN_OPS and not clean.startswith("_"):
                if not resolve_api_endpoint(clean, backend_name, engine):
                    unverified.append(f"{backend_name}::{op_name} [template] -> {clean} ({filepath})")
    return unverified


def _check_mapping_file_grounding(
    filepath: str,
    base_dir: str,
    engine: SnapshotGroundingEngine,
) -> list[str]:
    """Validate all operations inside a single backend mapping YAML file.

    Args:
        filepath: Path to the YAML mapping file.
        base_dir: Root backends directory path.
        engine: Snapshot grounding engine instance.

    Returns:
        List of unverified endpoint error strings.
    """
    rel_path: str = os.path.relpath(filepath, base_dir)
    backend_name: str = rel_path.split(os.sep)[0]
    if backend_name not in engine.config.targets:
        return []
    if engine.config.targets[backend_name].status == "missing_upstream":
        return []

    with open(filepath, encoding="utf-8") as f:
        data: dict[str, EagerValue] | None = yaml.safe_load(f)
    if not isinstance(data, dict):
        return []

    ops: EagerValue = data.get("operations", data)
    if not isinstance(ops, dict):
        return []

    unverified: list[str] = []
    for op_name, spec in ops.items():
        if isinstance(spec, dict):
            unverified.extend(_check_op_spec_grounding(backend_name, str(op_name), spec, filepath, engine))
    return unverified


def test_all_backend_mappings_snapshot_grounded() -> None:
    """Assert that declared backend mapping YAML files resolve against framework snapshots."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends"))

    mapping_files: list[str] = glob.glob(os.path.join(base_dir, "**", "mappings", "*.yaml"), recursive=True)
    assert len(mapping_files) > 100, f"Expected >100 mapping files, found {len(mapping_files)}"

    unverified_endpoints: list[str] = []
    for filepath in mapping_files:
        unverified_endpoints.extend(_check_mapping_file_grounding(filepath, base_dir, engine))

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


def test_all_backend_mappings_zero_parameter_hallucinations() -> None:
    """Validate that 100% of all backend mapping files have zero parameter hallucinations."""
    from scripts.validate_parameters import validate_backend_mappings_parameters

    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()
    errors: list[str] = validate_backend_mappings_parameters(engine)
    assert not errors, "Detected parameter hallucinations in backend mappings:\n" + "\n".join(errors)
