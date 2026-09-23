"""Unit tests for scripts/validate_parameters.py."""

from __future__ import annotations

import os
import runpy
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    SnapshotGroundingEngine,
)
from scripts.validate_parameters import (
    _extract_calls_from_ast,
    _normalize_code,
    main,
    validate_backend_mappings_parameters,
    validate_op_definitions_parameters,
)


def test_normalize_code() -> None:
    """Test normalizing template placeholders to valid python identifiers."""
    assert _normalize_code("  {x} + {0}  ") == "_var_x + _var_0"
    assert _normalize_code("torch.matmul({in0}, {in1})") == "torch.matmul(_var_in0, _var_in1)"


def test_extract_calls_from_ast() -> None:
    """Test extracting endpoints and argument structures from AST."""
    # Attribute call with chained name
    calls_attr: list[tuple[str, int, list[str]]] = _extract_calls_from_ast("torch.nn.functional.relu(x, inplace=True)")
    assert any(c[0] == "torch.nn.functional.relu" and c[1] == 1 and c[2] == ["inplace"] for c in calls_attr)

    # Simple name call
    calls_name: list[tuple[str, int, list[str]]] = _extract_calls_from_ast("abs(x)")
    assert any(c[0] == "abs" and c[1] == 1 and c[2] == [] for c in calls_name)

    # Complex attribute expression where root is not ast.Name
    calls_complex: list[tuple[str, int, list[str]]] = _extract_calls_from_ast("(a + b).dot(c)")
    assert any(c[0] == "dot" for c in calls_complex)

    # Call with non-attribute, non-name func (e.g. lambda call)
    calls_lambda: list[tuple[str, int, list[str]]] = _extract_calls_from_ast("(lambda y: y)(x)")
    assert calls_lambda == []

    # Call with placeholder / _var_ prefix
    calls_var: list[tuple[str, int, list[str]]] = _extract_calls_from_ast("{fn}(x)")
    assert calls_var == []

    # Call with **kwargs keyword without arg name
    calls_kwargs: list[tuple[str, int, list[str]]] = _extract_calls_from_ast("func(x, **kw)")
    assert any(c[0] == "func" and c[1] == 1 and c[2] == [] for c in calls_kwargs)

    # Syntax error fallback
    calls_syntax_err: list[tuple[str, int, list[str]]] = _extract_calls_from_ast("def invalid_syntax ::::")
    assert calls_syntax_err == []


def test_validate_backend_mappings_parameters_real() -> None:
    """Test validate_backend_mappings_parameters against the real repository mappings."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()
    errors: list[str] = validate_backend_mappings_parameters(engine)
    assert not errors

    # Filter for pytorch
    errors_pytorch: list[str] = validate_backend_mappings_parameters(engine, backend_filter="pytorch")
    assert not errors_pytorch


def test_validate_backend_mappings_parameters_synthetic() -> None:
    """Test branch coverage of validate_backend_mappings_parameters with synthetic YAML structures."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir: str = tmp_dir
        # Create directories
        torch_dir: str = os.path.join(base_dir, "pytorch", "mappings")
        other_dir: str = os.path.join(base_dir, "other", "mappings")
        os.makedirs(torch_dir)
        os.makedirs(other_dir)

        # 1. Non-dict file
        f_non_dict: str = os.path.join(torch_dir, "non_dict.yaml")
        with open(f_non_dict, "w", encoding="utf-8") as f:
            f.write("- item1\n- item2\n")

        # 2. Ops not a dict
        f_ops_not_dict: str = os.path.join(torch_dir, "ops_not_dict.yaml")
        with open(f_ops_not_dict, "w", encoding="utf-8") as f:
            f.write("operations: [1, 2, 3]\n")

        # 3. Spec not a dict, empty sources, invalid endpoint, and contract error
        f_mixed: str = os.path.join(torch_dir, "mixed.yaml")
        mixed_data: dict[str, Any] = {
            "operations": {
                "op_spec_not_dict": "invalid",
                "op_empty_src": {"ast_template": "", "custom_code": ""},
                "op_invalid_ep": {"ast_template": "torch.nonexistent_ep_xyz(a)"},
                "op_with_err": {"ast_template": "torch.matmul(a, b, c, d, e, f, g, h)"},
                "op_custom_err": {"custom_code": "torch.matmul(a, b, c, d, e, f, g, h)"},
            }
        }
        with open(f_mixed, "w", encoding="utf-8") as f:
            yaml.dump(mixed_data, f)

        # 4. Other backend for filter check
        f_other: str = os.path.join(other_dir, "other.yaml")
        with open(f_other, "w", encoding="utf-8") as f:
            yaml.dump({"operations": {}}, f)

        all_files: list[str] = [f_non_dict, f_ops_not_dict, f_mixed, f_other]
        real_relpath = os.path.relpath

        # Patch glob to return our synthetic files, and base_dir calculation
        with (
            patch("glob.glob", return_value=all_files),
            patch("scripts.validate_parameters.os.path.relpath", side_effect=lambda path, start: real_relpath(path, base_dir)),
        ):
            # Test with backend filter excluding other
            errors: list[str] = validate_backend_mappings_parameters(engine, backend_filter="pytorch")
            assert len(errors) == 2
            assert all("torch.matmul" in e for e in errors)

            # Test with backend not in engine targets
            mock_targets: dict[str, Any] = {"pytorch": MagicMock(status="missing_upstream")}
            with patch.object(engine.config, "targets", mock_targets):
                errors_missing: list[str] = validate_backend_mappings_parameters(engine)
                assert errors_missing == []

            # Test target entirely absent from config.targets
            with patch.object(engine.config, "targets", {}):
                errors_absent: list[str] = validate_backend_mappings_parameters(engine)
                assert errors_absent == []


def test_validate_op_definitions_parameters_real() -> None:
    """Test validate_op_definitions_parameters against the real repository definitions."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()
    errors: list[str] = validate_op_definitions_parameters(engine)
    assert not errors

    errors_pytorch: list[str] = validate_op_definitions_parameters(engine, backend_filter="pytorch")
    assert not errors_pytorch


def test_validate_op_definitions_parameters_synthetic() -> None:
    """Test branch coverage of validate_op_definitions_parameters with synthetic YAML definitions."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Non-dict file
        f_non_dict: str = os.path.join(tmp_dir, "non_dict.yaml")
        with open(f_non_dict, "w", encoding="utf-8") as f:
            f.write("- item1\n- item2\n")

        # 2. Variants not dict, and fallback to filename[:-5]
        f_no_variants: str = os.path.join(tmp_dir, "MyOp.yaml")
        with open(f_no_variants, "w", encoding="utf-8") as f:
            f.write("variants: not_a_dict\n")

        # 3. Mixed variants with invalid specs, empty generator, invalid ep, error
        f_mixed: str = os.path.join(tmp_dir, "MixedOp.yaml")
        mixed_data: dict[str, Any] = {
            "operation": "MixedOp",
            "variants": {
                "pytorch": {
                    "generator": "torch.matmul(a, b, c, d, e, f, g, h)",
                },
                "other_backend": {
                    "generator": "other.func(x)",
                },
                "empty_gen": {
                    "generator": "",
                },
                "invalid_ep": {
                    "generator": "torch.nonexistent_ep_xyz(a)",
                },
            },
        }
        with open(f_mixed, "w", encoding="utf-8") as f:
            yaml.dump(mixed_data, f)

        # 4. Target in engine config targets but var_spec is not a dict
        f_not_dict_spec: str = os.path.join(tmp_dir, "NotDictSpec.yaml")
        with open(f_not_dict_spec, "w", encoding="utf-8") as f:
            yaml.dump({"variants": {"pytorch": "invalid_spec_string"}}, f)

        all_files: list[str] = [f_non_dict, f_no_variants, f_mixed, f_not_dict_spec]

        with patch("glob.glob", return_value=all_files):
            # With backend_filter="pytorch"
            errors: list[str] = validate_op_definitions_parameters(engine, backend_filter="pytorch")
            assert len(errors) == 1
            assert "MixedOp.yaml [pytorch]" in errors[0]

            # With status missing_upstream
            mock_targets: dict[str, Any] = {"pytorch": MagicMock(status="missing_upstream")}
            with patch.object(engine.config, "targets", mock_targets):
                errors_missing: list[str] = validate_op_definitions_parameters(engine)
                assert errors_missing == []

            # With target missing from config
            with patch.object(engine.config, "targets", {}):
                errors_absent: list[str] = validate_op_definitions_parameters(engine)
                assert errors_absent == []


def test_main_success(capsys: pytest.CaptureFixture[str]) -> None:
    """Test main() execution when all checks pass."""
    with (
        patch("scripts.validate_parameters.validate_backend_mappings_parameters", return_value=[]),
        patch("scripts.validate_parameters.validate_op_definitions_parameters", return_value=[]),
        patch("sys.argv", ["validate_parameters.py", "--backend", "pytorch", "--strict"]),
    ):
        ret: int = main()
        assert ret == 0
        captured = capsys.readouterr()
        assert "Parameter validation PASSED" in captured.out


def test_main_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Test main() execution when parameter validation errors are found."""
    with (
        patch("scripts.validate_parameters.validate_backend_mappings_parameters", return_value=["mapping error 1"]),
        patch("scripts.validate_parameters.validate_op_definitions_parameters", return_value=["op error 1"]),
        patch("sys.argv", ["validate_parameters.py"]),
    ):
        ret: int = main()
        assert ret == 1
        captured = capsys.readouterr()
        assert "Parameter validation FAILED: 2 issues detected" in captured.out
        assert "mapping error 1" in captured.out
        assert "op error 1" in captured.out


def test_main_block() -> None:
    """Test execution of scripts.validate_parameters as __main__."""
    with (
        patch.object(sys, "argv", ["validate_parameters.py"]),
        patch("scripts.validate_parameters.validate_backend_mappings_parameters", return_value=[]),
        patch("scripts.validate_parameters.validate_op_definitions_parameters", return_value=[]),
        pytest.raises(SystemExit) as exc_info,
    ):
        runpy.run_module("scripts.validate_parameters", run_name="__main__")
    assert exc_info.value.code == 0
