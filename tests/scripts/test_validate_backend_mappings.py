"""Tests for scripts/validate_backend_mappings.py."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from ml_switcheroo_compiler.backends.snapshot_grounding import SnapshotGroundingEngine
from scripts.validate_backend_mappings import (
    _extract_op_api,
    _extract_ops_from_mapping_file,
    extract_endpoints_from_ast,
    main,
    normalize_template,
    resolve_api_endpoint,
    validate_arguments_against_snapshot,
    validate_mappings,
)


def test_normalize_template() -> None:
    """Test normalizing template expression strings."""
    assert normalize_template("{x} + {y}") == "_var_x + _var_y"
    assert normalize_template("torch.matmul({0}, {1})") == "torch.matmul(_var_0, _var_1)"


def test_extract_endpoints_from_ast() -> None:
    """Test extracting endpoints, binops, and kwargs from AST."""
    calls = extract_endpoints_from_ast("torch.matmul({0}, {1})")
    assert any(c[0] == "torch.matmul" for c in calls)

    binop_calls = extract_endpoints_from_ast("{x} + {y}")
    assert any(c[0] == "add" for c in binop_calls)

    module_calls = extract_endpoints_from_ast("backend_module.maximum({0}, 0)", backend_name="cupy")
    assert any(c[0] == "cupy.maximum" for c in module_calls)

    kwarg_calls = extract_endpoints_from_ast("kwargs.get('axis', -1)")
    assert any(c[0] == "get" for c in kwarg_calls)

    # Test hasattr branches
    extract_endpoints_from_ast("hasattr(x)")
    extract_endpoints_from_ast("hasattr(x, y)")
    extract_endpoints_from_ast("hasattr(x, 123)")

    # Test syntax error fallback
    assert extract_endpoints_from_ast("def invalid syntax ::::") == []


def test_validate_arguments_against_snapshot() -> None:
    """Test parameter and argument validation against snapshots."""
    engine = SnapshotGroundingEngine()

    # Empty or missing params returns no errors
    assert validate_arguments_against_snapshot("pytorch", "torch.completely_fake_xyz", ["a"], [], engine) == []

    # Valid call to torch.matmul
    errs = validate_arguments_against_snapshot("pytorch", "torch.matmul", ["_var_0", "_var_1"], [], engine)
    assert errs == []

    # Too many positional args
    errs_too_many = validate_arguments_against_snapshot("pytorch", "torch.matmul", ["a", "b", "c", "d", "e", "f"], [], engine)
    assert any("too many positional arguments" in e for e in errs_too_many)

    # Unknown keyword argument
    errs_kw = validate_arguments_against_snapshot("pytorch", "torch.matmul", ["a", "b"], ["invalid_kwarg_12345"], engine)
    assert any("unknown keyword argument" in e for e in errs_kw)


def test_resolve_api_endpoint_success() -> None:
    """Test resolving a valid standard library or installed endpoint."""
    assert resolve_api_endpoint("numpy.add") is True
    assert resolve_api_endpoint("numpy.linalg.norm") is True
    assert resolve_api_endpoint("getattr") is True
    assert resolve_api_endpoint("da.abs", "dask") is True
    assert resolve_api_endpoint("jnp.sin", "jax") is True


def test_resolve_api_endpoint_invalid_root() -> None:
    """Test resolving an endpoint with an unknown root module."""
    assert resolve_api_endpoint("unknown_lib.something") is False


def test_resolve_api_endpoint_missing_attr() -> None:
    """Test resolving an invalid endpoint attribute in a known root."""
    assert resolve_api_endpoint("numpy.nonexistent_attribute_xyz123") is False


def test_resolve_api_endpoint_submodule_traversal() -> None:
    """Test submodule import traversal during static inspection."""
    assert resolve_api_endpoint("torch.nn.functional.relu", "pytorch") is True
    assert resolve_api_endpoint("torch.nn.invalid_submod.dummy", "pytorch") is False


def test_resolve_api_endpoint_internal_helper() -> None:
    """Test resolving internal helper starting with underscore."""
    assert resolve_api_endpoint("_torch_variance", "pytorch") is True
    assert resolve_api_endpoint("_completely_fake_helper_xyz", "pytorch") is False


def test_resolve_api_endpoint_nonexistent_snapshot() -> None:
    """Test endpoint validation when snapshot directory is invalid."""
    engine: SnapshotGroundingEngine = SnapshotGroundingEngine(snapshot_dir="/non_existent")
    assert resolve_api_endpoint("torch.add", "pytorch", engine) is False


def test_extract_op_api() -> None:
    """Test extracting API endpoint from op specs."""
    assert _extract_op_api({"target_api": "numpy.sin"}) == "numpy.sin"
    assert _extract_op_api({"target_api": "", "ast_template": "numpy.cos(x)"}) == "numpy.cos"
    assert _extract_op_api({"target_api": "custom_op", "ast_template": "numpy.tan(x)"}) == "numpy.tan"
    assert _extract_op_api({"target_api": "", "ast_template": "::: invalid non-matching"}) == ""
    assert _extract_op_api({}) == ""


def test_extract_ops_from_mapping_file() -> None:
    """Test extracting operator dictionary from raw YAML mapping data."""
    assert _extract_ops_from_mapping_file({"backend_name": "numpy"}) == {}
    assert _extract_ops_from_mapping_file({"operations": {"op1": {"target_api": "a"}}}) == {"op1": {"target_api": "a"}}
    assert _extract_ops_from_mapping_file({"op2": {"target_api": "b"}, "invalid": 123}) == {"op2": {"target_api": "b"}}


def test_resolve_api_endpoint_expansion_fallback() -> None:
    """Test canonical expansion fallback when shorthand endpoint is not directly in snapshot."""
    from unittest.mock import MagicMock

    mock_engine = MagicMock(spec=SnapshotGroundingEngine)
    mock_engine.is_endpoint_valid.side_effect = lambda b, ep: ep == "dask.array.abs"
    assert resolve_api_endpoint("da.abs", "dask", mock_engine) is True

    # When expanded endpoint is also not valid
    mock_engine.is_endpoint_valid.return_value = False
    assert resolve_api_endpoint("da.nonexistent", "dask", mock_engine) is False


def test_validate_mappings_with_temp_files() -> None:
    """Test validate_mappings detecting valid, custom, lambda, and invalid endpoints."""
    with tempfile.TemporaryDirectory() as d:
        backend_dir: str = os.path.join(d, "numpy")
        mappings_dir: str = os.path.join(backend_dir, "mappings")
        os.makedirs(mappings_dir)

        pytorch_dir: str = os.path.join(d, "pytorch")
        pytorch_mappings: str = os.path.join(pytorch_dir, "mappings")
        os.makedirs(pytorch_mappings)

        mock_yaml: dict[str, object] = {
            "good_op": {"target_api": "numpy.add"},
            "custom_op": {"target_api": "custom_op"},
            "lambda_op": {"target_api": "lambda x: x"},
            "private_op": {"target_api": "_internal_helper"},
            "template_op": {"target_api": "{0}.abs"},
            "builtin_op": {"target_api": "getattr"},
            "ast_template_valid": {"ast_template": "numpy.sum(a, b)"},
            "ast_template_empty": {"target_api": ""},
            "invalid_op": {"target_api": "numpy.hallucinated_xyz"},
            "cross_fw_op": {"target_api": "torch.relu"},
            "template_cross_fw": {"ast_template": "da.abs({0})"},
            "template_prefix": {"ast_template": "numpy_prefix_helper({0})"},
            "template_np_prefix": {"ast_template": "np_helper({0})"},
            "template_unverified_call": {"ast_template": "numpy.fake_call_123({0})"},
            "direct_placeholder": {"target_api": "{0}"},
        }

        filepath: str = os.path.join(mappings_dir, "test_ops.yaml")
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(mock_yaml, f)

        py_yaml: dict[str, object] = {
            "bad_arg_op": {"ast_template": "torch.matmul(a, b, c, d, e, f, g)"},
        }
        py_filepath: str = os.path.join(pytorch_mappings, "py_ops.yaml")
        with open(py_filepath, "w", encoding="utf-8") as f:
            yaml.dump(py_yaml, f)

        skip_file: str = os.path.join(mappings_dir, "skip.yaml")
        with open(skip_file, "w", encoding="utf-8") as f:
            yaml.dump({"backend_name": "numpy"}, f)

        ops_file: str = os.path.join(mappings_dir, "ops_wrapper.yaml")
        with open(ops_file, "w", encoding="utf-8") as f:
            yaml.dump({"operations": {"sub_op": {"target_api": "numpy.subtract"}}}, f)

        non_dict_file: str = os.path.join(mappings_dir, "list.yaml")
        with open(non_dict_file, "w", encoding="utf-8") as f:
            yaml.dump(["item1", "item2"], f)

        orig_relpath = os.path.relpath

        def mock_relpath(path: str, start: object = None) -> str:
            return orig_relpath(path, d)

        with patch("glob.glob", side_effect=[[filepath, py_filepath, skip_file, ops_file, non_dict_file], [], []]), patch("os.path.relpath", side_effect=mock_relpath):
            errors = validate_mappings()

        err_str: str = " ".join(errors)
        assert "hallucinated_xyz" in err_str
        assert "cross-framework hallucination" in err_str
        assert "argument error" in err_str
        assert "good_op" not in err_str
        assert "sub_op" not in err_str


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Test main function handling both pass and failure cases.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest capture fixture.
    """
    with patch("scripts.validate_backend_mappings.validate_mappings", return_value=["err1"]):
        assert main() == 1
        assert "Hallucinations detected" in capsys.readouterr().out

    with patch("scripts.validate_backend_mappings.validate_mappings", return_value=[]):
        assert main() == 0
        assert "Passed" in capsys.readouterr().out


def test_load_yaml_config_fallback_and_unmapped_binop() -> None:
    """Test _load_yaml_config fallback branches and BinOp with unmapped operator."""
    from scripts.validate_backend_mappings import _load_yaml_config

    # 1. File does not exist (branch 22->27)
    res_nonexistent = _load_yaml_config("nonexistent_config_file_123.yaml")
    assert res_nonexistent == {}

    # 2. File exists but safe_load returns non-dict (branch 25->27)
    with patch("yaml.safe_load", return_value=["not", "a", "dict"]):
        res_nondict = _load_yaml_config("grounding_rules.yaml")
        assert res_nondict == {}

    # 3. BinOp with operator not in op_map, e.g. MatMult `@` (branch 188->142)
    endpoints = extract_endpoints_from_ast("a @ b")
    assert endpoints == []


def test_main_block(capsys: pytest.CaptureFixture[str]) -> None:
    """Test execution of script as __main__.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest capture fixture.
    """
    import runpy
    import sys

    with patch.object(sys, "argv", ["validate_backend_mappings.py"]):
        try:
            runpy.run_module("scripts.validate_backend_mappings", run_name="__main__")
        except SystemExit as e:
            assert e.code == 0
    captured: str = capsys.readouterr().out
    assert "Passed" in captured or "Validation" in captured
