"""Tests for scripts/validate_backend_mappings.py."""

import os
import tempfile
from unittest.mock import patch

import yaml

from scripts.validate_backend_mappings import main, resolve_api_endpoint, validate_mappings


def test_resolve_api_endpoint_success():
    """Test resolving a valid standard library or installed endpoint."""
    assert resolve_api_endpoint("numpy.add") is True
    assert resolve_api_endpoint("numpy.linalg.norm") is True


def test_resolve_api_endpoint_invalid_root():
    """Test resolving an endpoint with an unknown root module."""
    assert resolve_api_endpoint("unknown_lib.something") is False


def test_resolve_api_endpoint_missing_attr():
    """Test resolving an invalid endpoint attribute in a known root."""
    assert resolve_api_endpoint("numpy.nonexistent_attribute_xyz123") is False


def test_resolve_api_endpoint_import_exception():
    """Test fallback when module fails to import."""
    with patch("importlib.import_module", side_effect=ImportError("Module missing")):
        assert resolve_api_endpoint("cupy.add") is True


def test_validate_mappings_with_temp_files():
    """Test validate_mappings detecting valid, custom, lambda, and invalid endpoints."""
    with tempfile.TemporaryDirectory() as d:
        backend_dir = os.path.join(d, "numpy")
        mappings_dir = os.path.join(backend_dir, "mappings")
        os.makedirs(mappings_dir)

        mock_yaml = {
            "good_op": {"target_api": "numpy.add"},
            "custom_op": {"target_api": "custom_op"},
            "lambda_op": {"target_api": "lambda x: x"},
            "private_op": {"target_api": "_internal_helper"},
            "ast_template_valid": {"ast_template": "numpy.sum(a, b)"},
            "ast_template_empty": {"target_api": ""},
            "invalid_op": {"target_api": "numpy.hallucinated_xyz"},
            "cross_fw_op": {"target_api": "torch.relu"},
        }

        filepath = os.path.join(mappings_dir, "test_ops.yaml")
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(mock_yaml, f)

        # Also test a file with backend_name top-level key that gets skipped
        skip_file = os.path.join(mappings_dir, "skip.yaml")
        with open(skip_file, "w", encoding="utf-8") as f:
            yaml.dump({"backend_name": "numpy"}, f)

        # Also test a file with operations key
        ops_file = os.path.join(mappings_dir, "ops_wrapper.yaml")
        with open(ops_file, "w", encoding="utf-8") as f:
            yaml.dump({"operations": {"sub_op": {"target_api": "numpy.subtract"}}}, f)

        # Also non-dict yaml file
        non_dict_file = os.path.join(mappings_dir, "list.yaml")
        with open(non_dict_file, "w", encoding="utf-8") as f:
            yaml.dump(["item1", "item2"], f)

        orig_relpath = os.path.relpath

        def mock_relpath(path, start=None):
            return orig_relpath(path, d)

        with patch("glob.glob", side_effect=[[filepath, skip_file, ops_file, non_dict_file], [], []]), patch("os.path.relpath", side_effect=mock_relpath):
            errors = validate_mappings()

        err_str = " ".join(errors)
        assert "hallucinated_xyz" in err_str
        assert "cross-framework hallucination" in err_str
        assert "good_op" not in err_str
        assert "sub_op" not in err_str


def test_main(capsys):
    """Test main function handling both pass and failure cases."""
    with patch("scripts.validate_backend_mappings.validate_mappings", return_value=["err1"]):
        assert main() == 1
        assert "Hallucinations detected" in capsys.readouterr().out

    with patch("scripts.validate_backend_mappings.validate_mappings", return_value=[]):
        assert main() == 0
        assert "Passed" in capsys.readouterr().out


def test_main_block(capsys):
    """Test execution of script as __main__."""
    import runpy
    import sys

    with patch.object(sys, "argv", ["validate_backend_mappings.py"]):
        with patch("scripts.validate_backend_mappings.validate_mappings", return_value=[]):
            try:
                runpy.run_path("scripts/validate_backend_mappings.py", run_name="__main__")
            except SystemExit as e:
                assert e.code == 0
    assert "Passed" in capsys.readouterr().out
