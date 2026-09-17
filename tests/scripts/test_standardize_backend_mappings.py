"""Tests for the standardize_backend_mappings script."""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from unittest.mock import patch

import yaml

import scripts.standardize_backend_mappings as sbm


def test_build_default_ast_template() -> None:
    """Verify build_default_ast_template correctly branches across unary, binary, and general ops."""
    # Empty or custom_op target API
    assert sbm.build_default_ast_template("", "Add") == "{out} = {in0}"
    assert sbm.build_default_ast_template("custom_op", "Mul") == "{out} = {in0}"

    # Unary operations
    assert sbm.build_default_ast_template("np.abs", "Abs") == "{out} = np.abs({in0})"
    assert sbm.build_default_ast_template("np.sin", "Sin") == "{out} = np.sin({in0})"

    # Binary operations
    assert sbm.build_default_ast_template("np.add", "Add") == "{out} = np.add({in0}, {in1})"
    assert sbm.build_default_ast_template("np.matmul", "Matmul") == "{out} = np.matmul({in0}, {in1})"

    # Other/general operations
    assert sbm.build_default_ast_template("np.concatenate", "Concat") == "{out} = np.concatenate({inputs})"


def test_standardize_mapping_data() -> None:
    """Verify standardize_mapping_data parses kwargs, dtypes, templates, and schema validation."""
    # Existing template, kwarg_translations, None values in kwargs
    raw_spec = {
        "target_api": "np.add",
        "ast_template": "{out} = np.add({in0}, {in1})",
        "kwarg_translations": {"axis": "dim", "drop": None},
        "dtype_overrides": {"float64": "float32"},
    }
    result = sbm.standardize_mapping_data("numpy", "Add", raw_spec)
    assert result["operation"] == "Add"
    assert result["backend"] == "numpy"
    assert result["target_api"] == "np.add"
    assert result["ast_template"] == "{out} = np.add({in0}, {in1})"
    assert result["kwarg_map"] == {"axis": "dim", "drop": None}
    assert result["dtype_overrides"] == {"float64": "float32"}

    # Non-dict kwarg_map and non-dict dtype_overrides
    raw_non_dict = {
        "target_api": "np.abs",
        "kwarg_map": ["invalid_list"],
        "dtype_overrides": ["invalid_list"],
    }
    result_non_dict = sbm.standardize_mapping_data("numpy", "Abs", raw_non_dict)
    assert result_non_dict["kwarg_map"] == {}
    assert result_non_dict["dtype_overrides"] == {}


def test_run_standardization_and_main(tmp_path: Path) -> None:
    """Verify run_standardization across YAML layouts and file structures.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    backends_dir = tmp_path / "backends"
    numpy_map_dir = backends_dir / "numpy" / "mappings"
    numpy_map_dir.mkdir(parents=True)

    # 1. File with 'operation' key
    f1 = numpy_map_dir / "Add.yaml"
    with open(f1, "w", encoding="utf-8") as f:
        yaml.safe_dump({"operation": "Add", "target_api": "np.add"}, f)

    # 2. File with filename matching top-level key
    f2 = numpy_map_dir / "Mul.yaml"
    with open(f2, "w", encoding="utf-8") as f:
        yaml.safe_dump({"Mul": {"target_api": "np.multiply"}}, f)

    # 3. File with different top-level dict key
    f3 = numpy_map_dir / "Other.yaml"
    with open(f3, "w", encoding="utf-8") as f:
        yaml.safe_dump({"Custom": {"target_api": "np.custom"}}, f)

    # 4. File with flat dict where first key's value is not a dict
    f4 = numpy_map_dir / "Flat.yaml"
    with open(f4, "w", encoding="utf-8") as f:
        yaml.safe_dump({"target_api": "np.flat"}, f)

    # 5. File with non-dict content (should be skipped)
    f5 = numpy_map_dir / "Invalid.yaml"
    with open(f5, "w", encoding="utf-8") as f:
        yaml.safe_dump(["not", "a", "dict"], f)

    orig_dir = sbm.BACKENDS_DIR
    try:
        sbm.BACKENDS_DIR = str(backends_dir)
        count = sbm.run_standardization()
        assert count == 4

        with patch("builtins.print") as mock_print:
            sbm.main()
            mock_print.assert_called_with("Successfully standardized 4 backend mapping YAML files.")
    finally:
        sbm.BACKENDS_DIR = orig_dir


def test_runpy_entrypoint() -> None:
    """Verify script executes as main via runpy."""
    script_path = os.path.abspath(sbm.__file__)
    with patch("scripts.standardize_backend_mappings.run_standardization", return_value=0):
        runpy.run_path(script_path, run_name="__main__")
