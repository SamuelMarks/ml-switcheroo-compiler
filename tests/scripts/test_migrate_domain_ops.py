"""Tests for the migrate_domain_ops script."""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from unittest.mock import patch

import yaml

import scripts.migrate_domain_ops as mdo


def test_load_base_op(tmp_path: Path) -> None:
    """Verify _load_base_op across existing, non-dict, alias, and fallback paths.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    orig_base = mdo.BASE_DIR
    try:
        mdo.BASE_DIR = str(tmp_path)

        # 1. Existing op with dict data
        add_file = tmp_path / "Add.yaml"
        with open(add_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({"operation": "Add", "description": "Custom Add"}, f)
        res_add = mdo._load_base_op("Add")
        assert res_add["description"] == "Custom Add"

        # 2. Existing op with non-dict data
        bad_file = tmp_path / "bad.yaml"
        with open(bad_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(["not", "a", "dict"], f)
        res_bad = mdo._load_base_op("bad")
        assert res_bad["description"] == "Declarative bad operation."

        # 3. Non-existent op
        res_none = mdo._load_base_op("NonExistentOp")
        assert res_none["description"] == "Declarative NonExistentOp operation."

        # 4. Alias lookup via NAME_ALIASES
        norm_file = tmp_path / "Norm.yaml"
        with open(norm_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({"operation": "Norm", "description": "Norm base op"}, f)
        res_alias = mdo._load_base_op("NormL1")
        assert res_alias["description"] == "Norm base op"
    finally:
        mdo.BASE_DIR = orig_base


def test_migrate_domain_full_branches(tmp_path: Path) -> None:
    """Verify migrate_domain across binary, unary, creation, autodiff, and llvm_cpp branches.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    orig_base = mdo.BASE_DIR
    try:
        mdo.BASE_DIR = str(tmp_path)

        # Pre-seed an op with autodiff and existing llvm_cpp variant
        add_file = tmp_path / "Add.yaml"
        with open(add_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "operation": "Add",
                    "description": "Standard Add",
                    "autodiff": {"vjp": ["cot", "cot"]},
                    "variants": {"llvm_cpp": {"existing": True}},
                },
                f,
            )

        # Pre-seed an op with no existing llvm_cpp variant
        mul_file = tmp_path / "Mul.yaml"
        with open(mul_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({"operation": "Mul", "variants": {}}, f)

        # 1. Binary domain specification (covers Add, Mul, Sub, is_binary=True)
        spec_binary = {
            "domain": "binary",
            "ops": ["Add", "Mul", "Sub"],
            "inputs": ["x1", "x2"],
            "is_binary": True,
        }
        res_bin = mdo.migrate_domain("binary_test.yaml", spec_binary)
        assert "Add" in res_bin
        assert res_bin["Add"]["math_semantics"]["identity_element"] == 0
        assert res_bin["Mul"]["math_semantics"]["identity_element"] == 1
        assert "autodiff" in res_bin["Add"]
        assert res_bin["Add"]["signature"]["output_shape_formula"] == "broadcast(x1.shape, x2.shape)"
        assert res_bin["Add"]["signature"]["output_dtype_formula"] == "promote(x1.dtype, x2.dtype)"

        # 2. Unary domain specification (covers is_binary=False, default_inputs=['x'])
        spec_unary = {
            "domain": "unary",
            "ops": ["Abs"],
            "inputs": ["x"],
            "is_binary": False,
        }
        res_unary = mdo.migrate_domain("unary_test.yaml", spec_unary)
        assert "Abs" in res_unary
        assert res_unary["Abs"]["signature"]["output_shape_formula"] == "x.shape"
        assert res_unary["Abs"]["signature"]["output_dtype_formula"] == "x.dtype"

        # 3. Creation domain specification (covers default_inputs=[])
        spec_creation = {
            "domain": "creation",
            "ops": ["Zeros"],
            "inputs": [],
            "is_binary": False,
        }
        res_creation = mdo.migrate_domain("creation_test.yaml", spec_creation)
        assert "Zeros" in res_creation
        assert res_creation["Zeros"]["signature"]["output_shape_formula"] == "shape"
        assert res_creation["Zeros"]["signature"]["output_dtype_formula"] == "float32"

        # 4. Control flow domain (covers non-differentiable domain)
        spec_cf = {
            "domain": "control_flow",
            "ops": ["Cond"],
            "inputs": ["pred", "true_fn", "false_fn"],
            "is_binary": False,
        }
        res_cf = mdo.migrate_domain("cf_test.yaml", spec_cf)
        assert not res_cf["Cond"]["math_semantics"]["differentiable"]
    finally:
        mdo.BASE_DIR = orig_base


def test_main_and_runpy(tmp_path: Path) -> None:
    """Verify main execution and runpy invocation.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    orig_base = mdo.BASE_DIR
    try:
        mdo.BASE_DIR = str(tmp_path)
        with patch.dict(mdo.DOMAIN_SPECS, {"test.yaml": {"domain": "creation", "ops": ["Zeros"], "inputs": []}}, clear=True):
            with patch("builtins.print") as mock_print:
                mdo.main()
                mock_print.assert_called_with("Generated test.yaml with 1 operations.")

            script_path = os.path.abspath(mdo.__file__)
            runpy.run_path(script_path, run_name="__main__")
    finally:
        mdo.BASE_DIR = orig_base
