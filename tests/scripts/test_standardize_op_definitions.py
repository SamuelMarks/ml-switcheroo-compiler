"""Tests for the standardize_op_definitions script."""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from unittest.mock import patch

import yaml

import scripts.standardize_op_definitions as sod


def test_build_signature_section() -> None:
    """Verify build_signature_section with variadic, non-tensor, and empty arguments."""
    # 1. Variadic and tensor/array args
    sig1 = sod.build_signature_section(
        "Concat",
        {
            "std_args": [
                {"name": "tensors", "type": "list[tensor]", "is_variadic": True},
                {"name": "axis", "type": "int", "is_variadic": False},
            ]
        },
    )
    assert sig1["has_variadic_args"] is True
    assert sig1["input_tensors"] == ["tensors"]

    # 2. Non-tensor args fallback to arg names
    sig2 = sod.build_signature_section(
        "Custom",
        {"std_args": [{"name": "dim", "type": "int", "is_variadic": False}]},
    )
    assert sig2["input_tensors"] == ["dim"]

    # 3. Empty args fallback to ['x']
    sig3 = sod.build_signature_section("EmptyOp", {})
    assert sig3["input_tensors"] == ["x"]


def test_build_dtype_rules_section() -> None:
    """Verify build_dtype_rules_section across boolean, bitwise, cast, and standard ops."""
    # Boolean comparison ops
    d_bool = sod.build_dtype_rules_section("Equal", {})
    assert d_bool["promotion_matrix"] == "boolean"
    assert d_bool["output_dtype_inference"] == "bool"

    # Bitwise ops
    d_bitwise = sod.build_dtype_rules_section("BitwiseAnd", {})
    assert d_bitwise["promotion_matrix"] == "integer"
    assert d_bitwise["output_dtype_inference"] == "in0"

    # Cast ops
    d_cast = sod.build_dtype_rules_section("CastToFloat", {})
    assert d_cast["promotion_matrix"] == "cast"
    assert d_cast["output_dtype_inference"] == "target_dtype"

    # Standard binary ops (> 1 arg)
    d_bin = sod.build_dtype_rules_section("Add", {"std_args": [{"name": "x"}, {"name": "y"}]})
    assert d_bin["promotion_matrix"] == "standard"
    assert d_bin["output_dtype_inference"] == "promote(in0, in1)"

    # Standard unary ops (<= 1 arg)
    d_un = sod.build_dtype_rules_section("Abs", {"std_args": [{"name": "x"}]})
    assert d_un["promotion_matrix"] == "standard"
    assert d_un["output_dtype_inference"] == "in0"


def test_build_shape_signature_section() -> None:
    """Verify build_shape_signature_section across known signatures, matmul, conv, commutative, and unary."""
    known_sigs = {"KnownOp": {"pattern": "broadcast(in0, in1)", "category": "elementwise"}}

    # Known signature
    s_known = sod.build_shape_signature_section("KnownOp", known_sigs)
    assert s_known["pattern"] == "broadcast(in0, in1)"
    assert s_known["category"] == "elementwise"

    # Matmul ops
    s_matmul = sod.build_shape_signature_section("MatMul", {})
    assert s_matmul["pattern"] == "matmul(in0, in1)"
    assert s_matmul["category"] == "linalg"
    assert s_matmul["symbolic_expression"] == "[B, M, K], [B, K, N] -> [B, M, N]"

    # Conv ops
    s_conv = sod.build_shape_signature_section("Conv2D", {})
    assert s_conv["pattern"] == "conv(in0, in1)"
    assert s_conv["category"] == "nn"

    # Commutative fallback
    s_comm = sod.build_shape_signature_section("Multiply", {})
    assert s_comm["pattern"] == "broadcast(in0, in1)"
    assert s_comm["category"] == "elementwise"

    # Unary/other fallback
    s_other = sod.build_shape_signature_section("UnknownOp", {})
    assert s_other["pattern"] == "in0"
    assert s_other["category"] == "unary"


def test_build_invariants_and_autodiff_section() -> None:
    """Verify build_invariants_section and build_autodiff_section parsing."""
    # Invariants
    inv_add = sod.build_invariants_section("Add")
    assert inv_add["is_commutative"] is True
    assert inv_add["identity_value"] == 0

    inv_mul = sod.build_invariants_section("Mul")
    assert inv_mul["absorbing_value"] == 0

    # Autodiff with nested dict
    ad_dict = sod.build_autodiff_section("Add", {"autodiff": {"vjp": ["cot", "cot"], "jvp": "tan"}})
    assert ad_dict["vjp"] == ["cot", "cot"]
    assert ad_dict["jvp"] == "tan"

    # Autodiff with flat keys
    ad_flat = sod.build_autodiff_section("Sub", {"vjp": ["cot", "-cot"], "jvp": "tan"})
    assert ad_flat["vjp"] == ["cot", "-cot"]

    # Autodiff empty
    ad_empty = sod.build_autodiff_section("Custom", {})
    assert ad_empty["vjp"] is None
    assert ad_empty["jvp"] is None


def test_run_standardization_and_main(tmp_path: Path) -> None:
    """Verify run_standardization across single-op, multi-op, and invalid YAML files.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    orig_defs = sod.DEFS_DIR
    try:
        sod.DEFS_DIR = str(tmp_path)

        # 1. Single-op definition file
        op1 = {
            "operation": "Add",
            "description": "Add op",
            "std_args": [{"name": "x", "type": "tensor"}, {"name": "y", "type": "tensor"}],
        }
        with open(tmp_path / "Add.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(op1, f)

        # 2. Multi-op domain file
        multi = {
            "Abs": {
                "operation": "Abs",
                "description": "Abs op",
                "std_args": [{"name": "x", "type": "tensor"}],
            },
            "invalid_node": "not_a_dict",
        }
        with open(tmp_path / "unary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(multi, f)

        # 3. Non-dict YAML (should be skipped)
        with open(tmp_path / "list.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(["not", "a", "dict"], f)

        # 4. Non-YAML file (should be ignored)
        with open(tmp_path / "ignore.txt", "w", encoding="utf-8") as f:
            f.write("text")

        count = sod.run_standardization()
        assert count == 2

        with patch("builtins.print") as mock_print:
            sod.main()
            mock_print.assert_called_with("Successfully standardized 2 operation YAML definition files.")

        script_path = os.path.abspath(sod.__file__)
        with patch("scripts.standardize_op_definitions.run_standardization", return_value=0):
            runpy.run_path(script_path, run_name="__main__")
    finally:
        sod.DEFS_DIR = orig_defs
