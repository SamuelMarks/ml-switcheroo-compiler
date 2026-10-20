"""Tests for test_validate_rules."""

from __future__ import annotations

import os
import runpy
import sys
import tempfile
import typing
from unittest.mock import patch

import yaml

from scripts.validate_rules import main, validate_n_to_m

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_validate_n_to_m() -> None:
    """Test function test_validate_n_to_m."""
    with tempfile.TemporaryDirectory() as d:
        # Create fake backend directories
        be1_dir = os.path.join(d, "backends", "be1")
        be2_dir = os.path.join(d, "backends", "be2")
        os.makedirs(be1_dir)
        os.makedirs(be2_dir)

        # Valid op (supported by 2)
        # Invalid op (supported by 1)
        be1_data = {"backend_name": "be1", "operations": {"op_valid": {"target_api": "something"}, "op_invalid": {"custom_code": "something"}, "op_empty": {}}}

        be2_data = {"backend_name": "be2", "operations": {"op_valid": {"ast_template": "something"}}}

        with open(os.path.join(be1_dir, "mappings.yaml"), "w") as f:
            yaml.dump(be1_data, f)

        with open(os.path.join(be2_dir, "mappings.yaml"), "w") as f:
            yaml.dump(be2_data, f)

        single_op_file = os.path.join(be2_dir, "op_single.yaml")
        with open(single_op_file, "w") as f:
            yaml.dump({"operation": "op_valid", "backend": "be2", "target_api": "something"}, f)

        empty_single_op_file = os.path.join(be2_dir, "op_empty.yaml")
        with open(empty_single_op_file, "w") as f:
            yaml.dump({"operation": "op_empty", "backend": "be2"}, f)

        non_dict_file = os.path.join(be2_dir, "non_dict.yaml")
        with open(non_dict_file, "w") as f:
            yaml.dump(["not", "a", "dict"], f)

        non_dict_ops_file = os.path.join(be2_dir, "non_dict_ops.yaml")
        with open(non_dict_ops_file, "w") as f:
            yaml.dump({"backend_name": "be2", "operations": ["not_a_dict"]}, f)

        with patch(
            "glob.glob",
            side_effect=[
                [
                    os.path.join(be1_dir, "mappings.yaml"),
                    os.path.join(be2_dir, "mappings.yaml"),
                    single_op_file,
                    empty_single_op_file,
                    non_dict_file,
                    non_dict_ops_file,
                ],
                [],
                [],
            ],
        ):
            errors = validate_n_to_m()

        assert len(errors) == 1
        assert "op_invalid" in errors[0]
        assert "Rule 4" in errors[0]


def test_main(capsys: EagerTensorProtocol) -> None:
    """Test function test_main.

    Args:
        capsys: Test parameter `capsys`.
    """
    with patch("scripts.validate_rules.validate_n_to_m", return_value=["err1", "err2"]):
        assert main() == 1
        captured = capsys.readouterr()
        assert "Found 2 operations violating" in captured.err


def test_main_block(capsys: EagerTensorProtocol) -> None:
    """Test function test_main_block.

    Args:
        capsys: Test parameter `capsys`.
    """
    with patch.object(sys, "argv", ["validate_rules.py"]):
        with patch("glob.glob", return_value=[]):
            try:
                runpy.run_path("scripts/validate_rules.py", run_name="__main__")
            except SystemExit as e:
                assert e.code == 0
    captured = capsys.readouterr()
    assert "Found 0 operations violating" in captured.out
