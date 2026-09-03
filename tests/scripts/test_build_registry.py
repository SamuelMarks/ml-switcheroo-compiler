"""Tests for the build_registry script."""

import runpy
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import yaml

import scripts.build_registry as br


def test_build_registry_multi_op(tmp_path: Path) -> None:
    """Test building registry with various file types and structures.

    Args:
        tmp_path (Path): Pytest fixture for temporary directory.
    """
    def_dir = tmp_path / "definitions"
    def_dir.mkdir()
    out_file = tmp_path / "generated_registry.py"

    # 1. Normal multi-op dict
    data_multi: dict[str, dict[str, str]] = {"DummyOp1": {"signature": "(x) -> x"}, "DummyOp2": {"signature": "(y) -> y"}}
    with open(def_dir / "a_multi.yaml", "w") as f:
        yaml.dump(data_multi, f)

    # 2. Single-op file with "operation" key
    data_single = {"operation": "DummyOp3", "signature": "(z) -> z"}
    with open(def_dir / "b_single.yaml", "w") as f:
        yaml.dump(data_single, f)

    # 3. File with non-dict inner data to test branch `if isinstance(op_info, dict):`
    data_invalid = {"DummyOp4": "not a dict"}
    with open(def_dir / "c_invalid.yaml", "w") as f:
        yaml.dump(data_invalid, f)

    # 4. Non-yaml file to test branch `if filename.endswith(".yaml"):`
    with open(def_dir / "d_not_yaml.txt", "w") as f:
        f.write("ignore me")

    br.build(definitions_dir=str(def_dir), out_file=str(out_file))

    assert out_file.exists()
    content = out_file.read_text()

    # Check outputs
    assert "DummyOp1" in content
    assert "DummyOp2" in content
    assert "DummyOp3" in content
    assert "DummyOp4" not in content


def test_build_registry_main(tmp_path: Path) -> None:
    """Test the __main__ block of build_registry.py."""
    with patch("scripts.build_registry.build") as mock_build:
        br.main()
        mock_build.assert_called_once()

    # To cover `if __name__ == "__main__":`
    # run_path recompiles the file, so we can't patch build() from the module.
    # We let it run harmlessly by giving it empty directories.
    def_dir = tmp_path / "definitions"
    def_dir.mkdir()
    out_file = tmp_path / "generated.py"

    with patch.object(sys, "argv", ["build_registry.py"]):
        # We patch os.listdir to return empty, so it does no work.
        with patch("os.listdir", return_value=[]):
            with patch("builtins.open", mock_open()):
                runpy.run_path("scripts/build_registry.py", run_name="__main__")
