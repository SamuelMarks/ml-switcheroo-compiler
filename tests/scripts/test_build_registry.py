"""Tests for the build_registry script."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import scripts.build_registry as br


def test_build_registry_multi_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test building registry with a multi-op YAML file.

    Args:
        tmp_path (Path): Pytest fixture for temporary directory.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for monkeypatching.
    """
    def_dir = "src/ml_switcheroo_compiler/ops/definitions"
    dummy_file = os.path.join(def_dir, "zzz_dummy_multi.yaml")

    data: dict[str, dict[str, str]] = {"DummyOp1": {"signature": "(x) -> x"}, "DummyOp2": {"signature": "(y) -> y"}}

    with open(dummy_file, "w") as f:
        yaml.dump(data, f)

    try:
        # Run normally
        br.build()

        # We don't import OPS_REGISTRY directly because we want to read it anew
        # It's better to just check the file content
        out_file = "src/ml_switcheroo_compiler/ops/generated_registry.py"
        with open(out_file) as f:
            content = f.read()
        assert "DummyOp1" in content
        assert "DummyOp2" in content

    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        # Re-run build to restore original state
        br.build()


def test_build_registry_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the __main__ block of build_registry.py.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for monkeypatching.
    """
    with patch("scripts.build_registry.build") as mock_build:
        br.main()
        mock_build.assert_called_once()
