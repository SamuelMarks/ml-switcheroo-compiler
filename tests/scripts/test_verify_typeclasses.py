"""Unit tests for `scripts/verify_typeclasses.py` CLI."""

from __future__ import annotations

import pathlib

import pytest

from scripts.verify_typeclasses import main


def test_verify_typeclasses_cli_pass_and_fail(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify exit code 0 on real repository backends and exit code 1 on invalid fixture dir.

    Args:
        tmp_path (pathlib.Path): Temporary pytest fixture directory.
        capsys (pytest.CaptureFixture[str]): Standard output/error capture fixture.
    """
    rc_ok = main(["--backends-dir", "src/ml_switcheroo_compiler/backends"])
    captured_ok = capsys.readouterr()
    assert rc_ok == 0
    assert "Haskell-style backend typeclass and YAML verification passed." in captured_ok.out

    rc_fail = main(["--backends-dir", str(tmp_path / "missing_backends")])
    captured_fail = capsys.readouterr()
    assert rc_fail == 1
    assert "backends directory does not exist" in captured_fail.err
