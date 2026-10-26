"""Tests for the ml-switcheroo-compiler CLI and __main__ entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_switcheroo_compiler import __main__ as main_mod
from ml_switcheroo_compiler import __version__
from ml_switcheroo_compiler.cli import build_parser, compile_source_code, main


def test_main_module_exports() -> None:
    """Verify __main__ exports main."""
    assert main_mod.main is main
    assert "main" in main_mod.__all__


def test_build_parser_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --version outputs package version and exits 0.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_cli_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify invoking CLI without subcommand prints help and returns 0.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    rc = main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "ml-switcheroo-compiler" in captured.out


def test_cli_backends(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify backends subcommand lists registered compiler backends.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    rc = main(["backends"])
    assert rc == 0
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert "jax" in lines
    assert "pytorch" in lines
    assert "webgpu" in lines


def test_compile_source_code_targets() -> None:
    """Verify compile_source_code handles framework, webgpu, and wasm targets."""
    src = """import torch

def fn(x):
    return torch.add(x, x)
"""
    jax_out = compile_source_code(src, target="jax")
    assert "def fn" in jax_out

    wgsl_out = compile_source_code(src, target="webgpu")
    assert "@compute" in wgsl_out or "fn " in wgsl_out

    wasm_out = compile_source_code(src, target="wasm_simd")
    assert "wasm_simd128.h" in wasm_out


def test_cli_compile_stdout_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify compile subcommand both to stdout and to output file.

    Args:
        tmp_path (Path): Temporary directory path fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    in_file = tmp_path / "model.py"
    in_file.write_text(
        """import torch

def f(x):
    return x
""",
        encoding="utf-8",
    )

    rc_stdout = main(["compile", str(in_file), "--target", "jax"])
    assert rc_stdout == 0
    out_stdout = capsys.readouterr().out
    assert "def f" in out_stdout

    out_file = tmp_path / "out.py"
    rc_file = main(["compile", str(in_file), "-t", "jax", "-o", str(out_file)])
    assert rc_file == 0
    assert "def f" in out_file.read_text(encoding="utf-8")
