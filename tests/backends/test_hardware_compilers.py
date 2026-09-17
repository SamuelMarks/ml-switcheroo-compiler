"""Exhaustive unit tests for hardware compiler interfaces and syntax validators.

Verifies 100% statement, branch, and function coverage for BaseHardwareCompiler,
CUDACompiler, HIPCompiler, MetalCompiler, and ClangCompiler across native and emulated execution paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.hardware_compilers import (
    BaseHardwareCompiler,
    ClangCompiler,
    CUDACompiler,
    HIPCompiler,
    MetalCompiler,
)


def test_base_hardware_compiler_is_available() -> None:
    """Verify BaseHardwareCompiler toolchain availability checks."""
    base = BaseHardwareCompiler()
    assert base.executable_name == ""
    assert not base.is_available()

    base.executable_name = "existing_binary"
    with patch("shutil.which", return_value="/usr/bin/existing_binary"):
        assert base.is_available()

    with patch("shutil.which", return_value=None):
        assert not base.is_available()


def test_base_hardware_compiler_fallback_syntax_check_delimiter_branches() -> None:
    """Verify all delimiter and token branches in fallback syntax inspection."""
    base = BaseHardwareCompiler()

    # Empty or whitespace strings
    assert not base._fallback_syntax_check("", ["void"])
    assert not base._fallback_syntax_check("   \n\t  ", ["void"])

    # Unmatched closing delimiter on empty stack
    assert not base._fallback_syntax_check(")", ["void"])
    assert not base._fallback_syntax_check("}", ["void"])
    assert not base._fallback_syntax_check("]", ["void"])

    # Mismatched closing delimiter
    assert not base._fallback_syntax_check("(}", ["void"])
    assert not base._fallback_syntax_check("[)", ["void"])
    assert not base._fallback_syntax_check("{]", ["void"])

    # Unclosed opening delimiter remaining on stack
    assert not base._fallback_syntax_check("void foo() {", ["void"])
    assert not base._fallback_syntax_check("void foo([)", ["void"])

    # Missing required token
    assert not base._fallback_syntax_check("int foo() { return 0; }", ["void"])

    # Token boundary matching: substring should not match full token
    assert not base._fallback_syntax_check("avoid foo() { return 0; }", ["void"])

    # Valid syntax with balanced delimiters and tokens
    assert base._fallback_syntax_check("void foo(int x[2]) { if (x[0]) {} }", ["void"])


def test_cuda_compiler_compile_native(tmp_path: Path) -> None:
    """Verify CUDACompiler.compile when native nvcc is available."""
    cuda = CUDACompiler()
    out_file = tmp_path / "out.ptx"

    # Successful compilation with output file created
    def fake_run_success(cmd: list[str], **kwargs: object) -> MagicMock:
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text("// Simulated PTX content", encoding="utf-8")
        return MagicMock(returncode=0, stdout="compiled ok", stderr="")

    with patch.object(cuda, "is_available", return_value=True):
        with patch("subprocess.run", side_effect=fake_run_success):
            res = cuda.compile(
                "__global__ void k() {}",
                output_path=str(out_file),
                options=["-O2"],
            )
            assert res.success is True
            assert res.compiler_cli == "nvcc"
            assert res.output_path == str(out_file)
            assert res.assembly_or_ptx == "// Simulated PTX content"
            assert res.stdout == "compiled ok"
            assert res.returncode == 0

    # Successful compilation where output file was not created
    with patch.object(cuda, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
            res2 = cuda.compile("__global__ void k() {}")
            assert res2.success is True
            assert res2.assembly_or_ptx is None

    # Failed compilation with non-zero returncode
    with patch.object(cuda, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="error")):
            res_fail = cuda.compile("__global__ void k() {}")
            assert res_fail.success is False
            assert res_fail.output_path is None
            assert res_fail.stderr == "error"
            assert res_fail.returncode == 1


def test_cuda_compiler_compile_headless() -> None:
    """Verify CUDACompiler.compile headless fallback path."""
    cuda = CUDACompiler()
    with patch.object(cuda, "is_available", return_value=False):
        # Valid source
        res_ok = cuda.compile("__global__ void k() {}", output_path="/tmp/test.ptx")
        assert res_ok.success is True
        assert res_ok.compiler_cli == "nvcc_emulator"
        assert res_ok.output_path == "/tmp/test.ptx"
        assert res_ok.assembly_or_ptx == "// PTX emulator output"
        assert res_ok.returncode == 0

        # Invalid source
        res_err = cuda.compile("invalid cuda code")
        assert res_err.success is False
        assert res_err.compiler_cli == "nvcc_emulator"
        assert res_err.assembly_or_ptx is None
        assert res_err.returncode == 1


def test_cuda_compiler_validate_syntax() -> None:
    """Verify CUDACompiler.validate_syntax across native and fallback paths."""
    cuda = CUDACompiler()

    # Native available - returncode 0
    with patch.object(cuda, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            assert cuda.validate_syntax("__global__ void k() {}")

    # Native available - returncode != 0
    with patch.object(cuda, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            assert not cuda.validate_syntax("__global__ void k() {}")

    # Native available - exception during execution
    with patch.object(cuda, "is_available", return_value=True):
        with patch("subprocess.run", side_effect=OSError("Process failed")):
            assert cuda.validate_syntax("__global__ void k() {}")
            assert not cuda.validate_syntax("invalid")

    # Native available - exception during execution and tempfile already unlinked
    with patch.object(cuda, "is_available", return_value=True):
        with patch("subprocess.run", side_effect=OSError("Process failed")):
            with patch("os.path.exists", return_value=False):
                assert cuda.validate_syntax("__global__ void k() {}")

    # Native not available
    with patch.object(cuda, "is_available", return_value=False):
        assert cuda.validate_syntax("__global__ void k() {}")
        assert not cuda.validate_syntax("invalid")


def test_hip_compiler_compile_and_validate(tmp_path: Path) -> None:
    """Verify HIPCompiler compile and validate_syntax methods."""
    hip = HIPCompiler()
    out_file = tmp_path / "out.o"

    # Native compile success with explicit output_path and options
    with patch.object(hip, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="hip ok", stderr="")):
            res = hip.compile(
                "__global__ void k() {}",
                output_path=str(out_file),
                options=["-O1"],
            )
            assert res.success is True
            assert res.compiler_cli == "hipcc"
            assert res.output_path == str(out_file)

    # Native compile failure with default output_path and options
    with patch.object(hip, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="hip fail")):
            res_fail = hip.compile("__global__ void k() {}")
            assert res_fail.success is False
            assert res_fail.output_path is None
            assert res_fail.returncode == 1

    # Headless compile
    with patch.object(hip, "is_available", return_value=False):
        res_ok = hip.compile("__global__ void k() {}")
        assert res_ok.success is True
        assert res_ok.compiler_cli == "hipcc_emulator"

        res_bad = hip.compile("bad hip")
        assert res_bad.success is False
        assert res_bad.returncode == 1

    # validate_syntax native success, failure, exception, and fallback
    with patch.object(hip, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            assert hip.validate_syntax("__global__ void k() {}")

        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            assert not hip.validate_syntax("__global__ void k() {}")

        with patch("subprocess.run", side_effect=RuntimeError("hipcc crash")):
            assert hip.validate_syntax("__global__ void k() {}")
            assert not hip.validate_syntax("bad")

        with patch("subprocess.run", side_effect=RuntimeError("hipcc crash")):
            with patch("os.path.exists", return_value=False):
                assert hip.validate_syntax("__global__ void k() {}")

    with patch.object(hip, "is_available", return_value=False):
        assert hip.validate_syntax("__global__ void k() {}")
        assert not hip.validate_syntax("bad")


def test_metal_compiler_is_available() -> None:
    """Verify MetalCompiler.is_available branches."""
    metal = MetalCompiler()

    # xcrun not on PATH
    with patch("shutil.which", return_value=None):
        assert not metal.is_available()

    # xcrun found, metal command succeeds
    with patch("shutil.which", return_value="/usr/bin/xcrun"):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            assert metal.is_available()

        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            assert not metal.is_available()

        with patch("subprocess.run", side_effect=OSError("xcrun failed")):
            assert not metal.is_available()


def test_metal_compiler_compile_and_validate(tmp_path: Path) -> None:
    """Verify MetalCompiler compile and validate_syntax methods."""
    metal = MetalCompiler()
    code_with_runtime = "kernel void k() {}\n// Active runtime dispatch\nruntime_code();"
    out_file = tmp_path / "out.air"

    # Native compile success with explicit output_path and options
    with patch.object(metal, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="metal ok", stderr="")):
            res = metal.compile(
                code_with_runtime,
                output_path=str(out_file),
                options=["-O1"],
            )
            assert res.success is True
            assert res.compiler_cli == "xcrun metal"
            assert res.output_path == str(out_file)

    # Native compile failure with default output_path
    with patch.object(metal, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="metal fail")):
            res_fail = metal.compile(code_with_runtime)
            assert res_fail.success is False
            assert res_fail.output_path is None
            assert res_fail.returncode == 1

    # Headless compile
    with patch.object(metal, "is_available", return_value=False):
        res_ok = metal.compile(code_with_runtime)
        assert res_ok.success is True
        assert res_ok.compiler_cli == "metal_emulator"

        res_bad = metal.compile("invalid metal")
        assert res_bad.success is False
        assert res_bad.returncode == 1

    # validate_syntax native success, failure, exception, and fallback
    with patch.object(metal, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            assert metal.validate_syntax(code_with_runtime)

        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            assert not metal.validate_syntax(code_with_runtime)

        with patch("subprocess.run", side_effect=RuntimeError("metal crash")):
            assert metal.validate_syntax(code_with_runtime)
            assert not metal.validate_syntax("bad")

        with patch("subprocess.run", side_effect=RuntimeError("metal crash")):
            with patch("os.path.exists", return_value=False):
                assert metal.validate_syntax(code_with_runtime)

    with patch.object(metal, "is_available", return_value=False):
        assert metal.validate_syntax(code_with_runtime)
        assert not metal.validate_syntax("bad")


def test_clang_compiler_compile_and_validate(tmp_path: Path) -> None:
    """Verify ClangCompiler compile and validate_syntax methods."""
    clang = ClangCompiler()
    code = "void compute() {}"
    out_file = tmp_path / "out.o"

    # Native compile success with explicit output_path and options
    with patch.object(clang, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="clang ok", stderr="")):
            res = clang.compile(
                code,
                output_path=str(out_file),
                options=["-O2"],
            )
            assert res.success is True
            assert res.compiler_cli == "clang++"
            assert res.output_path == str(out_file)

    # Native compile failure with default output_path
    with patch.object(clang, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="clang fail")):
            res_fail = clang.compile(code)
            assert res_fail.success is False
            assert res_fail.output_path is None
            assert res_fail.returncode == 1

    # Headless compile
    with patch.object(clang, "is_available", return_value=False):
        res_ok = clang.compile(code)
        assert res_ok.success is True
        assert res_ok.compiler_cli == "clang_emulator"

        res_bad = clang.compile("bad c++")
        assert res_bad.success is False
        assert res_bad.returncode == 1

    # validate_syntax native success, failure, exception, and fallback
    with patch.object(clang, "is_available", return_value=True):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            assert clang.validate_syntax(code)

        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            assert not clang.validate_syntax(code)

        with patch("subprocess.run", side_effect=RuntimeError("clang crash")):
            assert clang.validate_syntax(code)
            assert not clang.validate_syntax("bad")

        with patch("subprocess.run", side_effect=RuntimeError("clang crash")):
            with patch("os.path.exists", return_value=False):
                assert clang.validate_syntax(code)

    with patch.object(clang, "is_available", return_value=False):
        assert clang.validate_syntax(code)
        assert not clang.validate_syntax("bad")
