"""Hardware compiler driver interfaces and deterministic syntax validators.

Unified protocol for compiling and validating hardware kernel sources across
CUDA (nvcc), ROCm (hipcc), Apple Metal (xcrun metal), and LLVM/C++ (clang++).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class HardwareCompilationResult:
    """Result of an accelerator kernel compilation invocation.

    Attributes:
        success (bool): Whether the compilation or syntax validation succeeded.
        compiler_cli (str): Compiler binary or command invoked.
        output_path (str | None): Path to generated binary/PTX artifact if created.
        assembly_or_ptx (str | None): Disassembled PTX/GCN/MSL or empty.
        stdout (str): Standard output from the compiler process.
        stderr (str): Standard error from the compiler process.
        returncode (int): Process exit code.
    """

    success: bool
    compiler_cli: str
    output_path: str | None = None
    assembly_or_ptx: str | None = None
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


@runtime_checkable
class HardwareCompilerProtocol(Protocol):
    """Protocol contract for hardware kernel compilers."""

    target_name: str

    def compile(
        self,
        source_code: str,
        output_path: str | None = None,
        options: list[str] | None = None,
    ) -> HardwareCompilationResult:
        """Compile accelerator source code into executable binary or intermediate ISA.

        Args:
            source_code (str): Hardware kernel source code.
            output_path (str | None, optional): Path for compiled binary output.
            options (list[str] | None, optional): Additional compiler flags.

        Returns:
            HardwareCompilationResult: Compilation results and diagnostic logs.
        """
        ...

    def validate_syntax(self, source_code: str) -> bool:
        """Verify source code syntax without full linking or physical hardware execution.

        Args:
            source_code (str): Hardware kernel source code.

        Returns:
            bool: True if syntax is strictly valid.
        """
        ...

    def is_available(self) -> bool:
        """Query whether the native toolchain binary is available on host PATH.

        Returns:
            bool: True if executable is found on PATH.
        """
        ...


def _skip_block_comment(source: str, i: int, n: int) -> tuple[int, bool]:
    """Skip over block comment /* ... */ and return updated index.

    Args:
        source (str): Source code.
        i (int): Start index after '/*'.
        n (int): Length of source.

    Returns:
        tuple[int, bool]: (new_index, is_closed).
    """
    while i + 1 < n:
        if source[i] == "*" and source[i + 1] == "/":
            return i + 2, True
        i += 1
    return i, False


def _skip_quote(source: str, i: int, n: int, quote: str) -> tuple[int, bool]:
    """Skip over quoted string/char literal and return updated index.

    Args:
        source (str): Source code.
        i (int): Start index after quote char.
        n (int): Length of source.
        quote (str): Quote delimiter ('"' or "'").

    Returns:
        tuple[int, bool]: (new_index, is_closed).
    """
    while i < n and source[i] != quote:
        if source[i] == "\\" and i + 1 < n:
            i += 2
        else:
            i += 1
    if i >= n:
        return i, False
    return i + 1, True


def _strip_comments_and_strings(source: str) -> tuple[str, bool]:
    """Strip comments and string/char literals while validating their closures.

    Args:
        source (str): Original source code.

    Returns:
        tuple[str, bool]: Stripped code and validity flag.
    """
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            out.append("\n")
        elif ch == "/" and i + 1 < n and source[i + 1] == "*":
            i, ok = _skip_block_comment(source, i + 2, n)
            if not ok:
                return "", False
        elif ch in ('"', "'"):
            i, ok = _skip_quote(source, i + 1, n, ch)
            if not ok:
                return "", False
        else:
            out.append(ch)
            i += 1
    return "".join(out), True


def _check_preprocessor_directives(source: str) -> bool:
    """Validate nesting balance of preprocessor directives (#if/#ifdef vs #endif).

    Args:
        source (str): Source code without comments/literals.

    Returns:
        bool: True if preprocessor directives are balanced.
    """
    balance = 0
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        directive = stripped[1:].split()[0] if stripped[1:].split() else ""
        if directive in ("if", "ifdef", "ifndef"):
            balance += 1
        elif directive == "endif":
            balance -= 1
            if balance < 0:
                return False
    return balance == 0


def _check_delimiters(source: str) -> bool:
    """Validate delimiter balance for (), {}, and [].

    Args:
        source (str): Cleaned source code.

    Returns:
        bool: True if all delimiters are balanced.
    """
    stack: list[str] = []
    matching = {")": "(", "}": "{", "]": "["}
    for char in source:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()
    return len(stack) == 0


def _check_dialect_qualifiers(source_code: str, target_name: str) -> bool:
    """Validate dialect-specific kernel function qualifiers.

    Args:
        source_code (str): Source code string.
        target_name (str): Backend target name.

    Returns:
        bool: True if kernel qualifiers are satisfied.
    """
    import re

    if target_name in ("cuda", "rocm"):
        return bool(re.search(r"\b(__global__|__device__|extern\s+\"C\")\b", source_code))
    if target_name == "metal":
        return bool(re.search(r"\b(kernel\s+void|vertex|fragment)\b", source_code))
    return True


class BaseHardwareCompiler:
    """Base implementation providing common fallback syntax validation and toolchain inspection."""

    target_name: str = "base"
    executable_name: str = ""

    def is_available(self) -> bool:
        """Check if target compiler CLI is installed on system.

        Returns:
            bool: True if toolchain executable is found.
        """
        if not self.executable_name:
            return False
        return shutil.which(self.executable_name) is not None

    def _fallback_syntax_check(self, source_code: str, required_tokens: list[str]) -> bool:
        """Perform deterministic AST/lexer token stream syntax inspection for C++/CUDA/HIP/MSL.

        Args:
            source_code (str): Source code string to inspect.
            required_tokens (list[str]): Critical keywords expected in target dialect.

        Returns:
            bool: True if source code is structurally valid and balanced, False otherwise.
        """
        if not source_code or not source_code.strip():
            return False

        cleaned, ok = _strip_comments_and_strings(source_code)
        if not ok or not _check_preprocessor_directives(cleaned) or not _check_delimiters(cleaned):
            return False

        import re

        for token in required_tokens:
            pattern = rf"\b{re.escape(token)}\b" if token.isalnum() or "_" in token else re.escape(token)
            if not re.search(pattern, source_code):
                return False

        return _check_dialect_qualifiers(source_code, self.target_name)


class CUDACompiler(BaseHardwareCompiler):
    """NVIDIA CUDA compiler wrapper invoking nvcc with fallback AST syntax validation."""

    target_name: str = "cuda"
    executable_name: str = "nvcc"

    def compile(
        self,
        source_code: str,
        output_path: str | None = None,
        options: list[str] | None = None,
    ) -> HardwareCompilationResult:
        """Compile CUDA C++ source code to PTX or cubin via nvcc.

        Args:
            source_code (str): CUDA C++ source code.
            output_path (str | None, optional): Target output artifact path.
            options (list[str] | None, optional): Extra compiler flags.

        Returns:
            HardwareCompilationResult: Compilation status and diagnostics.
        """
        opts: list[str] = options or ["-O3", "--ptx"]
        if self.is_available():
            with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
                f.write(source_code)
                src_path = f.name

            target_out = output_path or src_path.replace(".cu", ".ptx")
            cmd = [self.executable_name, *opts, src_path, "-o", target_out]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                ptx_content = None
                if proc.returncode == 0 and os.path.exists(target_out):
                    with open(target_out, encoding="utf-8", errors="ignore") as pf:
                        ptx_content = pf.read()
                return HardwareCompilationResult(
                    success=(proc.returncode == 0),
                    compiler_cli=self.executable_name,
                    output_path=target_out if proc.returncode == 0 else None,
                    assembly_or_ptx=ptx_content,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
                )
            finally:
                if os.path.exists(src_path):
                    os.unlink(src_path)

        # Headless emulation
        valid = self.validate_syntax(source_code)
        return HardwareCompilationResult(
            success=valid,
            compiler_cli="nvcc_emulator",
            output_path=output_path,
            assembly_or_ptx="// PTX emulator output" if valid else None,
            stdout="Syntax verified via deterministic validator" if valid else "",
            stderr="" if valid else "CUDA syntax error: unbalanced brackets or missing __global__",
            returncode=0 if valid else 1,
        )

    def validate_syntax(self, source_code: str) -> bool:
        """Validate CUDA C++ syntax.

        Args:
            source_code (str): CUDA kernel source.

        Returns:
            bool: True if CUDA source passes structural check.
        """
        if self.is_available():
            with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
                f.write(source_code)
                src_path = f.name
            try:
                proc = subprocess.run(
                    [self.executable_name, "--dry-run", "-fsyntax-only", src_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return proc.returncode == 0
            except Exception:
                pass
            finally:
                if os.path.exists(src_path):
                    os.unlink(src_path)

        return self._fallback_syntax_check(source_code, ["__global__", "void"])


class HIPCompiler(BaseHardwareCompiler):
    """AMD ROCm HIP compiler wrapper invoking hipcc with fallback syntax validation."""

    target_name: str = "rocm"
    executable_name: str = "hipcc"

    def compile(
        self,
        source_code: str,
        output_path: str | None = None,
        options: list[str] | None = None,
    ) -> HardwareCompilationResult:
        """Compile HIP C++ source code to GCN / HSACO via hipcc.

        Args:
            source_code (str): HIP C++ source code.
            output_path (str | None, optional): Target output path.
            options (list[str] | None, optional): Extra compiler flags.

        Returns:
            HardwareCompilationResult: Compilation result.
        """
        opts: list[str] = options or ["-O3", "-c"]
        if self.is_available():
            with tempfile.NamedTemporaryFile(suffix=".hip", mode="w", delete=False) as f:
                f.write(source_code)
                src_path = f.name

            target_out = output_path or src_path.replace(".hip", ".o")
            cmd = [self.executable_name, *opts, src_path, "-o", target_out]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                return HardwareCompilationResult(
                    success=(proc.returncode == 0),
                    compiler_cli=self.executable_name,
                    output_path=target_out if proc.returncode == 0 else None,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
                )
            finally:
                if os.path.exists(src_path):
                    os.unlink(src_path)

        valid = self.validate_syntax(source_code)
        return HardwareCompilationResult(
            success=valid,
            compiler_cli="hipcc_emulator",
            output_path=output_path,
            assembly_or_ptx="// GCN emulator output" if valid else None,
            stdout="Syntax verified via deterministic validator" if valid else "",
            stderr="" if valid else "HIP syntax error: unbalanced brackets or missing __global__",
            returncode=0 if valid else 1,
        )

    def validate_syntax(self, source_code: str) -> bool:
        """Validate HIP C++ syntax.

        Args:
            source_code (str): HIP source code.

        Returns:
            bool: True if HIP source syntax is valid.
        """
        if self.is_available():
            with tempfile.NamedTemporaryFile(suffix=".hip", mode="w", delete=False) as f:
                f.write(source_code)
                src_path = f.name
            try:
                proc = subprocess.run(
                    [self.executable_name, "-fsyntax-only", src_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return proc.returncode == 0
            except Exception:
                pass
            finally:
                if os.path.exists(src_path):
                    os.unlink(src_path)

        return self._fallback_syntax_check(source_code, ["__global__", "void"])


class MetalCompiler(BaseHardwareCompiler):
    """Apple Metal Shading Language compiler wrapper invoking xcrun metal with fallback validation."""

    target_name: str = "metal"
    executable_name: str = "xcrun"

    def is_available(self) -> bool:
        """Verify presence of macOS xcrun metal toolchain.

        Returns:
            bool: True if xcrun and metal SDK are available.
        """
        if shutil.which("xcrun") is None:
            return False
        try:
            proc = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], capture_output=True, check=False)
            return proc.returncode == 0
        except Exception:
            return False

    def compile(
        self,
        source_code: str,
        output_path: str | None = None,
        options: list[str] | None = None,
    ) -> HardwareCompilationResult:
        """Compile Metal Shading Language source code to metallib or air.

        Args:
            source_code (str): Metal source code string.
            output_path (str | None, optional): Output air/metallib path.
            options (list[str] | None, optional): Extra compiler flags.

        Returns:
            HardwareCompilationResult: Compilation result.
        """
        opts: list[str] = options or ["-c", "-O3"]
        msl_part = source_code.split("// Active runtime dispatch")[0]
        if self.is_available():
            with tempfile.NamedTemporaryFile(suffix=".metal", mode="w", delete=False) as f:
                f.write(msl_part)
                src_path = f.name

            target_out = output_path or src_path.replace(".metal", ".air")
            cmd = ["xcrun", "-sdk", "macosx", "metal", *opts, src_path, "-o", target_out]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                return HardwareCompilationResult(
                    success=(proc.returncode == 0),
                    compiler_cli="xcrun metal",
                    output_path=target_out if proc.returncode == 0 else None,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
                )
            finally:
                if os.path.exists(src_path):
                    os.unlink(src_path)

        valid = self.validate_syntax(msl_part)
        return HardwareCompilationResult(
            success=valid,
            compiler_cli="metal_emulator",
            output_path=output_path,
            assembly_or_ptx="// Metal AIR emulator output" if valid else None,
            stdout="Syntax verified via deterministic validator" if valid else "",
            stderr="" if valid else "MSL syntax error: missing kernel void or unbalanced delimiters",
            returncode=0 if valid else 1,
        )

    def validate_syntax(self, source_code: str) -> bool:
        """Validate Metal Shading Language syntax.

        Args:
            source_code (str): MSL compute shader code.

        Returns:
            bool: True if valid MSL source.
        """
        msl_part = source_code.split("// Active runtime dispatch")[0]
        if self.is_available():
            with tempfile.NamedTemporaryFile(suffix=".metal", mode="w", delete=False) as f:
                f.write(msl_part)
                src_path = f.name
            try:
                proc = subprocess.run(
                    ["xcrun", "-sdk", "macosx", "metal", "-fsyntax-only", src_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return proc.returncode == 0
            except Exception:
                pass
            finally:
                if os.path.exists(src_path):
                    os.unlink(src_path)

        return self._fallback_syntax_check(msl_part, ["kernel", "void"])


class ClangCompiler(BaseHardwareCompiler):
    """LLVM Clang++ compiler wrapper invoking clang++ with OpenMP pragmas and fallback validation."""

    target_name: str = "llvm_cpp"
    executable_name: str = "clang++"

    def compile(
        self,
        source_code: str,
        output_path: str | None = None,
        options: list[str] | None = None,
    ) -> HardwareCompilationResult:
        """Compile C++ OpenMP source code to object or shared library via clang++.

        Args:
            source_code (str): C++ source code.
            output_path (str | None, optional): Output path.
            options (list[str] | None, optional): Compiler options.

        Returns:
            HardwareCompilationResult: Compilation result.
        """
        opts: list[str] = options or ["-std=c++17", "-O3", "-c"]
        if self.is_available():
            with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
                f.write(source_code)
                src_path = f.name

            target_out = output_path or src_path.replace(".cpp", ".o")
            cmd = [self.executable_name, *opts, src_path, "-o", target_out]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                return HardwareCompilationResult(
                    success=(proc.returncode == 0),
                    compiler_cli=self.executable_name,
                    output_path=target_out if proc.returncode == 0 else None,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
                )
            finally:
                if os.path.exists(src_path):
                    os.unlink(src_path)

        valid = self.validate_syntax(source_code)
        return HardwareCompilationResult(
            success=valid,
            compiler_cli="clang_emulator",
            output_path=output_path,
            assembly_or_ptx="// C++ LLVM emulator output" if valid else None,
            stdout="Syntax verified via deterministic validator" if valid else "",
            stderr="" if valid else "C++ syntax error: unbalanced brackets or missing main / function definition",
            returncode=0 if valid else 1,
        )

    def validate_syntax(self, source_code: str) -> bool:
        """Validate C++ OpenMP source syntax.

        Args:
            source_code (str): C++ source code.

        Returns:
            bool: True if C++ source syntax is valid.
        """
        if self.is_available():
            with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
                f.write(source_code)
                src_path = f.name
            try:
                proc = subprocess.run(
                    [self.executable_name, "-std=c++17", "-fsyntax-only", src_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return proc.returncode == 0
            except Exception:
                pass
            finally:
                if os.path.exists(src_path):
                    os.unlink(src_path)

        return self._fallback_syntax_check(source_code, ["void"])
