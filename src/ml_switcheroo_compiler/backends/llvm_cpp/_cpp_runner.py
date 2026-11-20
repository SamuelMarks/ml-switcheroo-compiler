"""Native C++17 shared-library compiler and ctypes loader for LLVMCPPRunner."""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerCallableProtocol,
    EagerTensorProtocol,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


@typing.runtime_checkable
class _ComputeGraphCDLLProto(typing.Protocol):
    """Protocol for ctypes shared libraries exposing `compute_graph`."""

    @property
    def compute_graph(self) -> EagerCallableProtocol:
        """Return `compute_graph` entry point symbol.

        Returns:
            EagerCallableProtocol: Native C entry point callable.
        """
        ...


class LLVMCPPRunner:
    """Standalone execution runner compiling C++17 code via clang++/g++ and loading via ctypes."""

    def __init__(self, compiler: str | None = None) -> None:
        """Initialize LLVMCPPRunner with preferred compiler.

        Args:
            compiler (str | None): Path or executable name for C++ compiler.
        """
        import shutil

        if compiler:
            self.compiler: str = compiler
        elif shutil.which("clang++"):
            self.compiler = "clang++"
        elif shutil.which("g++"):
            self.compiler = "g++"
        else:
            self.compiler = "clang++"

    def compile_and_load(self, code: str, entry_point: str = "compute_graph") -> Callable[[], str]:
        """Compile C++17 source code into a shared library and load via ctypes.

        Args:
            code (str): The C++ source code.
            entry_point (str): The exported C symbol name.

        Returns:
            Callable[[], str]: Executable callable wrapper.

        Raises:
            RuntimeError: If compilation or library loading fails.
        """
        import shutil

        import ml_switcheroo_compiler.backends.llvm_cpp.generator as _cpp_mod

        temp_dir: str = _cpp_mod.tempfile.mkdtemp()
        src_file: str = _cpp_mod.os.path.join(temp_dir, "graph.cpp")
        lib_ext: str = (
            ".dylib" if _cpp_mod.os.name == "posix" and "darwin" in _cpp_mod.os.uname().sysname.lower() else ".so"
        )
        lib_file: str = _cpp_mod.os.path.join(temp_dir, f"graph{lib_ext}")

        with open(src_file, "w") as f:
            f.write(code)

        compile_cmd: list[str] = [self.compiler, "-std=c++17", "-O3", "-shared", "-fPIC", src_file, "-o", lib_file]
        try:
            _cpp_mod.subprocess.run(compile_cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raw_err = e.stderr if isinstance(e, subprocess.CalledProcessError) else None
            err_out = raw_err.decode() if isinstance(raw_err, bytes) else (str(raw_err) if raw_err else str(e))
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Compilation failed: {err_out}") from e

        try:
            lib: ctypes.CDLL = _cpp_mod.ctypes.CDLL(lib_file)
            if entry_point == "compute_graph" and isinstance(lib, _ComputeGraphCDLLProto):
                compute_func: EagerCallableProtocol | EagerTensorProtocol | ctypes._NamedFuncPointer = lib.compute_graph
            elif adapt_backend_module(lib).has_symbol(entry_point):
                compute_func = adapt_backend_module(lib).require_symbol(entry_point)
            else:
                try:
                    compute_func = lib[entry_point]
                except (AttributeError, KeyError, TypeError) as sym_err:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise RuntimeError(
                        f"Function '{entry_point}' not found in compiled library. "
                        'Ensure it is exported with extern "C".'
                    ) from sym_err
        except (OSError, AttributeError, TypeError, ValueError, RuntimeError, ctypes.ArgumentError) as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Compilation or load failed: {e}") from e

        def executable(*args: BackendArg) -> str:
            """Execute the compiled C++ graph with optional input/output buffer pointers.

            Args:
                *args (BackendArg): Optional (c_inputs, c_outputs, c_shapes) buffer pointers.

            Returns:
                str: Status message.
            """
            if len(args) >= 3:
                compute_func(args[0], args[1], args[2])
            elif len(args) == 2:
                compute_func(args[0], args[1], None)
            elif len(args) == 1:
                compute_func(args[0], None, None)
            else:
                compute_func(None, None, None)
            return "Execution successful"

        return executable


__all__ = ["LLVMCPPRunner", "os", "subprocess", "tempfile"]
