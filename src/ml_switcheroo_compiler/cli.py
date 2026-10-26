"""Command Line Interface for ml-switcheroo-compiler."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ml_switcheroo_compiler import __version__
from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source
from ml_switcheroo_compiler.backends.registry import BackendRegistry

__all__ = [
    "build_parser",
    "compile_source_code",
    "main",
]


def compile_source_code(source_code: str, target: str = "jax") -> str:
    """Compile or transpile Python source code to the specified target backend.

    Args:
        source_code (str): Input Python source code string.
        target (str): Target framework or hardware/edge backend name.

    Returns:
        str: Compiled or transpiled output code.
    """
    if target in ("webgpu", "edge_wgsl"):
        from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
        from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass

        graph = parse_ast_to_ir(source_code)
        shape_inference_pass(graph)
        return WebGPUCodeGenerator(graph).generate()

    if target in ("wasm", "wasm_simd", "edge_wasm", "edge_wasm_simd"):
        from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
        from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass

        graph = parse_ast_to_ir(source_code)
        shape_inference_pass(graph)
        return WasmCodeGenerator(graph).generate()

    return transpile_source(source_code, target_framework=target)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ml-switcheroo-compiler CLI.

    Returns:
        argparse.ArgumentParser: Configured CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="ml-switcheroo-compiler",
        description="Abstract ML Machine Compiler for Reverse-Mode AD, Optimization, and Multi-Backend Lowering.",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile or transpile a Python source file to a target backend.",
    )
    compile_parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input Python source file.",
    )
    compile_parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="jax",
        help="Target backend framework (default: jax).",
    )
    compile_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output file path (prints to stdout if omitted).",
    )

    subparsers.add_parser(
        "backends",
        help="List all registered compiler backends.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the ml-switcheroo-compiler command-line interface.

    Args:
        argv (Sequence[str] | None): Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        int: Process exit status code (0 for success).
    """
    parser = build_parser()
    parsed = parser.parse_args(list(argv) if argv is not None else None)

    if parsed.command == "compile":
        with open(parsed.input_file, encoding="utf-8") as infile:
            source_code = infile.read()
        result = compile_source_code(source_code, target=str(parsed.target))
        if parsed.output:
            with open(parsed.output, "w", encoding="utf-8") as outfile:
                outfile.write(result)
        else:
            print(result)
        return 0

    if parsed.command == "backends":
        for name in sorted(BackendRegistry._LAZY_MODULES):
            print(name)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
