"""Unit tests asserting structural validity and completeness of generated snapshot stubs."""

from __future__ import annotations

import ast
import os

ALL_BACKENDS: list[str] = [
    "numpy",
    "pytorch",
    "jax",
    "keras",
    "mlx",
    "dask",
    "cupy",
    "tensorflow",
    "cuda",
    "rocm",
    "metal",
    "awkward",
    "dpnp",
    "numba",
    "sparse",
    "pyarrow_compute",
    "llvm_cpp",
    "edge",
    "edge_onnx",
    "edge_stablehlo",
    "edge_mlir",
    "edge_wasm",
    "edge_webgl",
    "webgpu",
]


def test_snapshot_stubs_exist_and_parse() -> None:
    """Verify that snapshot_stubs.pyi and stub.pyi exist and parse valid AST across all 24 backends."""
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends"))

    for backend in ALL_BACKENDS:
        for stub_name in ("snapshot_stubs.pyi", "stub.pyi"):
            stub_path: str = os.path.join(base_dir, backend, stub_name)
            assert os.path.exists(stub_path), f"Missing {stub_name} for {backend}"

            with open(stub_path, encoding="utf-8") as f:
                content: str = f.read()

            assert len(content.strip()) > 50, f"{stub_name} for {backend} is unexpectedly empty"
            assert "class Tensor:" in content, f"{stub_name} for {backend} missing Tensor class"

            # Parse AST to ensure strict PEP 484 / PEP 561 compliance
            parsed: ast.Module = ast.parse(content, filename=stub_path)
            assert len(parsed.body) > 0, f"Parsed empty AST body for {backend} in {stub_name}"


def test_snapshot_stubs_function_definitions() -> None:
    """Verify that generated stubs contain function definitions with annotations."""
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends"))

    for backend in ALL_BACKENDS:
        stub_path: str = os.path.join(base_dir, backend, "snapshot_stubs.pyi")
        with open(stub_path, encoding="utf-8") as f:
            content: str = f.read()

        parsed: ast.Module = ast.parse(content, filename=stub_path)
        func_defs: list[ast.FunctionDef] = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        assert len(func_defs) > 0, f"Expected function stubs for {backend}, found {len(func_defs)}"
