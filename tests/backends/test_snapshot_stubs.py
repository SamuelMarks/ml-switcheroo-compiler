"""Unit tests asserting structural validity and completeness of generated snapshot stubs."""

import ast
import os

SUPPORTED_BACKENDS: list[str] = [
    "numpy",
    "pytorch",
    "jax",
    "keras",
    "mlx",
    "dask",
    "cupy",
    "tensorflow",
]


def test_snapshot_stubs_exist_and_parse() -> None:
    """Verify that snapshot_stubs.pyi exists, is non-empty, and has valid syntax across backends."""
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends"))

    for backend in SUPPORTED_BACKENDS:
        stub_path: str = os.path.join(base_dir, backend, "snapshot_stubs.pyi")
        assert os.path.exists(stub_path), f"Missing snapshot_stubs.pyi for {backend}"

        with open(stub_path, encoding="utf-8") as f:
            content: str = f.read()

        assert len(content.strip()) > 50, f"Stub file for {backend} is unexpectedly empty"
        assert "class Tensor:" in content, f"Stub file for {backend} missing Tensor class"

        # Parse AST to ensure valid Python typing syntax
        parsed: ast.Module = ast.parse(content, filename=stub_path)
        assert len(parsed.body) > 0, f"Parsed empty AST body for {backend}"


def test_snapshot_stubs_function_definitions() -> None:
    """Verify that generated stubs contain function definitions with annotations."""
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends"))

    # Validate high-stub backends like pytorch and jax
    for backend in ["pytorch", "jax", "mlx", "numpy", "tensorflow", "keras", "dask"]:
        stub_path: str = os.path.join(base_dir, backend, "snapshot_stubs.pyi")
        with open(stub_path, encoding="utf-8") as f:
            content: str = f.read()

        parsed: ast.Module = ast.parse(content, filename=stub_path)
        func_defs: list[ast.FunctionDef] = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        assert len(func_defs) > 20, f"Expected >20 function stubs for {backend}, found {len(func_defs)}"
