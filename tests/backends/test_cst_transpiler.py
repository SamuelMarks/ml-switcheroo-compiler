"""Unit tests for the CST transpiler backend.

This module contains test cases to verify the correctness of the CST-based transpilation
process, including import translation, function call mapping, syntax validation, AST
diffing, and dry-run type inference.
"""

from ml_switcheroo_compiler.backends.cst_transpiler import (
    CSTTransformer,
    transpile_source,
    type_infer_dry_run,
    validate_diff,
)


def test_transpile_source_jax_import() -> None:
    """Verifies that PyTorch imports are correctly transpiled to JAX imports.

    Returns:
    None
    """
    source = "from torch import nn\n"
    expected = "from jax import nn\n"
    assert transpile_source(source, target_framework="jax") == expected


def test_transpile_source_other_framework() -> None:
    """Verifies that imports remain unchanged when transpiling to an unsupported framework.

    Returns:
    None
    """
    source = "from torch import nn\n"
    assert transpile_source(source, target_framework="mlx") == source


def test_transpile_source_call() -> None:
    """Verifies that PyTorch function calls are correctly transpiled to JAX equivalents.

    Returns:
    None
    """
    source = "torch.add(x, y)\n"
    expected = "jax.numpy.add(x, y)\n"
    assert transpile_source(source, target_framework="jax") == expected


def test_validate_diff_different() -> None:
    """Verifies that validate_diff returns True when the transpiled code is syntactically.

    different but valid

    Returns:
    None
    """
    source = "x = 1\n"
    transpiled = "x = 2\n"
    assert validate_diff(source, transpiled) is True


def test_validate_diff_same() -> None:
    """Verifies that validate_diff returns False when the transpiled code is identical to.

    the source

    Returns:
    None
    """
    source = "x = 1\n"
    assert validate_diff(source, source) is False


def test_validate_diff_invalid_syntax() -> None:
    """Verifies that validate_diff returns False when the transpiled code contains invalid.

    syntax

    Returns:
    None
    """
    source = "x = 1\n"
    transpiled = "x = 2\n\ninvalid code"
    assert validate_diff(source, transpiled) is False


def test_type_infer_dry_run_success() -> None:
    """Verifies that type_infer_dry_run returns a success status for syntactically valid.

    code

    Returns:
    None
    """
    source = "x = 1\n"
    res = type_infer_dry_run(source)
    assert res == {"dry_run": "success"}


def test_type_infer_dry_run_failure() -> None:
    """Verifies that type_infer_dry_run returns a failed status for syntactically invalid.

    code

    Returns:
    None
    """
    source = "x = 1\n\ninvalid code"
    res = type_infer_dry_run(source)
    assert res == {"dry_run": "failed"}


def test_cst_transformer_init() -> None:
    """Verifies that CSTTransformer initializes with the default target framework.

    Returns:
    None
    """
    transformer = CSTTransformer()
    assert transformer.target_framework == "jax"


def test_transpile_source_other_call() -> None:
    """Verifies that non-framework function calls remain unchanged during transpilation.

    Returns:
    None
    """
    source = "print('hello')\n"
    assert transpile_source(source, target_framework="jax") == source
