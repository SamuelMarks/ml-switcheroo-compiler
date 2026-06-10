"""Tests for the CST Transpiler."""

from ml_switcheroo.backends.cst_transpiler import (
    CSTTransformer,
    transpile_source,
    validate_diff,
    type_infer_dry_run,
)


def test_transpile_source_jax_import() -> None:
    """Test transpilation of imports to jax."""
    source = "from torch import nn\n"
    expected = "from jax import nn\n"
    assert transpile_source(source, target_framework="jax") == expected


def test_transpile_source_other_framework() -> None:
    """Test transpilation of imports to other framework."""
    source = "from torch import nn\n"
    assert transpile_source(source, target_framework="mlx") == source


def test_transpile_source_call() -> None:
    """Test transpilation of torch calls to jax.numpy calls."""
    source = "torch.add(x, y)\n"
    expected = "jax.numpy.add(x, y)\n"
    assert transpile_source(source, target_framework="jax") == expected


def test_transpile_source_call_other_framework() -> None:
    """Test transpilation of calls to other framework."""
    source = "torch.add(x, y)\n"
    assert transpile_source(source, target_framework="mlx") == source


def test_validate_diff_different() -> None:
    """Test validate_diff with different syntactically valid code."""
    source = "x = 1\n"
    transpiled = "x = 2\n"
    assert validate_diff(source, transpiled) is True


def test_validate_diff_same() -> None:
    """Test validate_diff with the same code."""
    source = "x = 1\n"
    assert validate_diff(source, source) is False


def test_validate_diff_invalid_syntax() -> None:
    """Test validate_diff with invalid syntax in transpiled code."""
    source = "x = 1\n"
    transpiled = "x = 2\n\ninvalid code"
    assert validate_diff(source, transpiled) is False


def test_type_infer_dry_run_success() -> None:
    """Test type_infer_dry_run with valid syntax."""
    source = "x = 1\n"
    res = type_infer_dry_run(source)
    assert res == {"dry_run": "success"}


def test_type_infer_dry_run_failure() -> None:
    """Test type_infer_dry_run with invalid syntax."""
    source = "x = 1\n\ninvalid code"
    res = type_infer_dry_run(source)
    assert res == {"dry_run": "failed"}


def test_cst_transformer_init() -> None:
    """Test CSTTransformer initialization."""
    transformer = CSTTransformer()
    assert transformer.target_framework == "jax"
