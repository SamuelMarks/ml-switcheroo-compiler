"""Coverage tests for CST transpiler."""

from ml_switcheroo.backends.cst_transpiler import transpile_source


def test_transpile_non_torch_import() -> None:
    """Docstring."""
    src = "from os import path\n"
    res = transpile_source(src, target_framework="jax")
    assert "os" in res


def test_transpile_non_attribute_call() -> None:
    """Docstring."""
    src = "print('hello')\n"
    res = transpile_source(src, target_framework="jax")
    assert "print" in res
