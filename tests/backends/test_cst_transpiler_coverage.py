"""Unit and coverage tests for the Concrete Syntax Tree (CST) transpiler.

This module contains test cases to verify that the CST transpiler correctly handles and
preserves non-PyTorch code elements, such as standard library imports and built-in
function calls, during the transpilation process.
"""

from ml_switcheroo.backends.cst_transpiler import transpile_source


def test_transpile_non_torch_import() -> None:
    """Verify that the transpiler preserves non-PyTorch imports.

    This test ensures that standard library imports (e.g., `from os import path`)
    remain intact and are not modified when transpiling source code to JAX.
    """
    src = "from os import path\n"
    res = transpile_source(src, target_framework="jax")
    assert "os" in res


def test_transpile_non_attribute_call() -> None:
    """Verify that the transpiler preserves standard, non-attribute function calls.

    This test ensures that built-in or global function calls (e.g.,
    `print('hello')`)
    are correctly processed and preserved when transpiling source code to JAX.
    """
    src = "print('hello')\n"
    res = transpile_source(src, target_framework="jax")
    assert "print" in res
