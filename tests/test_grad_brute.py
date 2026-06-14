"""Provides required module functionality."""

from ml_switcheroo_compiler.grad import backward, custom_jvp


def test_grad_brute_coverage() -> None:
    """Execute the requested function."""
    backward(None)

    def my_fun() -> None:
        """Docstring."""
        pass

    assert custom_jvp(my_fun) == my_fun
