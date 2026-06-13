"""Module docstring."""

from ml_switcheroo_compiler.grad import backward, custom_jvp


def test_grad_brute_coverage() -> None:
    """Function docstring."""
    backward(None)

    def my_fun() -> None:
        pass

    assert custom_jvp(my_fun) == my_fun
