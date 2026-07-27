"""Test module."""

from ml_switcheroo_compiler.export.aot import compile_function


def test_aot():
    def f(x):
        return x + 1

    c = compile_function(f)
    assert c(1) == 2
