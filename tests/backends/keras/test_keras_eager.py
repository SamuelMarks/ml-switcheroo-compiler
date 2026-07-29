"""Test module."""

from ml_switcheroo_compiler.backends.keras.eager import execute_op


class DummyKerasOps:
    def add(self, *a, **k):
        return "add"


def test_keras_eager():
    import pytest

    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with pytest.raises(BackendNotSupportedError):
        execute_op(None, "UnknownKerasOpThatDoesNotExist")
