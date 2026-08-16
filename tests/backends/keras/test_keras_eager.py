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


def test_keras_eager_extra():
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.backends.keras.eager import _get_op_mapping, execute_op

    # Call it to hit the cached branch
    _get_op_mapping()
    _get_op_mapping()

    # global_eager_registry
    @global_eager_registry.register("KerasGlobalDummy")
    def _dummy_keras(keras_module, *args, **kwargs):
        return "keras_global"

    res = execute_op(None, "KerasGlobalDummy")
    assert res == "keras_global"
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.keras.eager import execute_op

    with patch("ml_switcheroo_compiler.backends.keras.eager._get_op_mapping") as mock_get_mapping:
        mock_get_mapping.return_value = {"TestOp2": lambda *args, **kwargs: "mapped_res"}
        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
            res = execute_op(None, "TestOp2")
            assert res == "mapped_res"
