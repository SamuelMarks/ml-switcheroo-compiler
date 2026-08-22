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
    from ml_switcheroo_compiler.backends.keras.eager import execute_op

    # global_eager_registry
    @global_eager_registry.register("KerasGlobalDummy")
    def _dummy_keras(keras_module, *args, **kwargs):
        return "keras_global"

    res = execute_op(None, "KerasGlobalDummy")
    assert res == "keras_global"
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.keras.eager import execute_op

    mock_schema = type("Dummy", (), {"operations": {"TestOp2": type("DummyOp", (), {"target_api": "custom_op", "custom_code": "lambda *args, **kwargs: 'mapped_res'"})()}})()
    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings", return_value=mock_schema):
        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
            res = execute_op(None, "TestOp2")
            assert res == "mapped_res"
