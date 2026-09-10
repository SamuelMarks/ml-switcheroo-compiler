"""Tests for Keras eager execution dispatch."""

import builtins
import sys
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.keras.eager import (
    _convert_arg,
    _get_keras_module,
    _try_dispatch_mapped_op,
    execute_op,
)
from ml_switcheroo_compiler.backends.mapping_loader import (
    BackendMappingSchema,
    OpMappingSchema,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def test_get_keras_module_branches() -> None:
    """Test branches in _get_keras_module."""
    # Case 1: keras already in sys.modules
    fake_keras = object()
    with patch.dict(sys.modules, {"keras": fake_keras}):
        assert _get_keras_module() is fake_keras

    # Case 2: keras not in sys.modules and import fails
    orig_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "keras":
            raise ImportError("no keras")
        return orig_import(name, *args, **kwargs)

    clean_modules = {k: v for k, v in sys.modules.items() if k != "keras"}
    with patch.dict(sys.modules, clean_modules, clear=True):
        with patch("builtins.__import__", side_effect=failing_import):
            assert _get_keras_module() is None


def test_convert_arg_branches() -> None:
    """Test branches in _convert_arg."""
    # Arg not list or tuple
    assert _convert_arg(42, None) == 42
    assert _convert_arg("hello", None) == "hello"

    # keras is None
    assert _convert_arg([1, 2, 3], None) == [1, 2, 3]

    # keras has no ops
    dummy_keras = type("DummyKeras", (), {})()
    assert _convert_arg([1, 2], dummy_keras) == [1, 2]

    # ops has convert_to_tensor that succeeds
    mock_ops = MagicMock()
    mock_ops.convert_to_tensor.return_value = "converted_tensor"
    dummy_keras.ops = mock_ops
    assert _convert_arg([1, 2], dummy_keras) == "converted_tensor"

    # ops has convert_to_tensor that raises Exception
    mock_ops.convert_to_tensor.side_effect = RuntimeError("failed to convert")
    assert _convert_arg([1, 2], dummy_keras) == [1, 2]


def test_try_dispatch_mapped_op_branches() -> None:
    """Test branches in _try_dispatch_mapped_op."""
    # Neither target_api nor custom_code
    op_empty = type("DummyOp", (), {"target_api": None, "custom_code": None})()
    assert _try_dispatch_mapped_op(op_empty, None, [], {}) is None

    # func is None or not callable
    op_non_callable = OpMappingSchema(target_api="some.target", custom_code=None)
    with patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=None):
        assert _try_dispatch_mapped_op(op_non_callable, None, [], {}) is None

    with patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=12345):
        assert _try_dispatch_mapped_op(op_non_callable, None, [], {}) is None

    # Function dispatch with kwarg translations
    called_args = []
    called_kwargs = {}

    def dummy_func(*args, **kwargs):
        called_args.extend(args)
        called_kwargs.update(kwargs)
        return "func_result"

    op_func = OpMappingSchema(
        target_api="keras.ops.add",
        custom_code=None,
        kwarg_translations={"dim": "axis"},
    )
    with patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=dummy_func):
        res = _try_dispatch_mapped_op(op_func, None, [1, 2], {"dim": -1, "keep": True})
        assert res == "func_result"
        assert called_args == [1, 2]
        assert called_kwargs == {"axis": -1, "keep": True}

    # is_method is True with args and callable method
    class MockTensor:
        def my_method(self, *args, **kwargs):
            return "method_called"

    mock_obj = MockTensor()
    op_method = OpMappingSchema(
        target_api="my_method",
        custom_code=None,
        is_method=True,
    )
    with patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=dummy_func):
        # Successful method dispatch
        res = _try_dispatch_mapped_op(op_method, None, [mock_obj, "extra"], {})
        assert res == "method_called"

        # is_method is True but args is empty
        res = _try_dispatch_mapped_op(op_method, None, [], {})
        assert res == "func_result"

        # is_method is True but args[0] does not have method or it is not callable
        res = _try_dispatch_mapped_op(op_method, None, [object()], {})
        assert res == "func_result"


def test_keras_eager_execute_op() -> None:
    """Test execute_op success and failure cases."""
    # Unknown op raises BackendNotSupportedError
    with pytest.raises(BackendNotSupportedError):
        execute_op(None, "UnknownKerasOpThatDoesNotExist")

    # Global eager registry fallback
    @global_eager_registry.register("KerasGlobalDummy")
    def _dummy_keras(keras_module, *args, **kwargs):
        return "keras_global"

    try:
        res = execute_op(None, "KerasGlobalDummy")
        assert res == "keras_global"
    finally:
        global_eager_registry._registry.pop("KerasGlobalDummy", None)

    # Dispatch mapped op returning result
    mock_schema = BackendMappingSchema(
        backend_name="keras",
        operations={"TestOp": OpMappingSchema(target_api="custom_op", custom_code="lambda *a, **k: 'mapped_ok'")},
    )
    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings", return_value=mock_schema):
        res = execute_op(None, "TestOp", [1, 2])
        assert res == "mapped_ok"

    # Mapped op where _try_dispatch_mapped_op returns None falls through
    mock_op_none = type("DummyOpNone", (), {"target_api": None, "custom_code": None})()
    mock_schema_none = type("DummySchemaNone", (), {"operations": {"TestOpNone": mock_op_none}})()
    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings", return_value=mock_schema_none):
        with pytest.raises(BackendNotSupportedError):
            execute_op(None, "TestOpNone")
