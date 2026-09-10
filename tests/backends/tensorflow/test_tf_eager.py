"""Tests for TensorFlow eager execution dispatch."""

import builtins
import sys
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.tensorflow.eager import execute_op
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def test_tf_not_installed() -> None:
    """Test execute_op when TensorFlow is not installed."""
    orig_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("No module named tensorflow")
        return orig_import(name, *args, **kwargs)

    clean_modules = {k: v for k, v in sys.modules.items() if k != "tensorflow"}
    with patch.dict(sys.modules, clean_modules, clear=True):
        with patch("builtins.__import__", side_effect=failing_import):
            with pytest.raises(BackendNotSupportedError, match="TensorFlow is not installed"):
                execute_op(None, "Add")


def test_tf_eager_execute_op() -> None:
    """Test execute_op with mapped operations and arguments."""
    mock_tf = MagicMock()
    mock_tf.convert_to_tensor.side_effect = lambda x: f"tensor_{x}"

    with patch.dict(sys.modules, {"tensorflow": mock_tf}):
        mock_tf.add.return_value = "tf_add_res"
        res = execute_op(None, "Add", [1, 2], 3)
        assert res == "tf_add_res"

        # Test exception during convert_to_tensor
        mock_tf.convert_to_tensor.side_effect = RuntimeError("cannot convert")
        res = execute_op(None, "Add", [1, 2], 3)
        assert res == "tf_add_res"


def test_tf_eager_execute_op_fallback() -> None:
    """Test execute_op fallback to eager registry and failure."""
    mock_tf = MagicMock()

    with patch.dict(sys.modules, {"tensorflow": mock_tf}):
        # Registered func in global_eager_registry
        def dummy_func(module, *args, **kwargs):
            return "dummy_res"

        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=dummy_func):
            res = execute_op(None, "UnknownOp")
            assert res == "dummy_res"

        # Unknown op with no registration
        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
            with pytest.raises(BackendNotSupportedError, match="not supported by tensorflow"):
                execute_op(None, "OpThatDoesNotExist")
