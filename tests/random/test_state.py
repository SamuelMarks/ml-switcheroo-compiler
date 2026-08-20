"""Tests for state.py."""

import sys
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.random.state import (
    _dispatch_random,
    _dispatch_random_eager,
    _emit_random_node,
    _get_numpy_rng,
    rng_uniform,
)


def test_emit_random_node_eager() -> None:
    """Test _emit_random_node in eager mode."""
    config.eager_mode = True
    with patch.object(sys.modules["ml_switcheroo_compiler.random.state"], "_dispatch_random_eager") as mock_dispatch:
        mock_dispatch.return_value = 42

        inp = MagicMock(spec=Tensor)
        result = _emit_random_node("RandomOp", [inp], (2,), dtypes.DType.Float32, {"attr1": "val1"})

        assert isinstance(result, Tensor)
        assert result.data == 42
        mock_dispatch.assert_called_once_with("randomop", "RandomOp", inp, attr1="val1", shape=(2,), dtype="float32")


def test_emit_random_node_tracing() -> None:
    """Test _emit_random_node in tracing mode."""
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.random.state.global_tracing_state") as mock_global_state, patch("ml_switcheroo_compiler.random.state.uuid.uuid4") as mock_uuid:
        mock_uuid.return_value = "mock-uuid"

        inp = MagicMock()
        inp.data.id = "inp-id"
        result = _emit_random_node("RandomOp", [inp], (2,), dtypes.DType.Float32, {"attr1": "val1"})

        assert isinstance(result, Tensor)
        assert result.data.id == "mock-uuid"
        assert result.data.shape == (2,)
        assert result.data.dtype == "float32"
        mock_global_state.add_node.assert_called_once()

        node = mock_global_state.add_node.call_args[0][0]
        assert node.id == "mock-uuid"
        assert node.op_type == "RandomOp"
        assert node.inputs == ["inp-id"]
        assert node.attributes == {"attr1": "val1"}
        assert node.shape_metadata == (2,)


def test_emit_random_node_tracing_no_attrs() -> None:
    """Test _emit_random_node in tracing mode with no attributes."""
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.random.state.global_tracing_state") as mock_global_state, patch("ml_switcheroo_compiler.random.state.uuid.uuid4") as mock_uuid:
        mock_uuid.return_value = "mock-uuid"

        result = _emit_random_node("RandomOp", [], (2,), dtypes.DType.Float32)

        assert isinstance(result, Tensor)
        node = mock_global_state.add_node.call_args[0][0]
        assert node.attributes == {}


def test_dispatch_random_eager() -> None:
    """Test _dispatch_random_eager."""
    with patch("ml_switcheroo_compiler.random.state.get_active_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = "mocked_result"
        mock_get_backend.return_value = mock_backend

        result = _dispatch_random_eager("func_name", "OpName", "arg1", kwarg1="val1")
        assert result == "mocked_result"
        mock_backend.execute_op.assert_called_once_with("OpName", "arg1", kwarg1="val1")


def test_dispatch_random_eager_mode() -> None:
    """Test _dispatch_random when eager mode is True."""
    config.eager_mode = True
    with patch.object(sys.modules["ml_switcheroo_compiler.random.state"], "_dispatch_random_eager") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"

        result = _dispatch_random("my_op_func", "arg1", kwarg1="val1")
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("my_op_func", "MyOpFunc", "arg1", kwarg1="val1")


def test_dispatch_random_tracing_has_op() -> None:
    """Test _dispatch_random when tracing and op exists."""
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.random.state.get_op") as mock_get_op:
        mock_op_cls = MagicMock()
        mock_op_instance = MagicMock()
        mock_op_instance.return_value = "mocked_result"
        mock_op_cls.return_value = mock_op_instance
        mock_get_op.return_value = mock_op_cls

        result = _dispatch_random("my_op_func", "arg1", kwarg1="val1")
        assert result == "mocked_result"
        mock_get_op.assert_called_once_with("MyOpFunc")
        mock_op_cls.assert_called_once()
        mock_op_instance.assert_called_once_with("arg1", kwarg1="val1")


def test_dispatch_random_tracing_no_op() -> None:
    """Test _dispatch_random when tracing and op does not exist."""
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.random.state.get_op", return_value=None), patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node") as mock_emit:
        mock_emit.return_value = "mocked_result"

        result = _dispatch_random("my_op_func", "arg1", kwarg1="val1")
        assert result == "mocked_result"
        mock_emit.assert_called_once_with("MyOpFunc", ["arg1"], {"kwarg1": "val1"}, (), "float32")


def test_rng_uniform() -> None:
    """Test rng_uniform function."""
    with patch.object(sys.modules["ml_switcheroo_compiler.random.state"], "_dispatch_random") as mock_dispatch:
        mock_dispatch.return_value = "mocked_result"

        result = rng_uniform("a", "b", (2,), dtypes.DType.Float64)
        assert result == "mocked_result"
        mock_dispatch.assert_called_once_with("rng_uniform", "a", "b", shape=(2,), dtype=dtypes.DType.Float64)


def test_get_numpy_rng() -> None:
    """Test _get_numpy_rng function."""
    with patch("ml_switcheroo_compiler.backends.registry.BackendRegistry.get") as mock_get:
        mock_backend_cls = MagicMock()
        mock_backend_cls.get_numpy_rng.return_value = "mocked_rng"
        mock_get.return_value = mock_backend_cls

        result = _get_numpy_rng("arg1", kwarg1="val1")
        assert result == "mocked_rng"
        mock_get.assert_called_once_with("numpy")
        mock_backend_cls.get_numpy_rng.assert_called_once_with("arg1", kwarg1="val1")


def test_emit_random_node_eager_with_shape_dtype() -> None:
    """Test _emit_random_node in eager mode with shape and dtype in attrs."""
    config.eager_mode = True
    with patch.object(sys.modules["ml_switcheroo_compiler.random.state"], "_dispatch_random_eager") as mock_dispatch:
        mock_dispatch.return_value = 42

        inp = MagicMock(spec=Tensor)
        result = _emit_random_node("RandomOp", [inp], (2,), dtypes.DType.Float32, {"shape": (2,), "dtype": "float32"})

        assert isinstance(result, Tensor)
        assert result.data == 42
        mock_dispatch.assert_called_once_with("randomop", "RandomOp", inp, shape=(2,), dtype="float32")
