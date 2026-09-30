"""Tests for test_matrix_coverage."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.creation.frontend_matrix as matrix_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import global_tracing_state

gru_mod = importlib.import_module("ml_switcheroo_compiler.ops.nn.gru")


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_frontend_matrix_coverage() -> None:
    """Verify 100% line and branch coverage for frontend_matrix operations."""
    orig_eager = config.eager_mode
    try:
        # eye & identity eager mode
        config.eager_mode = True
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = np.eye(3, dtype=np.float32)

        with patch("ml_switcheroo_compiler.ops.creation.frontend_matrix.get_active_backend", return_value=mock_backend):
            e1 = matrix_mod.eye(3)
            assert e1.shape == (3, 3)
            e2 = matrix_mod.eye(3, m=4, k=1, dtype=DType.Float64, device=Device("cpu"))
            assert e2.shape == (3, 4)

            # custom dtype without .value
            class DummyDtypeWithoutValue:
                """Mock dtype without value attribute."""

                name: str = "float32"

            e3 = matrix_mod.eye(2, dtype=DummyDtypeWithoutValue())
            assert e3.shape == (2, 2)

            # identity
            i1 = matrix_mod.identity(4)
            assert i1.shape == (4, 4)

            # _diag_eager with data having shape and raw dtype
            raw_input = np.array([[1.0, 2.0], [3.0, 4.0]])
            d1 = matrix_mod._diag_eager(raw_input, 0, Device("cpu"), None)
            assert isinstance(d1, Tensor)

            # _diag_eager with tensor input and data lacking shape
            mock_data_no_shape = 123.0
            mock_backend.execute_op.return_value = mock_data_no_shape
            d2 = matrix_mod._diag_eager(Tensor(raw_input, TensorConfig((2, 2), DType.Float32, Device("cpu"))), 0, Device("cpu"), DType.Float32)
            assert d2.shape == ()

            # diag in eager mode
            mock_backend.execute_op.return_value = np.array([1.0, 4.0])
            t_matrix = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), DType.Float32, Device("cpu")))
            res_diag_eager = matrix_mod.diag(t_matrix, 0)
            assert isinstance(res_diag_eager, Tensor)

        # eye non-eager mode
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.creation.frontend_matrix._emit_creation_node", return_value="emitted_eye"):
            res_graph_eye = matrix_mod.eye(3, 3)
            assert res_graph_eye == "emitted_eye"

        # diag non-eager mode
        # 1. invalid shape length (!= 2)
        t_1d = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, Device("cpu")))
        with pytest.raises(ValueError, match="diag requires a 1D or 2D tensor."):
            matrix_mod.diag(t_1d)

        # 2. not tracing
        with patch.object(global_tracing_state, "is_tracing", False):
            with pytest.raises(RuntimeError, match="Cannot emit diag node outside of a tracing context."):
                matrix_mod.diag(t_matrix)

        # 3. tracing is active
        with patch.object(global_tracing_state, "is_tracing", True):
            with patch.object(global_tracing_state, "add_node") as mock_add_node:
                # With input.data having id and dtype having value
                class MockDataWithId:
                    """Mock data payload with id."""

                    id: str = "tensor_input_id"

                t_traced = Tensor(MockDataWithId(), TensorConfig((3, 3), DType.Float32, Device("cpu")))
                res_traced = matrix_mod.diag(t_traced, 1)
                assert res_traced.shape == (2,)
                mock_add_node.assert_called_once()

                # With input having id directly and dtype lacking value
                class MockInputDirectId:
                    """Mock tensor object with id and custom dtype."""

                    id: str = "direct_id"
                    shape: tuple[int, ...] = (4, 4)
                    dtype: str = "float32"
                    device: None = None

                mock_add_node.reset_mock()
                res_traced2 = matrix_mod.diag(MockInputDirectId(), 0)
                assert res_traced2.shape == (4,)
                mock_add_node.assert_called_once()
    finally:
        config.eager_mode = orig_eager
