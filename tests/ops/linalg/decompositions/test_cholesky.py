"""Tests for test_cholesky."""

from __future__ import annotations

# ruff: noqa: E501
import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.cholesky import CholeskyEx, cholesky_ex

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_cholesky_ex_dummy() -> None:
    """Test function test_cholesky_ex_dummy."""
    assert CholeskyEx().infer_shape() == ()


def test_cholesky_ex() -> None:
    """Test function test_cholesky_ex."""
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array(0))
        mock_backend.return_value.array = lambda x: x
        (res1, res2) = cast("EagerCallableProtocol", cholesky_ex)(t, check_errors=True)
        assert isinstance(res1, Tensor)
        assert isinstance(res2, Tensor)
    config.eager_mode = False
    cholesky_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.cholesky"]
    with patch.object(cholesky_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_L", "mock_info")
        (res1, res2) = cast("EagerCallableProtocol", cholesky_ex)(t, check_errors=False)
        assert res1 == "mock_L"
        mock_emit.assert_called_once()


def test_cholesky() -> None:
    """Test function test_cholesky."""
    from ml_switcheroo_compiler.ops.linalg.decompositions.cholesky import Cholesky, cholesky

    assert Cholesky().infer_shape() == ()
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        mock_backend.return_value.array = lambda x: x
        res = cholesky(t)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    cholesky_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.cholesky"]
    with patch.object(cholesky_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_C"
        res = cholesky(t)
        assert res == "mock_C"
        mock_emit.assert_called_once()
