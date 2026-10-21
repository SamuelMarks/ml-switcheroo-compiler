"""Tests for test_lu."""

from __future__ import annotations

# ruff: noqa: E501
import sys
import typing
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuPivotsToPermutation, lu_factor, lu_pivots_to_permutation
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    pass


def test_lu_dummy() -> None:
    """Test function test_lu_dummy."""

    class MockTensor:
        """Test class MockTensor."""

        shape = (2, 3)

    t = MockTensor()
    assert as_callable(LuPivotsToPermutation)().infer_shape(t, permutation_size=4) == (2, 4)
    assert as_callable(LuPivotsToPermutation)().infer_shape(t) == (2, 0)
    assert LuPivotsToPermutation().infer_shape() == ()
    assert LuPivotsToPermutation().infer_shape(None) == ()

    from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuFactor

    # 1D tensor (rank < 2) returns (shape, shape)
    class Mock1DTensor:
        """Test class Mock1DTensor."""

        shape = (5,)

    assert as_callable(LuFactor)().infer_shape(as_callable(Mock1DTensor)()) == ((5,), (5,))

    # kwargs 'a' and 'input'
    assert as_callable(LuFactor)().infer_shape(a=as_callable(MockTensor)()) == ((2, 3), (2,))
    assert as_callable(LuFactor)().infer_shape(input=as_callable(MockTensor)()) == ((2, 3), (2,))


def test_lu_factor() -> None:
    """Test function test_lu_factor."""
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]))
        (res1, res2) = lu_factor(t)
        assert isinstance(res1, Tensor)
        assert isinstance(res2, Tensor)


def test_lu_pivots_to_permutation() -> None:
    """Test function test_lu_pivots_to_permutation."""
    input_data = np.array([0, 1], dtype=np.int32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "int32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([[1, 0], [0, 1]])
        res = lu_pivots_to_permutation(t, 2)
        assert isinstance(res, Tensor)


def test_lu_factor_trace() -> None:
    """Test function test_lu_factor_trace."""
    from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuFactor

    assert LuFactor().infer_shape() == ()
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = False
    lu_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.lu"]
    with patch.object(lu_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_lu", "mock_piv")
        (res1, res2) = lu_factor(t)
        assert as_tensor(res1) == as_tensor("mock_lu")
        assert as_tensor(res2) == as_tensor("mock_piv")

        # Non-tuple fallback branch
        mock_emit.return_value = "single_res"
        (res3, res4) = lu_factor(t)
        assert as_tensor(res3) == as_tensor("single_res")
        assert as_tensor(res4) == as_tensor("single_res")


def test_lu_pivots_trace() -> None:
    """Test function test_lu_pivots_trace."""
    input_data = np.array([0, 1], dtype=np.int32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "int32", "cpu"))
    config.eager_mode = False
    lu_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.lu"]
    with patch.object(lu_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_perm"
        res = lu_pivots_to_permutation(t, 2)
        assert as_tensor(res) == as_tensor("mock_perm")

        # Tuple branch
        mock_emit.return_value = ("tuple_perm",)
        res_tuple = lu_pivots_to_permutation(t, 2)
        assert as_tensor(res_tuple) == as_tensor("tuple_perm")
