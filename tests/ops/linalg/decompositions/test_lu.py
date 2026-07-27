# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuPivotsToPermutation, lu_factor, lu_pivots_to_permutation


def test_lu_dummy() -> None:

    class MockTensor:
        shape = (2, 3)

    t = MockTensor()
    assert LuPivotsToPermutation().infer_shape(t, permutation_size=4) == (2, 4)


def test_lu_factor() -> None:
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]))
        (res1, res2) = lu_factor(t)
        assert isinstance(res1, Tensor)
        assert isinstance(res2, Tensor)


def test_lu_pivots_to_permutation() -> None:
    input_data = np.array([0, 1], dtype=np.int32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "int32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.array([[1, 0], [0, 1]])
        res = lu_pivots_to_permutation(t, 2)
        assert isinstance(res, Tensor)


def test_lu_factor_trace() -> None:
    from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuFactor

    assert LuFactor().infer_shape() == ()
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    config.eager_mode = False
    lu_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.lu"]
    with patch.object(lu_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_lu", "mock_piv")
        (res1, res2) = lu_factor(t)
        assert res1 == "mock_lu"
        assert res2 == "mock_piv"


def test_lu_pivots_trace() -> None:
    input_data = np.array([0, 1], dtype=np.int32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "int32", "cpu"))
    config.eager_mode = False
    lu_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.lu"]
    with patch.object(lu_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_perm"
        res = lu_pivots_to_permutation(t, 2)
        assert res == "mock_perm"
