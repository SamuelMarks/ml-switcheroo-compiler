"""Tests for stateless random PRNG operations and Philox mixing."""

from __future__ import annotations

import typing
import unittest.mock as mock

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_stateless import (
    RandomGenerationConfig,
    index_shuffle,
    stateless_fold_in,
    stateless_parameterized_truncated_normal,
    stateless_split,
)


def create_eager_tensor(data: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> Tensor:
    """Create a CPU float32 Tensor from a NumPy array.

    Args:
        data (np.ndarray): Input array.

    Returns:
        Tensor: Wrapped compiler Tensor.
    """
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


def test_random_stateless_tracing() -> None:
    """Verify tracing node emission for stateless PRNG operations."""
    with ConfigContext(eager_mode=False):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
            with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
                seed_tensor = create_eager_tensor(np.array([42, 0]))
                res_split = stateless_split(seed_tensor, num=2)
                assert isinstance(res_split, Tensor)
                assert res_split.shape == (2, 2)

                res_trunc = stateless_parameterized_truncated_normal(2, seed_tensor, RandomGenerationConfig())
                assert isinstance(res_trunc, Tensor)
                assert res_trunc.shape == (2,)

                res_shuf = index_shuffle(seed_tensor, seed_tensor, 10)
                assert isinstance(res_shuf, Tensor)

                res_fold = stateless_fold_in(seed_tensor, 5)
                assert isinstance(res_fold, Tensor)


def test_random_stateless_eager_philox() -> None:
    """Verify eager counter-based Philox evaluation for stateless PRNG functions."""
    orig = config.eager_mode
    config.eager_mode = True
    try:
        seed_tensor = Tensor(np.array([10, 20], dtype=np.int64), TensorConfig((2,), DType.Int64, Device("cpu")))
        trunc = typing.cast(Tensor, stateless_parameterized_truncated_normal((2, 2), seed_tensor))
        assert trunc.shape == (2, 2)
        assert np.all(np.asarray(trunc.data) >= -2.0)
        assert np.all(np.asarray(trunc.data) <= 2.0)

        idx = index_shuffle(3, seed_tensor, 9)
        assert isinstance(idx, int)
        assert 0 <= idx <= 9

        idx_zero = index_shuffle(3, seed_tensor, 0)
        assert idx_zero == 3

        idx_tensor = Tensor(np.array([0, 1, 2], dtype=np.int32), TensorConfig((3,), DType.Int32, Device("cpu")))
        max_tensor = Tensor(np.array([5], dtype=np.int32), TensorConfig((1,), DType.Int32, Device("cpu")))
        idx_res = typing.cast(Tensor, index_shuffle(idx_tensor, [1, 2], max_tensor))
        assert idx_res.shape == (3,)

        folded_tensor = typing.cast(Tensor, stateless_fold_in(seed_tensor, 7))
        assert folded_tensor.shape == (2,)
        assert not np.array_equal(np.asarray(folded_tensor.data), np.array([10, 20]))

        folded_str = stateless_fold_in("my_seed", [[1, 2]])
        assert isinstance(folded_str, str)
        assert folded_str.startswith("my_seed_")

        folded_list = stateless_fold_in([5], None)
        assert isinstance(folded_list, list)
        assert len(folded_list) == 2
    finally:
        config.eager_mode = orig
