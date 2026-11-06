"""Tests for test_linear_ops."""

from __future__ import annotations

# ruff: noqa: E501
import sys
import typing

import numpy as np

import ml_switcheroo_compiler.ops.nn.linear_ops as linear
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue

gru_mod = sys.modules["ml_switcheroo_compiler.ops.nn.gru"]


def test_linear_ops_coverage() -> None:
    """Test function test_linear_ops_coverage."""
    config.eager_mode = True
    t_in = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.ones((3, 2)), TensorConfig(shape=(3, 2), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.ones((3,)), TensorConfig(shape=(3,), dtype=DType("float32"), device=Device("cpu")))

    def mock_matmul(*args: EagerValue) -> EagerValue:
        """Test function mock_matmul.

        Args:
            args: Test parameter `args`.
        """
        return "mm"

    def mock_add(*args: EagerValue) -> EagerValue:
        """Test function mock_add.

        Args:
            args: Test parameter `args`.
        """
        return "add"

    orig_matmul = ensure_eager_tensor(ensure_eager_tensor(linear).matmul)
    orig_add = ensure_eager_tensor(ensure_eager_tensor(linear).add)
    ensure_mutable(linear).matmul = ensure_eager_tensor(mock_matmul)
    ensure_mutable(linear).add = ensure_eager_tensor(mock_add)
    try:
        assert linear.linear(t_in, w) == "mm"
        assert linear.linear(t_in, w, b) == "add"
    finally:
        ensure_mutable(linear).matmul = ensure_eager_tensor(orig_matmul)
        ensure_mutable(linear).add = ensure_eager_tensor(orig_add)

    t_in1 = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_in2 = Tensor(np.ones((1, 4)), TensorConfig(shape=(1, 4), dtype=DType("float32"), device=Device("cpu")))
    w_bi = Tensor(np.ones((3, 2, 4)), TensorConfig(shape=(3, 2, 4), dtype=DType("float32"), device=Device("cpu")))

    def mock_einsum(*args: EagerValue) -> EagerValue:
        """Test function mock_einsum.

        Args:
            args: Test parameter `args`.
        """
        return "es"

    orig_einsum = ensure_eager_tensor(ensure_eager_tensor(linear).einsum)
    orig_add2 = ensure_eager_tensor(ensure_eager_tensor(linear).add)
    ensure_mutable(linear).einsum = ensure_eager_tensor(mock_einsum)
    ensure_mutable(linear).add = ensure_eager_tensor(mock_add)
    try:
        assert linear.bilinear(t_in1, t_in2, w_bi) == "es"
        assert linear.bilinear(t_in1, t_in2, w_bi, b) == "add"
    finally:
        ensure_mutable(linear).einsum = ensure_eager_tensor(orig_einsum)
        ensure_mutable(linear).add = ensure_eager_tensor(orig_add2)
