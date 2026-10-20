"""Tests for test_linear_ops."""

from __future__ import annotations

# ruff: noqa: E501
import sys
import typing

import numpy as np

import ml_switcheroo_compiler.ops.nn.linear_ops as linear
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

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

    orig_matmul = typing.cast("EagerTensorProtocol", getattr(linear, "matmul"))
    orig_add = typing.cast("EagerTensorProtocol", getattr(linear, "add"))
    setattr(linear, "matmul", mock_matmul)
    setattr(linear, "add", mock_add)
    try:
        assert linear.linear(t_in, w) == "mm"
        assert linear.linear(t_in, w, b) == "add"
    finally:
        setattr(linear, "matmul", orig_matmul)
        setattr(linear, "add", orig_add)

    t_in1 = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_in2 = Tensor(np.ones((1, 4)), TensorConfig(shape=(1, 4), dtype=DType("float32"), device=Device("cpu")))
    w_bi = Tensor(np.ones((3, 2, 4)), TensorConfig(shape=(3, 2, 4), dtype=DType("float32"), device=Device("cpu")))

    def mock_einsum(*args: EagerValue) -> EagerValue:
        """Test function mock_einsum.

        Args:
            args: Test parameter `args`.
        """
        return "es"

    orig_einsum = typing.cast("EagerTensorProtocol", getattr(linear, "einsum"))
    orig_add2 = typing.cast("EagerTensorProtocol", getattr(linear, "add"))
    setattr(linear, "einsum", mock_einsum)
    setattr(linear, "add", mock_add)
    try:
        assert linear.bilinear(t_in1, t_in2, w_bi) == "es"
        assert linear.bilinear(t_in1, t_in2, w_bi, b) == "add"
    finally:
        setattr(linear, "einsum", orig_einsum)
        setattr(linear, "add", orig_add2)
