"""Tests for test_linear_ops_coverage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.linear_ops as linear_ops_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy object with shape.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize dummy object without shape."""
        self.val: int = 42


def test_linear_ops_full_coverage() -> None:
    """Test full coverage for linear and bilinear transformations."""
    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        inp = Tensor(np.ones((2, 4), dtype=np.float32), TensorConfig((2, 4), "float32", "cpu"))
        weight = Tensor(np.ones((3, 4), dtype=np.float32), TensorConfig((3, 4), "float32", "cpu"))
        bias = Tensor(np.ones((3,), dtype=np.float32), TensorConfig((3,), "float32", "cpu"))

        res_no_bias = linear_ops_mod.linear(inp, weight, bias=None)
        assert res_no_bias is not None
        assert res_no_bias.shape == (2, 3)

        res_with_bias = linear_ops_mod.linear(inp, weight, bias=bias)
        assert res_with_bias is not None
        assert res_with_bias.shape == (2, 3)

        inp1 = Tensor(np.ones((2, 3), dtype=np.float32), TensorConfig((2, 3), "float32", "cpu"))
        inp2 = Tensor(np.ones((2, 4), dtype=np.float32), TensorConfig((2, 4), "float32", "cpu"))
        bi_weight = Tensor(np.ones((5, 3, 4), dtype=np.float32), TensorConfig((5, 3, 4), "float32", "cpu"))
        bi_bias = Tensor(np.ones((5,), dtype=np.float32), TensorConfig((5,), "float32", "cpu"))
        mock_einsum_out = Tensor(np.ones((2, 5), dtype=np.float32), TensorConfig((2, 5), "float32", "cpu"))

        with patch("ml_switcheroo_compiler.ops.nn.linear_ops.einsum", return_value=mock_einsum_out):
            bi_no_bias = linear_ops_mod.bilinear(inp1, inp2, bi_weight, bias=None)
            assert bi_no_bias is not None
            assert bi_no_bias.shape == (2, 5)

            bi_with_bias = linear_ops_mod.bilinear(inp1, inp2, bi_weight, bias=bi_bias)
            assert bi_with_bias is not None
            assert bi_with_bias.shape == (2, 5)
    finally:
        config.eager_mode = original_eager
