"""Behavioral unit tests for linear ops."""

from __future__ import annotations

import sys
import typing
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.linear_ops as linear
import ml_switcheroo_compiler.ops.nn.linear_ops as linear_ops_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from tests.testing_utils import extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg

gru_mod = sys.modules["ml_switcheroo_compiler.ops.nn.gru"]


def test_linear_ops_coverage() -> None:
    """Test function test_linear_ops_coverage."""
    config.eager_mode = True
    t_in = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    w = Tensor(np.ones((3, 2)), TensorConfig(shape=(3, 2), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.ones((3,)), TensorConfig(shape=(3,), dtype=DType("float32"), device=Device("cpu")))

    def mock_matmul(*args: BackendArg) -> BackendArg:
        """Test function mock_matmul.

        Args:
            *args (BackendArg): Test parameter `args`.

        Returns:
            BackendArg: Computed output value from `mock_matmul`.
        """
        return "mm"

    def mock_add(*args: BackendArg) -> BackendArg:
        """Test function mock_add.

        Args:
            *args (BackendArg): Test parameter `args`.

        Returns:
            BackendArg: Computed output value from `mock_add`.
        """
        return "add"

    with (
        patch.object(linear, "matmul", mock_matmul),
        patch.object(linear, "add", mock_add),
    ):
        assert linear.linear(t_in, w) == "mm"
        assert linear.linear(t_in, w, b) == "add"

    t_in1 = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t_in2 = Tensor(np.ones((1, 4)), TensorConfig(shape=(1, 4), dtype=DType("float32"), device=Device("cpu")))
    w_bi = Tensor(np.ones((3, 2, 4)), TensorConfig(shape=(3, 2, 4), dtype=DType("float32"), device=Device("cpu")))

    def mock_einsum(*args: BackendArg) -> BackendArg:
        """Test function mock_einsum.

        Args:
            *args (BackendArg): Test parameter `args`.

        Returns:
            BackendArg: Computed output value from `mock_einsum`.
        """
        return "es"

    with (
        patch.object(linear, "einsum", mock_einsum),
        patch.object(linear, "add", mock_add),
    ):
        assert linear.bilinear(t_in1, t_in2, w_bi) == "es"
        assert linear.bilinear(t_in1, t_in2, w_bi, b) == "add"


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
        assert extract_test_shape(res_no_bias) == (2, 3)

        res_with_bias = linear_ops_mod.linear(inp, weight, bias=bias)
        assert res_with_bias is not None
        assert extract_test_shape(res_with_bias) == (2, 3)

        inp1 = Tensor(np.ones((2, 3), dtype=np.float32), TensorConfig((2, 3), "float32", "cpu"))
        inp2 = Tensor(np.ones((2, 4), dtype=np.float32), TensorConfig((2, 4), "float32", "cpu"))
        bi_weight = Tensor(np.ones((5, 3, 4), dtype=np.float32), TensorConfig((5, 3, 4), "float32", "cpu"))
        bi_bias = Tensor(np.ones((5,), dtype=np.float32), TensorConfig((5,), "float32", "cpu"))
        mock_einsum_out = Tensor(np.ones((2, 5), dtype=np.float32), TensorConfig((2, 5), "float32", "cpu"))

        with patch("ml_switcheroo_compiler.ops.nn.linear_ops.einsum", return_value=mock_einsum_out):
            bi_no_bias = linear_ops_mod.bilinear(inp1, inp2, bi_weight, bias=None)
            assert bi_no_bias is not None
            assert extract_test_shape(bi_no_bias) == (2, 5)

            bi_with_bias = linear_ops_mod.bilinear(inp1, inp2, bi_weight, bias=bi_bias)
            assert bi_with_bias is not None
            assert extract_test_shape(bi_with_bias) == (2, 5)
    finally:
        config.eager_mode = original_eager
