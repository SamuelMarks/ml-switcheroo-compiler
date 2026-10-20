"""Tests for test_scaled_dot_product_attention_coverage."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.attention as attention_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


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


def test_attention_scaled_dot_product_coverage() -> None:
    """Test full coverage for attention module functions and classes."""
    op = attention_mod.ScaledDotProductAttention()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", DummyWithShape)((4, 8, 16)), None, None, None) == (4, 8, 16)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", DummyWithoutShape)(), None, None, None) == ()

    q = Tensor(np.ones((2, 4, 8), dtype=np.float32), TensorConfig((2, 4, 8), "float32", "cpu"))
    k = Tensor(np.ones((2, 4, 8), dtype=np.float32), TensorConfig((2, 4, 8), "float32", "cpu"))
    v = Tensor(np.ones((2, 4, 8), dtype=np.float32), TensorConfig((2, 4, 8), "float32", "cpu"))
    sf = Tensor(np.array(0.25, dtype=np.float32), TensorConfig((), "float32", "cpu"))

    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_be:
            backend = MagicMock()
            backend.execute_op.return_value = np.ones((2, 4, 8), dtype=np.float32)
            mock_be.return_value = backend

            res_t = attention_mod.scaled_dot_product_attention(q, k, v, sf)
            assert isinstance(res_t, Tensor)

            raw_arr = np.ones((2, 4, 8), dtype=np.float32)
            res_mixed = typing.cast("EagerCallableProtocol", attention_mod.scaled_dot_product_attention)(q, raw_arr, raw_arr, 0.25)
            assert isinstance(typing.cast("EagerValue", res_mixed), Tensor)

            class _NoShapeData:
                """Mock data without shape attribute."""

                def __init__(self) -> None:
                    """Initialize without shape."""
                    self.val: int = 1

            backend.execute_op.return_value = _NoShapeData()
            res_noshape = attention_mod.scaled_dot_product_attention(q, k, v, sf)
            assert isinstance(res_noshape, Tensor)
            assert res_noshape.shape == (2, 4, 8)

        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.nn.attention._emit_shape_node", return_value=q) as mock_emit:
            res_trace = attention_mod.scaled_dot_product_attention(q, k, v, sf)
            assert res_trace is q
            mock_emit.assert_called_once()
    finally:
        config.eager_mode = original_eager
