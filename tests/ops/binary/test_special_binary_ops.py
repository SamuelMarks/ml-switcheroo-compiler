"""Tests for test_special_binary_ops."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.binary.special as special_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


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


def test_special_binary_ops_coverage() -> None:
    """Test full branch coverage for special binary operations."""
    atan2_op = special_mod.Atan2()
    assert atan2_op.infer_shape((2, 3), (3,)) == (2, 3)
    assert typing.cast("EagerTensorProtocol", atan2_op.infer_shape([2, 3])) == typing.cast("EagerTensorProtocol", [2, 3])

    divmod_op = special_mod.Divmod()
    assert divmod_op.infer_shape((4, 4), (4, 4)) == (4, 4)
    assert typing.cast("EagerTensorProtocol", divmod_op.infer_shape([4, 4])) == typing.cast("EagerTensorProtocol", [4, 4])

    t1 = Tensor(np.array([4.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    t2 = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))

    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        with patch.object(EagerEvaluator, "evaluate", return_value=(t1, t2)) as mock_eval:
            res_eager = divmod_op(t1, t2)
            assert typing.cast("EagerTensorProtocol", res_eager) == typing.cast("EagerTensorProtocol", (t1, t2))
            mock_eval.assert_called_once_with("Divmod", t1, t2)

        config.eager_mode = False
        with (
            patch("ml_switcheroo_compiler.ops.binary.floor_divide", return_value=t1) as mock_fd,
            patch("ml_switcheroo_compiler.ops.binary.remainder", return_value=t2) as mock_rem,
        ):
            res_symbolic = divmod_op(t1, t2)
            assert typing.cast("EagerTensorProtocol", res_symbolic) == typing.cast("EagerTensorProtocol", (t1, t2))
            mock_fd.assert_called_once_with(t1, t2)
            mock_rem.assert_called_once_with(t1, t2)
    finally:
        config.eager_mode = original_eager

    allclose_op = special_mod.Allclose()
    assert allclose_op.infer_shape((2, 2), (2, 2)) == ()

    isclose_op = special_mod.Isclose()
    assert isclose_op.infer_shape((2, 2), (2, 2)) == (2, 2)
    assert typing.cast("EagerTensorProtocol", isclose_op.infer_shape([2, 2])) == typing.cast("EagerTensorProtocol", [2, 2])
