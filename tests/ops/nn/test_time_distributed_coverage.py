"""Tests for test_time_distributed_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.time_distributed as time_dist_mod
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


def test_time_distributed_full_coverage() -> None:
    """Test full coverage for time distributed wrapper operations."""
    td_op = time_dist_mod.TimeDistributed()
    assert td_op.infer_shape(x=None) == ()
    assert td_op.infer_shape(x=DummyWithShape((5,))) == (5,)
    assert td_op.infer_shape(x=DummyWithShape((2, 3, 4)), wrapped_op_name=None) == (2, 3, 4)
    assert td_op.infer_shape(x=DummyWithShape((2, 3, 4)), wrapped_op_name=123) == (2, 3, 4)  # type: ignore[arg-type]

    class MockInnerOpSuccess:
        """Mock inner operation returning a valid shape."""

        def infer_shape(self, x: DummyWithShape, **kwargs: int) -> tuple[int, int]:
            """Infer output shape.

            Args:
                x (DummyWithShape): Input tensor placeholder.
                **kwargs (int): Additional arguments.

            Returns:
                tuple[int, int]: Output shape.
            """
            return (6, 8)

    class MockInnerOpEmptyShape:
        """Mock inner operation returning an empty shape."""

        def infer_shape(self, x: DummyWithShape, **kwargs: int) -> tuple[()]:
            """Infer empty output shape.

            Args:
                x (DummyWithShape): Input tensor placeholder.
                **kwargs (int): Additional arguments.

            Returns:
                tuple[()]: Empty tuple.
            """
            return ()

    with patch("ml_switcheroo_compiler.ops.nn.time_distributed.get_op") as mock_get_op:
        mock_get_op.return_value = MockInnerOpSuccess
        assert td_op.infer_shape(x=DummyWithShape((2, 3, 4)), wrapped_op_name="MockSuccess") == (2, 3, 8)

        mock_get_op.return_value = MockInnerOpEmptyShape
        assert td_op.infer_shape(x=DummyWithShape((2, 3, 4)), wrapped_op_name="MockEmpty") == (2, 3, 4)

        mock_get_op.side_effect = RuntimeError("Operation not found")
        assert td_op.infer_shape(x=DummyWithShape((2, 3, 4)), wrapped_op_name="FailingOp") == (2, 3, 4)

    t_in = Tensor(np.ones((2, 3, 4), dtype=np.float32), TensorConfig((2, 3, 4), "float32", "cpu"))

    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_be:
            backend = MagicMock()
            backend.array.side_effect = lambda x: x
            backend.execute_op.return_value = np.ones((2, 3, 4), dtype=np.float32)
            mock_be.return_value = backend

            res_single = time_dist_mod.time_distributed(t_in, wrapped_op_name="Dense")
            assert isinstance(res_single, Tensor)

            backend.execute_op.return_value = (
                np.ones((2, 3, 4), dtype=np.float32),
                np.ones((2, 3, 4), dtype=np.float32),
            )
            res_tuple = time_dist_mod.time_distributed(t_in, wrapped_op_name="Split")
            assert isinstance(res_tuple, tuple)
            assert len(res_tuple) == 2

        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.nn.time_distributed.get_op") as mock_get_op:
            mock_op_instance = MagicMock()
            mock_op_instance.return_value = t_in
            mock_get_op.return_value = MagicMock(return_value=mock_op_instance)

            res_symbolic = time_dist_mod.time_distributed(t_in, wrapped_op_name="Dense")
            assert res_symbolic is t_in
    finally:
        config.eager_mode = original_eager
