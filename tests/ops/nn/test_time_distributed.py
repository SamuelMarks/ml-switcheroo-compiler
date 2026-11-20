"""Unit tests for TimeDistributed operation ensuring 100% line and branch coverage."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.nn.time_distributed as time_dist_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.nn.time_distributed import TimeDistributed, time_distributed
from tests.testing_utils import MockTensorStub

if typing.TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockEmptyShapeOp(OpDef):
    """Mock op returning empty shape for branch coverage."""

    def infer_shape(self, *args: BackendArg, **kwargs: BackendArg) -> tuple[int, ...]:
        """Return empty shape.

        Args:
            *args (BackendArg): Positional args.
            **kwargs (BackendArg): Keyword args.

        Returns:
            tuple[int, ...]: Empty tuple.
        """
        return ()


def test_time_distributed_shape_inference() -> None:
    """Verify shape inference across all branches of TimeDistributed."""
    td = TimeDistributed()

    # 1. x is None
    assert td.infer_shape(None) == ()

    # 2. Rank < 2 (1D tensor)
    t_1d = Tensor(None, TensorConfig((5,), "float32", "cpu"))
    assert td.infer_shape(t_1d) == (5,)

    # 3. Rank >= 2 without wrapped_op_name
    t_3d = Tensor(None, TensorConfig((2, 3, 4), "float32", "cpu"))
    assert td.infer_shape(t_3d) == (2, 3, 4)

    # 4. wrapped_op_name causes Exception (e.g. non-existent op)
    assert td.infer_shape(t_3d, wrapped_op_name="DefinitelyNonExistentOp123") == (2, 3, 4)

    # 5. wrapped_op_name valid and returns shape with len >= 1
    assert td.infer_shape(t_3d, wrapped_op_name="Relu") == (2, 3, 4)

    # 6. wrapped_op_name valid but inner shape is empty
    assert td.infer_shape(t_3d, wrapped_op_name="MockEmptyShapeOp") == (2, 3, 4)


def test_time_distributed_dispatch_eager(mocker: MockerFixture) -> None:
    """Verify eager dispatch calling TimeDistributed op in eager and graph mode.

    Args:
        mocker: Test parameter `mocker`.
    """
    import numpy as np

    data = np.zeros((2, 3, 4), dtype=np.float32)
    t_3d = Tensor(data, TensorConfig((2, 3, 4), "float32", "cpu"))

    # Eager mode
    config.eager_mode = True
    res = time_distributed(t_3d, wrapped_op_name="Relu")
    assert res is not None

    # Graph/tracing mode (config.eager_mode = False) executes the underlying decorated function
    config.eager_mode = False
    mock_op = mocker.patch("ml_switcheroo_compiler.ops.nn.time_distributed.get_op")
    mock_instance = mock_op.return_value.return_value
    pytest.MonkeyPatch().setattr(mock_instance, "return_value", "traced_result", raising=False)

    res_graph = time_distributed(t_3d, wrapped_op_name="Relu")
    assert res_graph == "traced_result"
    config.eager_mode = True


class DummyWithShape(MockTensorStub):
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy object with shape.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        super().__init__()
        self.shape = shape


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
    assert td_op.infer_shape(x=DummyWithShape((2, 3, 4)), wrapped_op_name=None) == (
        2,
        3,
        4,
    )
    assert td_op.infer_shape(x=DummyWithShape((2, 3, 4)), wrapped_op_name=123) == (
        2,
        3,
        4,
    )

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
