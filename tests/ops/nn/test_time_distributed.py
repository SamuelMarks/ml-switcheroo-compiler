"""Unit tests for TimeDistributed operation ensuring 100% line and branch coverage."""

from __future__ import annotations

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.nn.time_distributed import TimeDistributed, time_distributed


@register_op("MockEmptyShapeOp")
class MockEmptyShapeOp(OpDef):
    """Mock op returning empty shape for branch coverage."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Return empty shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

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


def test_time_distributed_dispatch_eager(mocker: pytest.MonkeyPatch) -> None:
    """Verify eager dispatch calling TimeDistributed op in eager and graph mode."""
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
    mock_instance.return_value = "traced_result"

    res_graph = time_distributed(t_3d, wrapped_op_name="Relu")
    assert res_graph == "traced_result"
    config.eager_mode = True
