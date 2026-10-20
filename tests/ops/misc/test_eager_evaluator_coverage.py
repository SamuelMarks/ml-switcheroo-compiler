"""Tests for test_eager_evaluator_coverage."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.eager_evaluator as eager_eval_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_eager_evaluator_full_coverage() -> None:
    """Test full coverage for EagerEvaluator strategies and output packing."""

    class StrategySubclass(eager_eval_mod.EvaluationStrategy):
        """Concrete subclass testing base abstract method call."""

        def evaluate(self, ctx: eager_eval_mod.EvaluationContext) -> Tensor | None:
            """Call super evaluate to trigger NotImplementedError.

            Args:
                ctx (eager_eval_mod.EvaluationContext): Evaluation context.

            Returns:
                Tensor | None: Result.
            """
            return typing.cast("Tensor | None", typing.cast("EagerTensorProtocol", getattr(super(), "evaluate"))(ctx))

    dummy_ctx = eager_eval_mod.EvaluationContext(
        op_cls=OpDef,
        op_type="DummyOp",
        args=[],
        kwargs={},
        backend=MagicMock,
    )
    with pytest.raises(NotImplementedError):
        StrategySubclass().evaluate(dummy_ctx)

    class CustomEvalOp:
        """Mock op implementing eager_eval."""

        def eager_eval(self, *args: int, **kwargs: int) -> int:
            """Execute eager evaluation.

            Args:
                *args (int): Positional integers.
                **kwargs (int): Keyword integers.

            Returns:
                int: Result sum.
            """
            return sum(args) + sum(kwargs.values())

    class CustomForwardOp:
        """Mock op implementing forward."""

        def forward(self, *args: int, **kwargs: int) -> int:
            """Execute forward evaluation.

            Args:
                *args (int): Positional integers.
                **kwargs (int): Keyword integers.

            Returns:
                int: Result product.
            """
            return 100

    class NoCustomEvalOp:
        """Mock op without eager_eval or forward."""

    custom_strat = eager_eval_mod.CustomEagerEvalStrategy()
    ctx_eager = eager_eval_mod.EvaluationContext(op_cls=CustomEvalOp, op_type="CustomEvalOp", args=[1, 2], kwargs={"c": 3}, backend=MagicMock)
    assert custom_strat.evaluate(ctx_eager) == 6

    ctx_forward = eager_eval_mod.EvaluationContext(op_cls=CustomForwardOp, op_type="CustomForwardOp", args=[], kwargs={}, backend=MagicMock)
    assert custom_strat.evaluate(ctx_forward) == 100

    ctx_none = eager_eval_mod.EvaluationContext(op_cls=NoCustomEvalOp, op_type="NoCustomEvalOp", args=[], kwargs={}, backend=MagicMock)
    with pytest.raises(NotImplementedError, match="does not implement custom eager evaluation"):
        custom_strat.evaluate(ctx_none)

    backend_strat = eager_eval_mod.BackendExecuteOpStrategy()
    mock_be = MagicMock()
    mock_be.execute_op.return_value = 888
    ctx_be = eager_eval_mod.EvaluationContext(op_cls=OpDef, op_type="Add", args=[10, 20], kwargs={}, backend=mock_be)
    assert backend_strat.evaluate(ctx_be) == 888

    strat_custom = eager_eval_mod.EagerEvaluator._get_strategy(CustomEvalOp)
    assert isinstance(typing.cast("EagerValue", strat_custom), eager_eval_mod.CustomEagerEvalStrategy)

    strat_be = eager_eval_mod.EagerEvaluator._get_strategy(OpDef)
    assert isinstance(typing.cast("EagerValue", strat_be), eager_eval_mod.BackendExecuteOpStrategy)

    t_in = Tensor(np.array([1.0, 2.0], dtype=np.float32), TensorConfig((2,), "float32", "cpu"))
    packed_tuple = typing.cast("EagerCallableProtocol", eager_eval_mod.EagerEvaluator._pack_outputs)(
        (typing.cast("EagerCallableProtocol", np.array)([1.0], dtype=np.float32), typing.cast("EagerCallableProtocol", DummyWithoutShape)()),
        t_in,
        "cpu",
    )
    assert isinstance(typing.cast("EagerValue", packed_tuple), tuple)
    assert typing.cast("EagerValue", len(packed_tuple)) == 2

    packed_single_noshape = typing.cast("EagerCallableProtocol", eager_eval_mod.EagerEvaluator._pack_outputs)(
        typing.cast("EagerCallableProtocol", DummyWithoutShape)(),
        t_in,
        "cpu",
    )
    assert isinstance(typing.cast("EagerValue", packed_single_noshape), Tensor)
    assert packed_single_noshape.shape == ()

    with (
        patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_active_be,
        patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=OpDef),
    ):
        mock_active_be.return_value.execute_op.return_value = np.array([42.0], dtype=np.float32)
        res_no_tensor_input = eager_eval_mod.EagerEvaluator.evaluate("Add", 10, 20)
        assert isinstance(res_no_tensor_input, Tensor)
