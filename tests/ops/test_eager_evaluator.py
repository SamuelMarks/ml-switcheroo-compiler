"""Test eager_evaluator.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.ops.eager_evaluator import EvaluationContext, EvaluationStrategy

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_evaluation_strategy_base() -> None:
    """Test base class."""

    class DummyStrategy(EvaluationStrategy):
        """Test class DummyStrategy."""

        def evaluate(self, ctx: EagerValue) -> EagerValue:
            """Test function evaluate.

            Args:
                ctx: Test parameter `ctx`.
            """
            return cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(super(), "evaluate")))(ctx)

    ctx = cast("EagerCallableProtocol", EvaluationContext)(op_cls=None, op_type="foo", args=[], kwargs={}, backend=None)
    with pytest.raises(NotImplementedError):
        DummyStrategy().evaluate(ctx)
