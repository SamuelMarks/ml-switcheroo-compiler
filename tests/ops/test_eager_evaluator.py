"""Test eager_evaluator.py."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.eager_evaluator import EvaluationContext, EvaluationStrategy

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_evaluation_strategy_base() -> None:
    """Test base class."""

    class DummyStrategy(EvaluationStrategy):
        """Test class DummyStrategy."""

        def evaluate(self, ctx: EvaluationContext) -> BackendArg:
            """Test function evaluate.

            Args:
                ctx (EvaluationContext): Test parameter `ctx`.

            Returns:
                BackendArg: Computed output value from `evaluate`.
            """
            return EvaluationStrategy.evaluate(self, ctx)

    ctx = EvaluationContext(op_cls=None, op_type="foo", args=[], kwargs={}, backend=None)
    with pytest.raises(NotImplementedError):
        DummyStrategy().evaluate(ctx)
