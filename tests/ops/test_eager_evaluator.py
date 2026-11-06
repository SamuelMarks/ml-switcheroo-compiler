"""Test eager_evaluator.py."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.eager_evaluator import EvaluationContext, EvaluationStrategy
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_evaluation_strategy_base() -> None:
    """Test base class."""

    class DummyStrategy(EvaluationStrategy):
        """Test class DummyStrategy."""

        def evaluate(self, ctx: EagerValue) -> EagerValue:
            """Test function evaluate.

            Args:
                ctx: Test parameter `ctx`.
            """
            return ensure_callable(ensure_eager_tensor(ensure_eager_tensor(super()).evaluate))(ctx)

    ctx = ensure_callable(EvaluationContext)(op_cls=None, op_type="foo", args=[], kwargs={}, backend=None)
    with pytest.raises(NotImplementedError):
        DummyStrategy().evaluate(ctx)
