"""Test eager_evaluator.py."""

from ml_switcheroo_compiler.ops.eager_evaluator import EvaluationContext, EvaluationStrategy


def test_evaluation_strategy_base():
    """Test base class."""

    class DummyStrategy(EvaluationStrategy):
        def evaluate(self, ctx):
            return super().evaluate(ctx)

    ctx = EvaluationContext(op_cls=None, op_type="foo", raw_args=[], kwargs={}, backend=None)
    assert DummyStrategy().evaluate(ctx) is None
