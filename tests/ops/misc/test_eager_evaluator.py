"""Behavioral unit tests for eager evaluator."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.eager_evaluator as eager_eval_mod
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.eager_evaluator import (
    BackendExecuteOpStrategy,
    CustomEagerEvalStrategy,
    EagerEvaluator,
    EvaluationContext,
)
from tests.testing_utils import MockTestStub, ensure_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_eager_evaluator() -> None:
    """Test function test_eager_evaluator."""

    class DummyOpEager:
        """Test class DummyOpEager."""

        def eager_eval(self, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function eager_eval.

            Args:
                *a (BackendArg): Test parameter `a`.
                **k (BackendArg): Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `eager_eval`.
            """
            return "custom_eager"

    class DummyOpBackend(OpDef):
        """Test class DummyOpBackend."""

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def execute_op(self, op: BackendArg, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op (BackendArg): Test parameter `op`.
                *a (BackendArg): Test parameter `a`.
                **k (BackendArg): Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return op

    ctx1 = EvaluationContext(DummyOpEager, "Op1", [], {}, DummyBackend())
    assert CustomEagerEvalStrategy().evaluate(ctx1) == "custom_eager"

    ctx2 = EvaluationContext(DummyOpBackend, "Op2", [], {}, DummyBackend())
    assert BackendExecuteOpStrategy().evaluate(ctx2) == "Op2"

    assert isinstance(EagerEvaluator._get_strategy(DummyOpEager), CustomEagerEvalStrategy)
    assert isinstance(EagerEvaluator._get_strategy(DummyOpBackend), BackendExecuteOpStrategy)

    t = Tensor(data="data", config=TensorConfig((1,), type("D", (), {"value": "float32"})(), "cpu"))

    res1 = EagerEvaluator._pack_outputs("raw_res", t, "cpu")
    assert isinstance(res1, Tensor)
    assert res1.data == "raw_res"

    # Check evaluate full
    from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY as OpRegistry

    class Op1(OpDef):
        """Test class Op1."""

        op_name = "Op1"

        def eager_eval(self, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function eager_eval.

            Args:
                *a (BackendArg): Test parameter `a`.
                **k (BackendArg): Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `eager_eval`.
            """
            return a[0] if a else None

        def infer_shape(self, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function infer_shape.

            Args:
                *a (BackendArg): Test parameter `a`.
                **k (BackendArg): Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `infer_shape`.
            """
            return ()

    try:
        OpRegistry["Op1"] = Op1
        OpRegistry["Op2"] = DummyOpBackend

        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend()):
            res2 = EagerEvaluator.evaluate("Op1", t)
            assert isinstance(res2, Tensor)
            assert res2.data == "data"
            res3 = EagerEvaluator.evaluate("Op2", t)
            assert isinstance(res3, Tensor)
            assert res3.data == "Op2"
    finally:
        OpRegistry.pop("Op1", None)
        OpRegistry.pop("Op2", None)


def test_eager_evaluator_pack_outputs() -> None:
    """Test function test_eager_evaluator_pack_outputs."""
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    t = Tensor(data="data", config=TensorConfig((1,), type("D", (), {"value": "float32"})(), "cpu"))

    class DummyData:
        """Test class DummyData."""

        shape = (1,)
        dtype = "float32"

    res = EagerEvaluator._pack_outputs((DummyData(),), t, "cpu")
    assert isinstance(res, tuple)
    assert len(res) == 1
    assert isinstance(res[0], Tensor)


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape(MockTestStub):
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
            return ensure_tensor(eager_eval_mod.EvaluationStrategy.evaluate(self, ctx))

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
    ctx_eager = eager_eval_mod.EvaluationContext(
        op_cls=CustomEvalOp, op_type="CustomEvalOp", args=[1, 2], kwargs={"c": 3}, backend=MagicMock
    )
    assert custom_strat.evaluate(ctx_eager) == 6

    ctx_forward = eager_eval_mod.EvaluationContext(
        op_cls=CustomForwardOp, op_type="CustomForwardOp", args=[], kwargs={}, backend=MagicMock
    )
    assert custom_strat.evaluate(ctx_forward) == 100

    ctx_none = eager_eval_mod.EvaluationContext(
        op_cls=NoCustomEvalOp, op_type="NoCustomEvalOp", args=[], kwargs={}, backend=MagicMock
    )
    with pytest.raises(NotImplementedError, match="does not implement custom eager evaluation"):
        custom_strat.evaluate(ctx_none)

    backend_strat = eager_eval_mod.BackendExecuteOpStrategy()
    mock_be = MagicMock()
    mock_be.execute_op.return_value = 888
    ctx_be = eager_eval_mod.EvaluationContext(op_cls=OpDef, op_type="Add", args=[10, 20], kwargs={}, backend=mock_be)
    assert backend_strat.evaluate(ctx_be) == 888

    strat_custom = eager_eval_mod.EagerEvaluator._get_strategy(CustomEvalOp)
    assert isinstance(strat_custom, eager_eval_mod.CustomEagerEvalStrategy)

    strat_be = eager_eval_mod.EagerEvaluator._get_strategy(OpDef)
    assert isinstance(strat_be, eager_eval_mod.BackendExecuteOpStrategy)

    t_in = Tensor(np.array([1.0, 2.0], dtype=np.float32), TensorConfig((2,), "float32", "cpu"))
    packed_tuple = eager_eval_mod.EagerEvaluator._pack_outputs(
        (np.array([1.0], dtype=np.float32), DummyWithoutShape()),
        t_in,
        "cpu",
    )
    assert isinstance(packed_tuple, tuple)
    assert len(packed_tuple) == 2

    packed_single_noshape = eager_eval_mod.EagerEvaluator._pack_outputs(
        DummyWithoutShape(),
        t_in,
        "cpu",
    )
    assert isinstance(packed_single_noshape, Tensor)
    assert packed_single_noshape.shape == ()

    with (
        patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_active_be,
        patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=OpDef),
    ):
        mock_active_be.return_value.execute_op.return_value = np.array([42.0], dtype=np.float32)
        res_no_tensor_input = eager_eval_mod.EagerEvaluator.evaluate("Add", 10, 20)
        assert isinstance(res_no_tensor_input, Tensor)
