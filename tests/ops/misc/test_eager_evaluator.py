"""Tests for test_eager_evaluator."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.eager_evaluator import BackendExecuteOpStrategy, CustomEagerEvalStrategy, EagerEvaluator, EvaluationContext
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_eager_evaluator() -> None:
    """Test function test_eager_evaluator."""

    class DummyOpEager:
        """Test class DummyOpEager."""

        def eager_eval(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eager_eval.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return "custom_eager"

    class DummyOpBackend:
        """Test class DummyOpBackend."""

        pass

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op: EagerValue, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op: Test parameter `op`.
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return op

    ctx1 = ensure_callable(EvaluationContext)(DummyOpEager, "Op1", [], {}, ensure_callable(DummyBackend)())
    assert ensure_callable(CustomEagerEvalStrategy)().evaluate(ctx1) == "custom_eager"

    ctx2 = ensure_callable(EvaluationContext)(DummyOpBackend, "Op2", [], {}, ensure_callable(DummyBackend)())
    assert ensure_callable(BackendExecuteOpStrategy)().evaluate(ctx2) == "Op2"

    assert isinstance(ensure_callable(EagerEvaluator._get_strategy)(ensure_callable(DummyOpEager)()), CustomEagerEvalStrategy)
    assert isinstance(ensure_callable(EagerEvaluator._get_strategy)(ensure_callable(DummyOpBackend)()), BackendExecuteOpStrategy)

    t = Tensor(data="data", config=TensorConfig((1,), type("D", (), {"value": "float32"})(), "cpu"))

    res1 = EagerEvaluator._pack_outputs("raw_res", t, "cpu")
    assert isinstance(res1, Tensor)
    assert res1.data == "raw_res"

    # Check evaluate full
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY as OpRegistry

    class Op1(OpDef):
        """Test class Op1."""

        op_name = "Op1"

        def eager_eval(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function eager_eval.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return a[0] if a else None

        def infer_shape(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function infer_shape.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return ()

    try:
        OpRegistry["Op1"] = Op1
        ensure_eager_tensor(OpRegistry)["Op2"] = DummyOpBackend

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
