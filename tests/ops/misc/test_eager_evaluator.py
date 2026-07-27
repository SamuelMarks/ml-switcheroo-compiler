from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.eager_evaluator import BackendExecuteOpStrategy, CustomEagerEvalStrategy, EagerEvaluator, EvaluationContext


def test_eager_evaluator():
    class DummyOpEager:
        def eager_eval(self, *a, **k):
            return "custom_eager"

    class DummyOpBackend:
        pass

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            return op

    ctx1 = EvaluationContext(DummyOpEager, "Op1", [], {}, DummyBackend())
    assert CustomEagerEvalStrategy().evaluate(ctx1) == "custom_eager"

    ctx2 = EvaluationContext(DummyOpBackend, "Op2", [], {}, DummyBackend())
    assert BackendExecuteOpStrategy().evaluate(ctx2) == "Op2"

    assert isinstance(EagerEvaluator._get_strategy(DummyOpEager()), CustomEagerEvalStrategy)
    assert isinstance(EagerEvaluator._get_strategy(DummyOpBackend()), BackendExecuteOpStrategy)

    t = Tensor(data="data", config=TensorConfig((1,), type("D", (), {"value": "float32"})(), "cpu"))

    res1 = EagerEvaluator._pack_outputs("raw_res", t, "cpu")
    assert isinstance(res1, Tensor)
    assert res1.data == "raw_res"

    # Check evaluate full
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY as OpRegistry

    class Op1(OpDef):
        op_name = "Op1"

        def eager_eval(self, *a, **k):
            return a[0] if a else None

        def infer_shape(self, *a, **k):
            return ()

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


def test_eager_evaluator_pack_outputs():
    from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator

    t = Tensor(data="data", config=TensorConfig((1,), type("D", (), {"value": "float32"})(), "cpu"))

    class DummyData:
        shape = (1,)
        dtype = "float32"

    res = EagerEvaluator._pack_outputs((DummyData(),), t, "cpu")
    assert isinstance(res, tuple)
    assert len(res) == 1
    assert isinstance(res[0], Tensor)
