"""Test module."""

from unittest.mock import patch

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.keras.eager import execute_op


class DummyKerasOps:
    def add(self, *a, **k):
        return "add"


def test_keras_eager():
    def dummy_reg(mod, *a, **k):
        return "reg"

    global_eager_registry.register("TestKerasOp")(dummy_reg)
    assert execute_op(None, "TestKerasOp") == "reg"

    with patch("keras.ops.add", return_value="add", create=True):
        if "Add" in global_eager_registry._registry:
            del global_eager_registry._registry["Add"]
        assert execute_op(None, "Add", 1, 2) == "add"

        if "UnknownKeras" in global_eager_registry._registry:
            del global_eager_registry._registry["UnknownKeras"]
        res = execute_op(None, "UnknownKeras")
        assert getattr(res, "shape", None) == (1,) or res is None

        def exploding_zeros(*args):
            raise Exception("Boom")

        with patch("numpy.zeros", exploding_zeros):
            assert execute_op(None, "UnknownKeras") is None
