from ml_switcheroo_compiler.ops.lax_ops import approx_max_k
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.tracer import _tracer
from ml_switcheroo_compiler.ops import lax_ops
import numpy as np
from unittest.mock import patch


def test_lax_ops_tracing():
    with ConfigContext(eager_mode=False):
        t1 = Tensor(None, TensorConfig((2,), "float32", None))
        t1._node = "dummy_node"

        t2 = Tensor(None, TensorConfig((2,), "float32", None))
        t2._node = "dummy_node2"

        class DummyGraph:
            def create_node(self, op_type):
                class Node:
                    def add_input(self, inp):
                        pass

                return Node()

            def add_node(self, node):
                pass

        _tracer.is_tracing = True
        _tracer.active_graph = DummyGraph()
        try:
            res = approx_max_k(t1, t2, 5)
            assert res is not None
        finally:
            _tracer.is_tracing = False
            _tracer.active_graph = None


def test_lax_ops_tracing_no_graph():
    with ConfigContext(eager_mode=False):
        t1 = Tensor(None, TensorConfig((2,), "float32", None))
        res = approx_max_k(t1)
        assert res is not None


def test_lax_ops_tracing_no_args():
    with ConfigContext(eager_mode=False):
        res = approx_max_k()
        assert res is None


def test_lax_ops_infer_shape():
    opdef = lax_ops._MockOpDef()
    t1 = Tensor(np.array([1, 2]), TensorConfig((2,), "float32", None))
    res = opdef.infer_shape(t1)
    assert res == (2,)
    res2 = opdef.infer_shape()
    assert res2 == ()


def test_lax_ops_eager():
    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.lax_ops.get_active_backend") as mock_backend:
            mock_backend.return_value.execute_op.return_value = "eager_result"
            res = approx_max_k()
            assert res == "eager_result"
