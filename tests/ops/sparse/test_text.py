import importlib.util

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.sparse.frontend import (
    smm,
    sparse_add,
    sparse_bincount,
    sparse_concat,
    sparse_cross_hashed,
    sparse_dense_matmul,
    sparse_expand_dims,
    sparse_eye,
    sparse_fill_empty_rows,
    sparse_map_values,
    sparse_mask,
    sparse_maximum,
    sparse_minimum,
    sparse_reduce_max,
    sparse_reduce_sum,
    sparse_reorder,
    sparse_reset_shape,
    sparse_reshape,
    sparse_retain,
    sparse_sampled_add,
    sparse_segment_mean,
    sparse_segment_sqrt_n,
    sparse_segment_sum,
    sparse_slice,
    sparse_softmax,
    sparse_split,
    sparse_to_dense,
    sparse_to_indicator,
    sparse_transpose,
)
from ml_switcheroo_compiler.ops.text.frontend import AsStringConfig, as_string, edit_distance, regex_full_match, regex_replace, string_join, string_length, string_lower, string_split, string_substr, string_to_number, string_upper
from ml_switcheroo_compiler.ops.text.frontend import lookup as f_lookup
from ml_switcheroo_compiler.ops.text.frontend import string_to_hash as f_string_to_hash
from ml_switcheroo_compiler.ops.text.frontend import text_vectorization as f_text_vectorization
from ml_switcheroo_compiler.ops.text.ops import ArrayRepr, ArrayStr, AsString, EditDistance, Hashing, IntegerLookup, Lookup, RegexFullMatch, RegexReplace, StringJoin, StringLength, StringLookup, StringLower, StringSplit, StringSubstr, StringToHash, StringToNumber, StringUpper, TextVectorization
from ml_switcheroo_compiler.ops.text.ops import lookup as o_lookup
from ml_switcheroo_compiler.ops.text.ops import string_to_hash as o_string_to_hash
from ml_switcheroo_compiler.ops.text.ops import text_vectorization as o_text_vectorization

spec = importlib.util.spec_from_file_location("text_py_mod", "src/ml_switcheroo_compiler/ops/text.py")
text_py_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text_py_mod)
CreateToken = text_py_mod.CreateToken
create_token = text_py_mod.create_token
import ml_switcheroo_compiler.backends.registry as registry_mod
from ml_switcheroo_compiler.tracing import global_tracing_state


class MockBackend:
    def execute_op(self, op_name, *args, **kwargs):
        if op_name == "StringSplit":
            return np.zeros(()), np.zeros(())
        return np.zeros(())

    def array(self, x):
        return np.array(x)


@pytest.fixture
def mock_backend(monkeypatch):
    monkeypatch.setattr(registry_mod, "get_active_backend", lambda: MockBackend())
    monkeypatch.setattr("ml_switcheroo_compiler.ops.text.frontend.get_active_backend", lambda: MockBackend())
    monkeypatch.setattr("ml_switcheroo_compiler.ops.text.ops.get_active_backend", lambda: MockBackend())


@pytest.fixture(params=[True, False])
def eager_mode(request, monkeypatch):
    monkeypatch.setattr(config, "eager_mode", request.param)
    monkeypatch.setattr(global_tracing_state, "is_tracing", not request.param)
    return request.param


def test_sparse_frontend(eager_mode, mock_backend, monkeypatch):
    class MockOp:
        def infer_shape(self, *args, **kwargs):
            return ()

    monkeypatch.setattr("ml_switcheroo_compiler.ops.sparse.frontend.get_op", lambda name: MockOp)
    monkeypatch.setattr(global_tracing_state, "add_node", lambda node: None)

    t = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.Float32, device="cpu"))

    ops = [
        sparse_bincount,
        sparse_cross_hashed,
        sparse_expand_dims,
        sparse_eye,
        sparse_fill_empty_rows,
        sparse_map_values,
        sparse_mask,
        sparse_maximum,
        sparse_minimum,
        sparse_reduce_max,
        sparse_reduce_sum,
        sparse_reorder,
        sparse_reset_shape,
        sparse_reshape,
        sparse_retain,
        sparse_segment_mean,
        sparse_segment_sqrt_n,
        sparse_segment_sum,
        sparse_slice,
        sparse_softmax,
        sparse_to_indicator,
        sparse_transpose,
        sparse_add,
        sparse_dense_matmul,
        sparse_sampled_add,
        smm,
        sparse_concat,
        sparse_split,
        sparse_to_dense,
    ]

    for op_func in ops:
        res = op_func(t, kwarg1=1)
        if eager_mode:
            assert hasattr(res, "data")
        else:
            assert res is not None


def test_text_frontend(eager_mode, mock_backend, monkeypatch):
    class MockNodeBuilder:
        @staticmethod
        def extract_proxy_inputs(inputs):
            return ["dummy_id"], None, None

    monkeypatch.setattr("ml_switcheroo_compiler.ops.text.frontend.TracingNodeBuilder", MockNodeBuilder)

    class MockState:
        is_tracing = not eager_mode

        def add_node(self, node):
            pass

    monkeypatch.setattr(global_tracing_state, "add_node", lambda node: None)

    t = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.String, device="cpu"))
    t_float = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.Float32, device="cpu"))

    f_string_to_hash(t, 5)
    regex_replace(t, "a", "b")
    regex_full_match(t, "a")
    string_join([t, t], "-")
    string_length(t)
    string_substr(t, 0, 1)

    string_split(t, " ")

    f_lookup(t, t)
    f_text_vectorization(t, kwarg=1)
    string_to_number(t)
    string_lower(t)
    string_upper(t)
    edit_distance(t, t)
    as_string(t_float)
    as_string(t_float, AsStringConfig())


def test_text_ops(eager_mode, mock_backend, monkeypatch):
    class MockTextOp:
        def __call__(self, *args, **kwargs):
            return args[0] if args else None

    monkeypatch.setattr("ml_switcheroo_compiler.ops.text.ops.get_op", lambda name: MockTextOp)

    t = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.String, device="cpu"))

    o_string_to_hash(t, num_bins=5)
    o_lookup(t)
    o_text_vectorization(t)


def test_text_opdef_infer_shapes():
    t = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.String, device="cpu"))

    assert StringToHash().infer_shape(t) == ()
    assert RegexReplace().infer_shape(t) == ()
    assert StringSplit().infer_shape(t) == ()
    assert Lookup().infer_shape(t) == ()
    assert Lookup().infer_shape() == ()
    assert Hashing().infer_shape(t) == ()
    assert StringLookup().infer_shape(t) == ()
    assert IntegerLookup().infer_shape(t) == ()
    assert TextVectorization().infer_shape(t) == ()

    assert StringToNumber().infer_shape(t) == ()
    assert StringToNumber().infer_shape() == ()

    assert StringLower().infer_shape(t) == ()
    assert StringUpper().infer_shape(t) == ()
    assert StringJoin().infer_shape(t) == ()
    assert StringLength().infer_shape(t) == ()
    assert StringSubstr().infer_shape(t) == ()
    assert RegexFullMatch().infer_shape(t) == ()

    assert EditDistance().infer_shape(t, t) == ()
    assert AsString().infer_shape(t) == ()
    assert ArrayRepr().infer_shape(t) == ()
    assert ArrayStr().infer_shape(t) == ()


def test_create_token(monkeypatch):
    monkeypatch.setattr(CreateToken, "__abstractmethods__", frozenset())
    monkeypatch.setattr(CreateToken, "infer_shape", lambda self, *args, **kwargs: ())
    monkeypatch.setattr(CreateToken, "__call__", lambda self, *args, **kwargs: args[0])

    assert CreateToken.op_name == "CreateToken"
    assert create_token("tok") == "tok"


def test_string_split_trace_fails_without_tracing(monkeypatch):
    monkeypatch.setattr(config, "eager_mode", False)
    monkeypatch.setattr(global_tracing_state, "is_tracing", False)
    t = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.String, device="cpu"))
    with pytest.raises(RuntimeError):
        string_split(t, " ")
