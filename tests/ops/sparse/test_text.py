"""Tests for test_text."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.registry as registry_mod
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
from ml_switcheroo_compiler.ops.text import CreateToken, create_token
from ml_switcheroo_compiler.ops.text.frontend import AsStringConfig, as_string, edit_distance, regex_full_match, regex_replace, string_join, string_length, string_lower, string_split, string_substr, string_to_number, string_upper
from ml_switcheroo_compiler.ops.text.frontend import lookup as f_lookup
from ml_switcheroo_compiler.ops.text.frontend import string_to_hash as f_string_to_hash
from ml_switcheroo_compiler.ops.text.frontend import text_vectorization as f_text_vectorization
from ml_switcheroo_compiler.ops.text.ops import ArrayRepr, ArrayStr, AsString, EditDistance, Hashing, IntegerLookup, Lookup, RegexFullMatch, RegexReplace, StringJoin, StringLength, StringLookup, StringLower, StringSplit, StringSubstr, StringToHash, StringToNumber, StringUpper, TextVectorization
from ml_switcheroo_compiler.ops.text.ops import lookup as o_lookup
from ml_switcheroo_compiler.ops.text.ops import string_to_hash as o_string_to_hash
from ml_switcheroo_compiler.ops.text.ops import text_vectorization as o_text_vectorization
from ml_switcheroo_compiler.tracing import global_tracing_state

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockBackend:
    """Test class MockBackend."""

    def execute_op(self, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function execute_op.

        Args:
            op_name: Test parameter `op_name`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        if op_name == "StringSplit":
            return (np.zeros(()), np.zeros(()))
        return np.zeros(())

    def array(self, x: EagerValue) -> EagerValue:
        """Test function array.

        Args:
            x: Test parameter `x`.
        """
        return np.array(x)


@pytest.fixture
def mock_backend(monkeypatch: EagerValue) -> None:
    """Test function mock_backend.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(monkeypatch, "setattr")))(registry_mod, "get_active_backend", lambda: cast("EagerCallableProtocol", MockBackend)())
    cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(monkeypatch, "setattr")))("ml_switcheroo_compiler.ops.text.frontend.get_active_backend", lambda: cast("EagerCallableProtocol", MockBackend)())


@pytest.fixture(params=[True, False])
def eager_mode(request: EagerValue, monkeypatch: EagerValue) -> EagerValue:
    """Test function eager_mode.

    Args:
        request: Test parameter `request`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(monkeypatch, "setattr")))(config, "eager_mode", cast("EagerTensorProtocol", getattr(request, "param")))
    cast("EagerCallableProtocol", cast("EagerTensorProtocol", getattr(monkeypatch, "setattr")))(global_tracing_state, "is_tracing", not cast("EagerTensorProtocol", getattr(request, "param")))
    return cast("EagerTensorProtocol", getattr(request, "param"))


def test_sparse_frontend(eager_mode: EagerTensorProtocol, mock_backend: EagerTensorProtocol, monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_sparse_frontend.

    Args:
        eager_mode: Test parameter `eager_mode`.
        mock_backend: Test parameter `mock_backend`.
        monkeypatch: Test parameter `monkeypatch`.
    """

    class MockOp:
        """Test class MockOp."""

        def infer_shape(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function infer_shape.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return ()

    monkeypatch.setattr("ml_switcheroo_compiler.ops.sparse.frontend.get_op", lambda name: MockOp)
    cast("EagerCallableProtocol", monkeypatch.setattr)(global_tracing_state, "add_node", lambda node: None)
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


def test_text_frontend(eager_mode: EagerTensorProtocol, mock_backend: EagerTensorProtocol, monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_text_frontend.

    Args:
        eager_mode: Test parameter `eager_mode`.
        mock_backend: Test parameter `mock_backend`.
        monkeypatch: Test parameter `monkeypatch`.
    """

    class MockNodeBuilder:
        """Test class MockNodeBuilder."""

        @staticmethod
        def extract_proxy_inputs(inputs: EagerValue) -> EagerValue:
            """Test function extract_proxy_inputs.

            Args:
                inputs: Test parameter `inputs`.
            """
            return (["dummy_id"], None, None)

    monkeypatch.setattr("ml_switcheroo_compiler.ops.text.frontend.TracingNodeBuilder", MockNodeBuilder)

    class MockState:
        """Test class MockState."""

        is_tracing = not eager_mode

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    cast("EagerCallableProtocol", monkeypatch.setattr)(global_tracing_state, "add_node", lambda node: None)
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
    assert as_string(t_float, AsStringConfig()) is not None


def test_text_ops(eager_mode: EagerTensorProtocol, mock_backend: EagerTensorProtocol, monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_text_ops.

    Args:
        eager_mode: Test parameter `eager_mode`.
        mock_backend: Test parameter `mock_backend`.
        monkeypatch: Test parameter `monkeypatch`.
    """

    class MockTextOp:
        """Test class MockTextOp."""

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return args[0] if args else None

    monkeypatch.setattr("ml_switcheroo_compiler.ops.text.ops.get_op", lambda name: MockTextOp)
    t = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.String, device="cpu"))
    assert o_string_to_hash(t, num_bins=5) is t
    assert o_lookup(t) is t
    assert o_text_vectorization(t) is t


def test_text_opdef_infer_shapes() -> None:
    """Test function test_text_opdef_infer_shapes."""
    t = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.String, device="cpu"))
    assert cast("EagerCallableProtocol", StringToHash)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", RegexReplace)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", StringSplit)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", Lookup)().infer_shape(t) == ()
    assert Lookup().infer_shape() == ()
    assert cast("EagerCallableProtocol", Hashing)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", StringLookup)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", IntegerLookup)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", TextVectorization)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", StringToNumber)().infer_shape(t) == ()
    assert StringToNumber().infer_shape() == ()
    assert cast("EagerCallableProtocol", StringLower)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", StringUpper)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", StringJoin)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", StringLength)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", StringSubstr)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", RegexFullMatch)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", EditDistance)().infer_shape(t, t) == ()
    assert cast("EagerCallableProtocol", AsString)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", ArrayRepr)().infer_shape(t) == ()
    assert cast("EagerCallableProtocol", ArrayStr)().infer_shape(t) == ()


def test_create_token(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_create_token.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    if hasattr(CreateToken, "__abstractmethods__"):
        cast("EagerCallableProtocol", monkeypatch.setattr)(CreateToken, "__abstractmethods__", cast("EagerCallableProtocol", frozenset)())
    monkeypatch.setattr(CreateToken, "infer_shape", lambda self, *args, **kwargs: ())
    monkeypatch.setattr(CreateToken, "__call__", lambda self, *args, **kwargs: args[0])
    assert CreateToken.op_name == "CreateToken"
    assert cast("EagerTensorProtocol", create_token("tok")) == cast("EagerTensorProtocol", "tok")


def test_string_split_trace_fails_without_tracing(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_string_split_trace_fails_without_tracing.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    cast("EagerCallableProtocol", monkeypatch.setattr)(config, "eager_mode", False)
    cast("EagerCallableProtocol", monkeypatch.setattr)(global_tracing_state, "is_tracing", False)
    t = Tensor(np.zeros(()), TensorConfig(shape=(), dtype=DType.String, device="cpu"))
    with pytest.raises(RuntimeError):
        string_split(t, " ")
