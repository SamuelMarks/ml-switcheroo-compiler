"""Tests for test_strings."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.strings import _np_as_string, _np_create_token, _np_string_to_hash, _np_text_vectorization

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_np_string_to_hash() -> None:
    """Test function test_np_string_to_hash."""
    res = _np_string_to_hash(None, np.array(["hello", "world"]), 10)
    assert cast("EagerTensorProtocol", getattr(res, "dtype")) == np.int32
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)


def test_np_text_vectorization() -> None:
    # default (int) with hello world
    """Test function test_np_text_vectorization."""
    res = _np_text_vectorization(None, np.array(["hello world", "test", "test2"]))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (3, 2)

    # hello world match multi_hot
    res2 = _np_text_vectorization(None, np.array(["hello world", "test", "test2"]), output_mode="multi_hot")
    assert cast("EagerTensorProtocol", getattr(res2, "shape")) == (3, 3)

    # NO hello world
    res3 = _np_text_vectorization(None, np.array(["test", "test2", "test3"]))
    assert cast("EagerTensorProtocol", getattr(res3, "shape")) == (3,)


def test_np_as_string() -> None:
    """Test function test_np_as_string."""
    res = _np_as_string(None, 5)
    assert cast("EagerTensorProtocol", getattr(res, "tolist"))() == ["5"]

    res2 = _np_as_string(None, np.array([1, 2]))
    assert cast("EagerTensorProtocol", getattr(res2, "tolist"))() == ["1", "2"]


def test_np_create_token() -> None:
    """Test function test_np_create_token."""
    res = _np_create_token(None)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == ()
    assert cast("EagerTensorProtocol", getattr(res, "dtype")) == np.int32
    assert res == 0

    # Test real subword BPE/WordPiece tokenization pipeline
    res_tokenizer = _np_create_token(None, "hello world unknown_word")
    # hello -> 4, world -> 5, unknown_word -> contains prefix/suffix subword splitting or <unk> (1)
    assert cast("EagerTensorProtocol", getattr(res_tokenizer, "tolist"))() == [4, 5, 1]


def test_np_text_vectorization_size() -> None:
    # size != 3
    """Test function test_np_text_vectorization_size."""
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.strings import _np_text_vectorization

    res = _np_text_vectorization(None, np.array(["hello world", "test"]))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)
