"""Behavioral unit tests for strings."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager import string_ops as strings_mod
from ml_switcheroo_compiler.backends.eager.string_ops import (
    _np_as_string,
    _np_create_token,
    _np_string_to_hash,
    _np_text_vectorization,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from tests.testing_utils import require_callable


def test_np_string_to_hash() -> None:
    """Test function test_np_string_to_hash."""
    res = _np_string_to_hash(None, np.array(["hello", "world"]), 10)
    assert res.dtype == np.int32
    assert res.shape == (2,)


def test_np_text_vectorization() -> None:
    # default (int) with hello world
    """Test function test_np_text_vectorization."""
    res = _np_text_vectorization(None, np.array(["hello world", "test", "test2"]))
    assert res.shape == (3, 2)

    # hello world match multi_hot
    res2 = _np_text_vectorization(None, np.array(["hello world", "test", "test2"]), output_mode="multi_hot")
    assert res2.shape == (3, 3)

    # NO hello world
    res3 = _np_text_vectorization(None, np.array(["test", "test2", "test3"]))
    assert res3.shape == (3,)


def test_np_as_string() -> None:
    """Test function test_np_as_string."""
    res = _np_as_string(None, 5)
    assert np.array_equal(np.asarray(res), np.array(["5"]))

    res2 = _np_as_string(None, np.array([1, 2]))
    assert np.array_equal(np.asarray(res2), np.array(["1", "2"]))


def test_np_create_token() -> None:
    """Test function test_np_create_token."""
    res = _np_create_token(None)
    assert res.shape == ()
    assert res.dtype == np.int32
    assert res == 0

    # Test real subword BPE/WordPiece tokenization pipeline
    res_tokenizer = _np_create_token(None, "hello world unknown_word")
    # hello -> 4, world -> 5, unknown_word -> contains prefix/suffix subword splitting or <unk> (1)
    assert res_tokenizer.tolist() == [4, 5, 1]


def test_np_text_vectorization_size() -> None:
    # size != 3
    """Test function test_np_text_vectorization_size."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.string_ops import _np_text_vectorization

    res = _np_text_vectorization(None, np.array(["hello world", "test"]))
    assert res.shape == (2,)


def test_strings_ops() -> None:
    """Test all string operations and branches in strings.

    Returns:
        None
    """
    # StringToHash
    hashed = strings_mod._np_string_to_hash(np, np.array(["foo", "bar"]), 100)
    assert hashed.shape == (2,)
    assert require_callable(numpy_eager_registry.get)("StringToHash")(np, np.array(["foo", "bar"]), 100).shape == (2,)

    # TextVectorization
    # Match branch: ndim == 1, size == 3, "hello world" in inputs[0]
    in_vec = ["hello world", "sample two", "sample three"]
    multi_hot_res = strings_mod._np_text_vectorization(np, in_vec, output_mode="multi_hot")
    assert multi_hot_res.shape == (3, 3)

    int_res = strings_mod._np_text_vectorization(np, in_vec, output_mode="int")
    assert int_res.shape == (3, 2)

    # Fallback branch
    other_vec = strings_mod._np_text_vectorization(np, ["random", "words"])
    assert np.array_equal(other_vec, ["random", "words"])
    assert require_callable(numpy_eager_registry.get)("TextVectorization")(np, ["foo"]).shape == (1,)

    # AsString
    assert np.array_equal(strings_mod._np_as_string(np, 123), np.array(["123"]))
    assert np.array_equal(
        strings_mod._np_as_string(np, np.array([4, 5])),
        np.array(["4", "5"]),
    )
    assert np.array_equal(require_callable(numpy_eager_registry.get)("AsString")(np, 999), np.array(["999"]))

    # CreateToken
    # Empty or non-string args
    assert strings_mod._np_create_token(np) == 0
    assert strings_mod._np_create_token(np, 123) == 0

    # With default vocab
    tok_default = strings_mod._np_create_token(np, "hello world")
    assert np.array_equal(tok_default, [4, 5])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("CreateToken")(np, "hello world"), [4, 5])

    # With custom vocab: word in vocab, subword prefix & suffix, subword prefix without suffix, not found
    custom_vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "play": 2,
        "##ing": 3,
        "walk": 4,
    }
    tok_custom = strings_mod._np_create_token(np, "play playing walked zzz", vocab=custom_vocab)
    assert np.array_equal(tok_custom, [2, 2, 3, 4, 1, 1])
