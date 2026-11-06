"""Tests for test_strings_coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import strings as strings_mod
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_strings_ops() -> None:
    """Test all string operations and branches in strings.

    Returns:
        None
    """
    # StringToHash
    hashed = strings_mod._np_string_to_hash(ensure_eager_tensor(np), np.array(["foo", "bar"]), 100)
    assert ensure_eager_tensor(hashed).shape == (2,)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("StringToHash")(np, ensure_callable(np.array)(["foo", "bar"]), 100)).shape == (2,)

    # TextVectorization
    # Match branch: ndim == 1, size == 3, "hello world" in inputs[0]
    in_vec = ["hello world", "sample two", "sample three"]
    multi_hot_res = strings_mod._np_text_vectorization(ensure_eager_tensor(np), in_vec, output_mode="multi_hot")
    assert ensure_eager_tensor(multi_hot_res).shape == (3, 3)

    int_res = strings_mod._np_text_vectorization(ensure_eager_tensor(np), in_vec, output_mode="int")
    assert ensure_eager_tensor(int_res).shape == (3, 2)

    # Fallback branch
    other_vec = strings_mod._np_text_vectorization(ensure_eager_tensor(np), ["random", "words"])
    assert ensure_callable(np.array_equal)(other_vec, ["random", "words"])
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("TextVectorization")(np, ["foo"])).shape == (1,)

    # AsString
    assert ensure_callable(np.array_equal)(ensure_callable(strings_mod._np_as_string)(np, 123), ensure_callable(np.array)(["123"]))
    assert ensure_callable(np.array_equal)(ensure_callable(strings_mod._np_as_string)(np, ensure_callable(np.array)([4, 5])), ensure_callable(np.array)(["4", "5"]))
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("AsString")(np, 999), ensure_callable(np.array)(["999"]))

    # CreateToken
    # Empty or non-string args
    assert strings_mod._np_create_token(ensure_eager_tensor(np)) == 0
    assert strings_mod._np_create_token(ensure_eager_tensor(np), 123) == 0

    # With default vocab
    tok_default = strings_mod._np_create_token(ensure_eager_tensor(np), "hello world")
    assert ensure_callable(np.array_equal)(tok_default, [4, 5])
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("CreateToken")(np, "hello world"), [4, 5])

    # With custom vocab: word in vocab, subword prefix & suffix, subword prefix without suffix, not found
    custom_vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "play": 2,
        "##ing": 3,
        "walk": 4,
    }
    tok_custom = strings_mod._np_create_token(ensure_eager_tensor(np), "play playing walked zzz", vocab=custom_vocab)
    assert ensure_callable(np.array_equal)(tok_custom, [2, 2, 3, 4, 1, 1])
