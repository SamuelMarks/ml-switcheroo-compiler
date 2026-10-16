"""Test Numpy eager strings coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.strings import _np_create_token

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_string_to_hash_bucket() -> None:
    # Test valid BPE/vocab translation
    """Test function test_string_to_hash_bucket."""
    vocab = {"hello": 0, "wor": 1, "##ld": 2, "<unk>": 99}
    res = _np_create_token(None, "hello world", vocab=vocab)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res, cast("Callable[..., EagerTensorProtocol]", np.array)([0, 1, 2], dtype=np.int32))

    # Test fallback vocab creation
    res2 = _np_create_token(None, "hello world")
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res2, cast("Callable[..., EagerTensorProtocol]", np.array)([4, 5], dtype=np.int32))

    # Test unknown chars without exact suffix match
    vocab2 = {"a": 0, "<unk>": 99}
    res3 = _np_create_token(None, "abc", vocab=vocab2)
    assert cast("EagerTensorProtocol", 99) in cast("EagerTensorProtocol", res3)  # should fallback to <unk>

    # Test complete unknown. The loop assigns found=False, breaks after trying 'x','y','z'.
    # Then line 128 adds vocab.get('<unk>', 1).
    res4 = _np_create_token(None, "xyz", vocab=vocab2)
    assert cast("EagerTensorProtocol", 99) in cast("EagerTensorProtocol", res4)


def test_string_to_hash_bucket_not_string() -> None:
    """Test function test_string_to_hash_bucket_not_string."""
    res = _np_create_token(None, 123)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res, cast("Callable[..., EagerTensorProtocol]", np.array)(0, dtype=np.int32))
