"""Test Numpy eager strings coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.strings import _np_create_token
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_string_to_hash_bucket() -> None:
    # Test valid BPE/vocab translation
    """Test function test_string_to_hash_bucket."""
    vocab = {"hello": 0, "wor": 1, "##ld": 2, "<unk>": 99}
    res = _np_create_token(typing.cast("BackendModuleProtocol", None), "hello world", vocab=vocab)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res, typing.cast("EagerCallableProtocol", np.array)([0, 1, 2], dtype=np.int32))

    # Test fallback vocab creation
    res2 = _np_create_token(typing.cast("BackendModuleProtocol", None), "hello world")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res2, typing.cast("EagerCallableProtocol", np.array)([4, 5], dtype=np.int32))

    # Test unknown chars without exact suffix match
    vocab2 = {"a": 0, "<unk>": 99}
    res3 = _np_create_token(typing.cast("BackendModuleProtocol", None), "abc", vocab=vocab2)
    assert typing.cast("EagerTensorProtocol", 99) in typing.cast("EagerTensorProtocol", res3)  # should fallback to <unk>

    # Test complete unknown. The loop assigns found=False, breaks after trying 'x','y','z'.
    # Then line 128 adds vocab.get('<unk>', 1).
    res4 = _np_create_token(typing.cast("BackendModuleProtocol", None), "xyz", vocab=vocab2)
    assert typing.cast("EagerTensorProtocol", 99) in typing.cast("EagerTensorProtocol", res4)


def test_string_to_hash_bucket_not_string() -> None:
    """Test function test_string_to_hash_bucket_not_string."""
    res = _np_create_token(typing.cast("BackendModuleProtocol", None), 123)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res, typing.cast("EagerCallableProtocol", np.array)(0, dtype=np.int32))
