"""Test Numpy eager strings coverage."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.strings import _np_create_token


def test_string_to_hash_bucket():
    # Test valid BPE/vocab translation
    vocab = {"hello": 0, "wor": 1, "##ld": 2, "<unk>": 99}
    res = _np_create_token(None, "hello world", vocab=vocab)
    assert np.array_equal(res, np.array([0, 1, 2], dtype=np.int32))

    # Test fallback vocab creation
    res2 = _np_create_token(None, "hello world")
    assert np.array_equal(res2, np.array([4, 5], dtype=np.int32))

    # Test unknown chars without exact suffix match
    vocab2 = {"a": 0, "<unk>": 99}
    res3 = _np_create_token(None, "abc", vocab=vocab2)
    assert 99 in res3  # should fallback to <unk>

    # Test complete unknown. The loop assigns found=False, breaks after trying 'x','y','z'.
    # Then line 128 adds vocab.get('<unk>', 1).
    res4 = _np_create_token(None, "xyz", vocab=vocab2)
    assert 99 in res4


def test_string_to_hash_bucket_not_string():
    res = _np_create_token(None, 123)
    assert np.array_equal(res, np.array(0, dtype=np.int32))
