"""Tests for numpy eager lookups."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.lookups import (
    _np_hashing,
    _np_integer_lookup,
    _np_lookup,
    _np_string_lookup,
)
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_hashing() -> None:
    """Test hashing.

    Returns:
        None
    """
    inputs = np.array([1, 2])
    res = _np_hashing(ensure_eager_tensor(np), inputs, num_bins=10)
    assert ensure_callable(np.array_equal)(res, inputs)


def test_integer_lookup() -> None:
    """Test integer lookup.

    Returns:
        None
    """
    inputs = np.array([1, 2])
    res = _np_integer_lookup(ensure_eager_tensor(np), inputs)
    assert ensure_callable(np.array_equal)(res, inputs)


def test_lookup() -> None:
    """Test lookup.

    Returns:
        None
    """
    inputs = np.array([1, 2])
    vocab = np.array([1, 2, 3])
    res = _np_lookup(ensure_eager_tensor(np), inputs, vocab)
    assert ensure_callable(np.array_equal)(res, ensure_callable(np.array)([0, 1], dtype=np.int32))


def test_string_lookup() -> None:
    """Test string lookup.

    Returns:
        None
    """
    inputs = np.array(["a", "b"])
    res = _np_string_lookup(ensure_eager_tensor(np), inputs)
    assert ensure_callable(np.array_equal)(res, inputs)
