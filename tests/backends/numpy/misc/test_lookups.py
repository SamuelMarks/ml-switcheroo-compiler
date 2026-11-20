"""Behavioral unit tests for lookups."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager import lookup_ops as lookups_module
from ml_switcheroo_compiler.backends.eager.lookup_ops import (
    _np_hashing,
    _np_integer_lookup,
    _np_lookup,
    _np_string_lookup,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import require_callable


def test_hashing() -> None:
    """Test hashing.

    Returns:
        None
    """
    inputs = np.array([1, 2])
    res = _np_hashing(np, inputs, num_bins=10)
    assert np.array_equal(ensure_ndarray(res), inputs)


def test_integer_lookup() -> None:
    """Test integer lookup.

    Returns:
        None
    """
    inputs = np.array([1, 2])
    res = _np_integer_lookup(np, inputs)
    assert np.array_equal(ensure_ndarray(res), inputs)


def test_lookup() -> None:
    """Test lookup.

    Returns:
        None
    """
    inputs = np.array([1, 2])
    vocab = np.array([1, 2, 3])
    res = _np_lookup(np, inputs, vocab)
    assert np.array_equal(res, np.array([0, 1], dtype=np.int32))


def test_string_lookup() -> None:
    """Test string lookup.

    Returns:
        None
    """
    inputs = np.array(["a", "b"])
    res = _np_string_lookup(np, inputs)
    assert np.array_equal(ensure_ndarray(res), inputs)


def test_lookups_ops() -> None:
    """Test lookup operations directly and through eager registry.

    Returns:
        None
    """
    inputs = np.array([1, 2, 3])
    assert np.array_equal(ensure_ndarray(lookups_module._np_hashing(np, inputs, num_bins=10)), inputs)
    reg_hashing = numpy_eager_registry.get("Hashing")
    assert np.array_equal(require_callable(reg_hashing)(np, inputs, num_bins=10), inputs)

    assert np.array_equal(ensure_ndarray(lookups_module._np_integer_lookup(np, inputs)), inputs)
    reg_int_lookup = numpy_eager_registry.get("IntegerLookup")
    assert np.array_equal(require_callable(reg_int_lookup)(np, inputs), inputs)

    vocab = np.array([2, 3, 4])
    lookup_res = lookups_module._np_lookup(np, inputs, vocab)
    assert np.array_equal(lookup_res, np.array([0, 0, 1]))

    reg_lookup = numpy_eager_registry.get("Lookup")
    assert reg_lookup is not None

    str_inputs = np.array(["a", "b"])
    assert np.array_equal(ensure_ndarray(lookups_module._np_string_lookup(np, str_inputs)), str_inputs)
    reg_str_lookup = numpy_eager_registry.get("StringLookup")
    assert np.array_equal(require_callable(reg_str_lookup)(np, str_inputs), str_inputs)
