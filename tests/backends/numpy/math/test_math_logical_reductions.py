"""Tests for numpy eager math logical reductions."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_logical_reductions import (
    _np_all,
    _np_count_nonzero,
)


def test_all() -> None:
    """Test all.

    Returns:
        None
    """
    res = _np_all(np, np.array([True, True]))
    assert res


def test_count_nonzero() -> None:
    """Test count nonzero.

    Returns:
        None
    """
    res = _np_count_nonzero(np, np.array([0, 1, 0, 1]))
    assert res == 2
