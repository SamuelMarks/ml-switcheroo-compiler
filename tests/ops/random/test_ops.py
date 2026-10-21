"""Tests for test_ops."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.random_ops import RngBitGenerator, RngUniform
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_random_ops_infer_shape_coverage() -> None:
    """Test the random ops infer shape coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    res1 = as_callable(RngBitGenerator)().infer_shape(None, (1, 2), None)
    assert res1 == (1, 2)
    res2 = as_callable(RngUniform)().infer_shape(None, None, (3, 4), None)
    assert res2 == (3, 4)
