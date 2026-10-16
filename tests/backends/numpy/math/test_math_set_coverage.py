"""Tests for test_math_set_coverage."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_set as math_set

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_math_set() -> None:
    """Test Union1d, Intersect1d, and Isin directly and via registry."""
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([2, 3, 4], dtype=np.int64)

    union = math_set._np_union1d(np, a, b)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(union, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2, 3, 4]))

    reg_union = numpy_eager_registry.get("Union1d")
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", reg_union)(np, a, b), cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2, 3, 4]))

    mock_mod = types.SimpleNamespace(
        intersect1d=lambda *args, **kw: np.array([2, 3]),
        isin=lambda *args, **kw: np.array([False, True, True]),
    )

    inter = math_set._np_intersect1d_(mock_mod, a, b)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(inter, cast("Callable[..., EagerTensorProtocol]", np.array)([2, 3]))

    reg_inter = numpy_eager_registry.get("Intersect1d")
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", reg_inter)(mock_mod, a, b), cast("Callable[..., EagerTensorProtocol]", np.array)([2, 3]))

    isin = math_set._np_isin_(mock_mod, a, b)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(isin, cast("Callable[..., EagerTensorProtocol]", np.array)([False, True, True]))

    reg_isin = numpy_eager_registry.get("Isin")
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", reg_isin)(mock_mod, a, b), cast("Callable[..., EagerTensorProtocol]", np.array)([False, True, True]))
