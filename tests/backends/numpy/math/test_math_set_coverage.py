"""Tests for test_math_set_coverage."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_set as math_set
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_math_set() -> None:
    """Test Union1d, Intersect1d, and Isin directly and via registry."""
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([2, 3, 4], dtype=np.int64)

    union = math_set._np_union1d(np, a, b)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(union, typing.cast("EagerCallableProtocol", np.array)([1, 2, 3, 4]))

    reg_union = numpy_eager_registry.get("Union1d")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", reg_union)(np, a, b), typing.cast("EagerCallableProtocol", np.array)([1, 2, 3, 4]))

    mock_mod = types.SimpleNamespace(
        intersect1d=lambda *args, **kw: np.array([2, 3]),
        isin=lambda *args, **kw: np.array([False, True, True]),
    )

    inter = math_set._np_intersect1d_(typing.cast("BackendModuleProtocol", mock_mod), a, b)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(inter, typing.cast("EagerCallableProtocol", np.array)([2, 3]))

    reg_inter = numpy_eager_registry.get("Intersect1d")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", reg_inter)(mock_mod, a, b), typing.cast("EagerCallableProtocol", np.array)([2, 3]))

    isin = math_set._np_isin_(typing.cast("BackendModuleProtocol", mock_mod), a, b)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(isin, typing.cast("EagerCallableProtocol", np.array)([False, True, True]))

    reg_isin = numpy_eager_registry.get("Isin")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", reg_isin)(mock_mod, a, b), typing.cast("EagerCallableProtocol", np.array)([False, True, True]))
