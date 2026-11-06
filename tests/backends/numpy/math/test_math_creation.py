"""Tests for test_math_creation."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_creation as creation_mod
from ml_switcheroo_compiler.ops.linalg.linear_operator import (
    LinearOperatorIdentity,
    LinearOperatorScaledIdentity,
    LinearOperatorZeros,
)
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod

if typing.TYPE_CHECKING:
    pass


def test_math_creation_operations() -> None:
    """Test all array creation operations in math_creation.

    Returns:
        None
    """
    # Fromfunction
    fn_res = creation_mod._np_fromfunction_(ensure_eager_tensor(np), lambda i, j: i + j, (2, 2), dtype=np.dtype(int))
    assert ensure_eager_tensor(fn_res).shape == (2, 2)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Fromfunction")(np, lambda i, j: i + j, (2, 2), dtype=int)).shape == (2, 2)

    # Fromiter
    iter_res = creation_mod._np_fromiter_(ensure_eager_tensor(np), [1, 2, 3], dtype=np.dtype(int))
    assert ensure_callable(np.array_equal)(iter_res, [1, 2, 3])
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("Fromiter")(np, [1, 2, 3], dtype=int), [1, 2, 3])

    # Frompyfunc
    ufunc = creation_mod._np_frompyfunc_(ensure_eager_tensor(np), lambda x: x + 1, 1, 1)
    assert ensure_callable(np.array_equal)(ensure_callable(ufunc)([1, 2]), [2, 3])
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("Frompyfunc")(np, lambda x: ensure_eager_tensor(x) + ensure_eager_tensor(1), 1, 1)([1, 2]), [2, 3])

    # Geomspace
    geom_res = creation_mod._np_geomspace_(ensure_eager_tensor(np), 1, 1000, 4)
    assert ensure_callable(len)(geom_res) == 4
    assert ensure_callable(len)(ensure_callable(numpy_eager_registry.get)("Geomspace")(np, 1, 1000, 4)) == 4

    # Zeros, Ones, Empty, Full
    assert ensure_callable(np.array_equal)(ensure_callable(creation_mod._np_zeros_)(np, (2, 3)), ensure_callable(np.zeros)((2, 3)))
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("Zeros")(np, (2, 3)), ensure_callable(np.zeros)((2, 3)))

    assert ensure_callable(np.array_equal)(ensure_callable(creation_mod._np_ones_)(np, (2, 3)), ensure_callable(np.ones)((2, 3)))
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("Ones")(np, (2, 3)), ensure_callable(np.ones)((2, 3)))

    assert ensure_eager_tensor(creation_mod._np_empty_(ensure_eager_tensor(np), (2, 3))).shape == (2, 3)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("Empty")(np, (2, 3))).shape == (2, 3)

    assert ensure_callable(np.array_equal)(ensure_callable(creation_mod._np_full_)(np, (2, 3), 7), ensure_callable(np.full)((2, 3), 7))
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("Full")(np, (2, 3), 7), ensure_callable(np.full)((2, 3), 7))

    # Like variations
    proto = np.ones((2, 2))
    assert ensure_callable(np.array_equal)(ensure_callable(creation_mod._np_zeros_like_)(np, proto), ensure_callable(np.zeros)((2, 2)))
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("ZerosLike")(np, proto), ensure_callable(np.zeros)((2, 2)))

    assert ensure_callable(np.array_equal)(ensure_callable(creation_mod._np_ones_like_)(np, proto), ensure_callable(np.ones)((2, 2)))
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("OnesLike")(np, proto), ensure_callable(np.ones)((2, 2)))

    assert ensure_eager_tensor(creation_mod._np_empty_like_(ensure_eager_tensor(np), proto)).shape == (2, 2)
    assert ensure_eager_tensor(ensure_callable(numpy_eager_registry.get)("EmptyLike")(np, proto)).shape == (2, 2)

    assert ensure_callable(np.array_equal)(ensure_callable(creation_mod._np_full_like_)(np, proto, 9), ensure_callable(np.full)((2, 2), 9))
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("FullLike")(np, proto, 9), ensure_callable(np.full)((2, 2), 9))

    # Arange
    assert ensure_callable(np.array_equal)(ensure_callable(creation_mod._np_arange_)(np, 0, 5), ensure_callable(np.arange)(0, 5))
    assert ensure_callable(np.array_equal)(ensure_callable(numpy_eager_registry.get)("Arange")(np, 0, 5), ensure_callable(np.arange)(0, 5))

    # LinearOperator creation
    op_id = creation_mod._np_linearoperatoridentity(ensure_eager_tensor(np))
    assert isinstance(op_id, LinearOperatorIdentity)
    assert isinstance(ensure_callable(numpy_eager_registry.get)("LinearOperatorIdentity")(np), LinearOperatorIdentity)

    op_scale = creation_mod._np_linearoperatorscaledidentity(ensure_eager_tensor(np))
    assert isinstance(op_scale, LinearOperatorScaledIdentity)
    assert isinstance(ensure_callable(numpy_eager_registry.get)("LinearOperatorScaledIdentity")(np), LinearOperatorScaledIdentity)

    op_zero = creation_mod._np_linearoperatorzeros(ensure_eager_tensor(np))
    assert isinstance(op_zero, LinearOperatorZeros)
    assert isinstance(ensure_callable(numpy_eager_registry.get)("LinearOperatorZeros")(np), LinearOperatorZeros)

    # FromDlpack
    dummy_with_dlpack = types.SimpleNamespace(from_dlpack=lambda x: np.array([42]))
    assert ensure_callable(np.array_equal)(ensure_callable(creation_mod._np_fromdlpack)(dummy_with_dlpack, "tensor"), ensure_callable(np.array)([42]))
    dummy_without_dlpack = types.SimpleNamespace()
    assert creation_mod._np_fromdlpack(ensure_mod(dummy_without_dlpack), 99) == 99
    assert ensure_callable(numpy_eager_registry.get)("FromDlpack")(dummy_without_dlpack, 99) == 99

    # Frombuffer
    assert creation_mod._np_frombuffer(ensure_eager_tensor(np)) is None
    buf_res = creation_mod._np_frombuffer(ensure_eager_tensor(np), b"abcdef", dtype=np.dtype(np.uint8))
    assert buf_res is not None and ensure_callable(len)(buf_res) == 6
    assert ensure_callable(numpy_eager_registry.get)("Frombuffer")(np, b"abcdef", dtype=np.uint8) is not None
