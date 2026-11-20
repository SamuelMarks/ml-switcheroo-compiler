"""Tests for test_math_creation."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.ops.linalg.linear_operator import (
    LinearOperatorIdentity,
    LinearOperatorScaledIdentity,
    LinearOperatorZeros,
)
from tests.backends.numpy.eager._math_advanced_test_ns import math_creation as creation_mod
from tests.testing_utils import MockTestStub, NumpyTestArray, require_callable


def test_math_creation_operations() -> None:
    """Test all array creation operations in math_creation.

    Returns:
        None
    """
    # Fromfunction
    fn_res = creation_mod._np_fromfunction_(np, lambda i, j: i + j, (2, 2), dtype=np.dtype(int))
    assert fn_res.shape == (2, 2)
    assert require_callable(numpy_eager_registry.get)("Fromfunction")(
        np, lambda i, j: i + j, (2, 2), dtype=int
    ).shape == (2, 2)

    # Fromiter
    iter_res = creation_mod._np_fromiter_(np, [1, 2, 3], dtype=np.dtype(int))
    assert np.array_equal(iter_res, [1, 2, 3])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Fromiter")(np, [1, 2, 3], dtype=int), [1, 2, 3])

    # Frompyfunc
    ufunc = creation_mod._np_frompyfunc_(np, lambda x: x + 1, 1, 1)
    assert np.array_equal(ensure_ndarray(require_callable(ufunc)([1, 2])), [2, 3])
    assert np.array_equal(
        require_callable(numpy_eager_registry.get)("Frompyfunc")(np, lambda x: x + 1, 1, 1)([1, 2]),
        [2, 3],
    )

    # Geomspace
    geom_res = creation_mod._np_geomspace_(np, 1, 1000, 4)
    assert len(geom_res) == 4
    assert len(require_callable(numpy_eager_registry.get)("Geomspace")(np, 1, 1000, 4)) == 4

    # Zeros, Ones, Empty, Full
    assert np.array_equal(creation_mod._np_zeros_(np, (2, 3)), np.zeros((2, 3)))
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Zeros")(np, (2, 3)), np.zeros((2, 3)))

    assert np.array_equal(creation_mod._np_ones_(np, (2, 3)), np.ones((2, 3)))
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Ones")(np, (2, 3)), np.ones((2, 3)))

    assert creation_mod._np_empty_(np, (2, 3)).shape == (2, 3)
    assert require_callable(numpy_eager_registry.get)("Empty")(np, (2, 3)).shape == (2, 3)

    assert np.array_equal(creation_mod._np_full_(np, (2, 3), 7), np.full((2, 3), 7))
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Full")(np, (2, 3), 7), np.full((2, 3), 7))

    # Like variations
    proto = np.ones((2, 2))
    assert np.array_equal(creation_mod._np_zeros_like_(np, proto), np.zeros((2, 2)))
    assert np.array_equal(require_callable(numpy_eager_registry.get)("ZerosLike")(np, proto), np.zeros((2, 2)))

    assert np.array_equal(creation_mod._np_ones_like_(np, proto), np.ones((2, 2)))
    assert np.array_equal(require_callable(numpy_eager_registry.get)("OnesLike")(np, proto), np.ones((2, 2)))

    assert creation_mod._np_empty_like_(np, proto).shape == (2, 2)
    assert require_callable(numpy_eager_registry.get)("EmptyLike")(np, proto).shape == (2, 2)

    assert np.array_equal(creation_mod._np_full_like_(np, proto, 9), np.full((2, 2), 9))
    assert np.array_equal(require_callable(numpy_eager_registry.get)("FullLike")(np, proto, 9), np.full((2, 2), 9))

    # Arange
    assert np.array_equal(creation_mod._np_arange_(np, 0, 5), np.arange(0, 5))
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Arange")(np, 0, 5), np.arange(0, 5))

    # LinearOperator creation
    assert isinstance(creation_mod._np_linearoperatoridentity(np), LinearOperatorIdentity)
    assert isinstance(require_callable(numpy_eager_registry.get)("LinearOperatorIdentity")(np), LinearOperatorIdentity)

    assert isinstance(creation_mod._np_linearoperatorscaledidentity(np), LinearOperatorScaledIdentity)
    assert isinstance(
        require_callable(numpy_eager_registry.get)("LinearOperatorScaledIdentity")(np), LinearOperatorScaledIdentity
    )

    assert isinstance(creation_mod._np_linearoperatorzeros(np), LinearOperatorZeros)
    assert isinstance(require_callable(numpy_eager_registry.get)("LinearOperatorZeros")(np), LinearOperatorZeros)

    # FromDlpack
    class _DlpackStub(MockTestStub):
        """Stub backend module providing `from_dlpack`."""

        @staticmethod
        def from_dlpack(x: BackendArg) -> NumpyTestArray:
            """Return mock DLPack conversion result.

            Args:
                x: Input tensor capsule or object.

            Returns:
                NumpyTestArray: Constant array `[42]`.
            """
            del x
            return np.array([42])

    dummy_with_dlpack = _DlpackStub()
    assert np.array_equal(creation_mod._np_fromdlpack(dummy_with_dlpack, "tensor"), np.array([42]))
    dummy_without_dlpack = MockTestStub()
    assert creation_mod._np_fromdlpack(dummy_without_dlpack, 99) == 99
    assert require_callable(numpy_eager_registry.get)("FromDlpack")(dummy_without_dlpack, 99) == 99

    # Frombuffer
    assert creation_mod._np_frombuffer(np) is None
    buf_res = creation_mod._np_frombuffer(np, b"abcdef", dtype=np.dtype(np.uint8))
    assert buf_res is not None and len(buf_res) == 6
    assert require_callable(numpy_eager_registry.get)("Frombuffer")(np, b"abcdef", dtype=np.uint8) is not None
