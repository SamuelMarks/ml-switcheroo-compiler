"""Tests for test_core_math_bitwise."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_bitwise import _np_packbits, _np_unpackbits, _signbit
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, coerce_operand_list, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_math_bitwise_coverage() -> None:
    """Test 100% lines and branches of math_bitwise.

    Returns:
        None
    """
    # 1. _signbit
    # Case A: backend has signbit
    res_sb = _signbit(np, np.array([-1.0, 0.0, 1.0]))
    assert bool(ensure_ndarray(res_sb)[0]) is True
    assert bool(ensure_ndarray(res_sb)[2]) is False

    # Case B: backend lacks signbit (fallback x < 0)
    class DummyNoSignbit(MockTestStub):
        """Dummy backend without signbit."""

        pass

    res_fallback = _signbit(DummyNoSignbit(), np.array([-2, 5]))
    assert bool(ensure_ndarray(res_fallback)[0]) is True
    assert bool(ensure_ndarray(res_fallback)[1]) is False

    # 2. _np_packbits
    # Case A: backend has packbits
    class DummyPackbits(MockTestStub):
        """Dummy backend with packbits."""

        def packbits(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock packbits.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `packbits`.
            """
            return "packed"

    assert _np_packbits(DummyPackbits(), [1, 0]) == "packed"

    # Case B: backend lacks packbits (fallback to numpy.packbits)
    class DummyNoPackbits(MockTestStub):
        """Dummy backend without packbits."""

        pass

    arr = np.array([[[1, 0, 1, 1, 0, 0, 1, 0]]], dtype=np.uint8)
    packed_res = _np_packbits(DummyNoPackbits(), arr, axis=-1)
    assert extract_test_shape(packed_res) == (1, 1, 1)

    # 3. _np_unpackbits
    # Case A: backend has unpackbits
    class DummyUnpackbits(MockTestStub):
        """Dummy backend with unpackbits."""

        def unpackbits(self, *args: BackendArg, **kwargs: BackendArg) -> str:
            """Mock unpackbits.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                str: Computed output value from `unpackbits`.
            """
            return "unpacked"

    assert _np_unpackbits(DummyUnpackbits(), [180]) == "unpacked"

    # Case B: backend lacks unpackbits (fallback to numpy.unpackbits)
    unpacked_res = _np_unpackbits(DummyNoPackbits(), np.array([180], dtype=np.uint8))
    assert len(coerce_operand_list(unpacked_res)) == 8

    # Registry verification
    assert global_eager_registry.get("Signbit") is not None
    assert global_eager_registry.get("Packbits") is not None
    assert global_eager_registry.get("Unpackbits") is not None
