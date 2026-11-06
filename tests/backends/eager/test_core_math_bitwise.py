"""Tests for test_core_math_bitwise."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_bitwise import (
    _np_packbits,
    _np_unpackbits,
    _signbit,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_math_bitwise_coverage() -> None:
    """Test 100% lines and branches of math_bitwise.

    Returns:
        None
    """
    # 1. _signbit
    # Case A: backend has signbit
    res_sb = _signbit(ensure_eager_tensor(np), np.array([-1.0, 0.0, 1.0]))
    assert bool(ensure_eager_tensor(res_sb)[0]) is True
    assert bool(ensure_eager_tensor(res_sb)[2]) is False

    # Case B: backend lacks signbit (fallback x < 0)
    class DummyNoSignbit:
        """Dummy backend without signbit."""

        pass

    res_fallback = _signbit(ensure_mod(DummyNoSignbit()), np.array([-2, 5]))
    assert bool(ensure_eager_tensor(res_fallback)[0]) is True
    assert bool(ensure_eager_tensor(res_fallback)[1]) is False

    # 2. _np_packbits
    # Case A: backend has packbits
    class DummyPackbits:
        """Dummy backend with packbits."""

        def packbits(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock packbits.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "packed"

    assert _np_packbits(ensure_mod(DummyPackbits()), [1, 0]) == "packed"

    # Case B: backend lacks packbits (fallback to numpy.packbits)
    class DummyNoPackbits:
        """Dummy backend without packbits."""

        pass

    arr = np.array([[[1, 0, 1, 1, 0, 0, 1, 0]]], dtype=np.uint8)
    packed_res = _np_packbits(ensure_mod(DummyNoPackbits()), arr, axis=-1)
    assert ensure_eager_tensor(packed_res).shape == (1, 1, 1)

    # 3. _np_unpackbits
    # Case A: backend has unpackbits
    class DummyUnpackbits:
        """Dummy backend with unpackbits."""

        def unpackbits(self, *args: EagerValue, **kwargs: EagerValue) -> str:
            """Mock unpackbits.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return "unpacked"

    assert _np_unpackbits(ensure_mod(DummyUnpackbits()), [180]) == "unpacked"

    # Case B: backend lacks unpackbits (fallback to numpy.unpackbits)
    unpacked_res = _np_unpackbits(ensure_mod(DummyNoPackbits()), np.array([180], dtype=np.uint8))
    assert ensure_callable(len)(unpacked_res) == 8

    # Registry verification
    assert global_eager_registry.get("Signbit") is not None
    assert global_eager_registry.get("Packbits") is not None
    assert global_eager_registry.get("Unpackbits") is not None
