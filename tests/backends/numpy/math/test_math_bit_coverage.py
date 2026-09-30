"""Tests for test_math_bit_coverage."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_bit as bit_module


def test_math_bit_ops() -> None:
    """Test bitwise operations directly and through eager registry.

    Returns:
        None
    """
    # Clz - non-integer raises TypeError
    with pytest.raises(TypeError, match="Clz requires integer inputs"):
        bit_module._np_clz(np, np.array([1.0, 2.0]))

    # Clz - positive and negative integers
    pos_ints = np.array([1, 16, 0], dtype=np.int32)
    clz_pos = bit_module._np_clz(np, pos_ints)
    assert clz_pos[0] == 31
    assert clz_pos[1] == 27
    assert clz_pos[2] == 32

    neg_ints = np.array([-1, -16], dtype=np.int32)
    clz_neg = bit_module._np_clz(np, neg_ints)
    assert clz_neg[0] == 0
    assert clz_neg[1] == 0

    reg_clz = numpy_eager_registry.get("Clz")
    reg_pos = reg_clz(np, pos_ints)
    assert reg_pos[0] == 31

    # BitcastConvertType
    arr = np.array([1.0], dtype=np.float32)
    bitcast_i32 = bit_module._np_bitcast_convert_type(np, arr, "int32")
    assert bitcast_i32.dtype == np.int32

    bitcast_fallback = bit_module._np_bitcast_convert_type(np, arr, "unknown_custom_dtype")
    assert bitcast_fallback.dtype == np.float32

    reg_bitcast = numpy_eager_registry.get("BitcastConvertType")
    assert reg_bitcast(np, arr, "int32").dtype == np.int32

    # Packbits and Unpackbits
    binary_arr = np.array([[[1, 0, 1, 0, 0, 0, 0, 0]]], dtype=np.uint8)
    packed = bit_module._np_packbits(np, binary_arr)
    assert packed.dtype == np.uint8
    unpacked = bit_module._np_unpackbits(np, packed)
    assert unpacked.dtype == np.uint8

    reg_pack = numpy_eager_registry.get("Packbits")
    assert reg_pack(np, binary_arr).dtype == np.uint8
    reg_unpack = numpy_eager_registry.get("Unpackbits")
    assert reg_unpack(np, packed).dtype == np.uint8

    # BitwiseCount
    count_input = np.array([0, 1, 3, 7], dtype=np.int32)
    counts = bit_module._np_bitwise_count(np, count_input)
    assert np.array_equal(counts, [0, 1, 2, 3])
    reg_bw_count = numpy_eager_registry.get("BitwiseCount")
    assert np.array_equal(reg_bw_count(np, count_input), [0, 1, 2, 3])
