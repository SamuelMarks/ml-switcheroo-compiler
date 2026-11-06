"""Tests for test_math_bit_coverage."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_bit as bit_module
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_math_bit_ops() -> None:
    """Test bitwise operations directly and through eager registry.

    Returns:
        None
    """
    # Clz - non-integer raises TypeError
    with pytest.raises(TypeError, match="Clz requires integer inputs"):
        bit_module._np_clz(ensure_eager_tensor(np), np.array([1.0, 2.0]))

    # Clz - positive and negative integers
    pos_ints = np.array([1, 16, 0], dtype=np.int32)
    clz_pos = bit_module._np_clz(ensure_eager_tensor(np), pos_ints)
    assert ensure_eager_tensor(clz_pos)[0] == 31
    assert ensure_eager_tensor(clz_pos)[1] == 27
    assert ensure_eager_tensor(clz_pos)[2] == 32

    neg_ints = np.array([-1, -16], dtype=np.int32)
    clz_neg = bit_module._np_clz(ensure_eager_tensor(np), neg_ints)
    assert ensure_eager_tensor(clz_neg)[0] == 0
    assert ensure_eager_tensor(clz_neg)[1] == 0

    reg_clz = numpy_eager_registry.get("Clz")
    reg_pos = ensure_callable(reg_clz)(np, pos_ints)
    assert ensure_eager_tensor(reg_pos)[0] == 31

    # BitcastConvertType
    arr = np.array([1.0], dtype=np.float32)
    bitcast_i32 = bit_module._np_bitcast_convert_type(ensure_eager_tensor(np), arr, "int32")
    assert ensure_eager_tensor(ensure_eager_tensor(bitcast_i32).dtype) == np.int32

    bitcast_fallback = bit_module._np_bitcast_convert_type(ensure_eager_tensor(np), arr, "unknown_custom_dtype")
    assert ensure_eager_tensor(ensure_eager_tensor(bitcast_fallback).dtype) == np.float32

    reg_bitcast = numpy_eager_registry.get("BitcastConvertType")
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_callable(reg_bitcast)(np, arr, "int32")).dtype) == np.int32

    # Packbits and Unpackbits
    binary_arr = np.array([[[1, 0, 1, 0, 0, 0, 0, 0]]], dtype=np.uint8)
    packed = bit_module._np_packbits(ensure_eager_tensor(np), binary_arr)
    assert ensure_eager_tensor(ensure_eager_tensor(packed).dtype) == np.uint8
    unpacked = bit_module._np_unpackbits(ensure_eager_tensor(np), packed)
    assert ensure_eager_tensor(ensure_eager_tensor(unpacked).dtype) == np.uint8

    reg_pack = numpy_eager_registry.get("Packbits")
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_callable(reg_pack)(np, binary_arr)).dtype) == np.uint8
    reg_unpack = numpy_eager_registry.get("Unpackbits")
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_callable(reg_unpack)(np, packed)).dtype) == np.uint8

    # BitwiseCount
    count_input = np.array([0, 1, 3, 7], dtype=np.int32)
    counts = bit_module._np_bitwise_count(ensure_eager_tensor(np), count_input)
    assert ensure_callable(np.array_equal)(counts, [0, 1, 2, 3])
    reg_bw_count = numpy_eager_registry.get("BitwiseCount")
    assert ensure_callable(np.array_equal)(ensure_callable(reg_bw_count)(np, count_input), [0, 1, 2, 3])
