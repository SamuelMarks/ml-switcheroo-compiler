"""Behavioral unit tests for math bit."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._math_advanced_test_ns import math_bit as bit_module
from tests.testing_utils import extract_test_dtype, require_callable

if typing.TYPE_CHECKING:
    pass


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
    assert ensure_ndarray(clz_pos)[0] == 31
    assert ensure_ndarray(clz_pos)[1] == 27
    assert ensure_ndarray(clz_pos)[2] == 32

    neg_ints = np.array([-1, -16], dtype=np.int32)
    clz_neg = bit_module._np_clz(np, neg_ints)
    assert ensure_ndarray(clz_neg)[0] == 0
    assert ensure_ndarray(clz_neg)[1] == 0

    reg_clz = numpy_eager_registry.get("Clz")
    reg_pos = require_callable(reg_clz)(np, pos_ints)
    assert reg_pos[0] == 31

    # BitcastConvertType
    arr = np.array([1.0], dtype=np.float32)
    bitcast_i32 = bit_module._np_bitcast_convert_type(np, arr, "int32")
    assert extract_test_dtype(bitcast_i32) == np.int32

    bitcast_fallback = bit_module._np_bitcast_convert_type(np, arr, "unknown_custom_dtype")
    assert extract_test_dtype(bitcast_fallback) == np.float32

    reg_bitcast = numpy_eager_registry.get("BitcastConvertType")
    assert require_callable(reg_bitcast)(np, arr, "int32").dtype == np.int32

    # Packbits and Unpackbits
    binary_arr = np.array([[[1, 0, 1, 0, 0, 0, 0, 0]]], dtype=np.uint8)
    packed = bit_module._np_packbits(np, binary_arr)
    assert extract_test_dtype(packed) == np.uint8
    unpacked = bit_module._np_unpackbits(np, packed)
    assert extract_test_dtype(unpacked) == np.uint8

    reg_pack = numpy_eager_registry.get("Packbits")
    assert require_callable(reg_pack)(np, binary_arr).dtype == np.uint8
    reg_unpack = numpy_eager_registry.get("Unpackbits")
    assert require_callable(reg_unpack)(np, packed).dtype == np.uint8

    # BitwiseCount
    count_input = np.array([0, 1, 3, 7], dtype=np.int32)
    counts = bit_module._np_bitwise_count(np, count_input)
    assert np.array_equal(ensure_ndarray(counts), [0, 1, 2, 3])
    reg_bw_count = numpy_eager_registry.get("BitwiseCount")
    assert np.array_equal(require_callable(reg_bw_count)(np, count_input), [0, 1, 2, 3])
