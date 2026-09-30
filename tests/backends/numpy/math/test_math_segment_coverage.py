"""Tests for test_math_segment_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_segment as math_segment


def test_math_segment() -> None:
    """Test SegmentSum and SparseSegmentSum directly and via registry."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int64)

    # Branch 1: num_segments is None
    res_none = math_segment._np_segment_sum(np, data, segment_ids, num_segments=None)
    assert res_none.shape == (2, 2)
    assert np.allclose(res_none[0], [4.0, 6.0])
    assert np.allclose(res_none[1], [5.0, 6.0])

    # Branch 2: num_segments is provided
    res_num = math_segment._np_segment_sum(np, data, segment_ids, num_segments=4)
    assert res_num.shape == (4, 2)
    assert np.allclose(res_num[0], [4.0, 6.0])
    assert np.allclose(res_num[1], [5.0, 6.0])
    assert np.allclose(res_num[2], [0.0, 0.0])
    assert np.allclose(res_num[3], [0.0, 0.0])

    # Registry SegmentSum
    reg_seg = numpy_eager_registry.get("SegmentSum")
    res_reg = reg_seg(np, data, segment_ids)
    assert np.allclose(res_reg, res_none)

    # SparseSegmentSum directly
    sparse_data = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    sparse_sum = math_segment._np_sparsesegmentsum(np, sparse_data)
    assert np.allclose(sparse_sum, [[6.0, 8.0]])
