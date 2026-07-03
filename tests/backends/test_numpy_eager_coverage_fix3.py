"""Module docstring."""

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_misc as mod


def test_segment_sum_coverage() -> object:
    """Function docstring."""
    data = np.array([1, 2, 3, 4, 5])
    segment_ids = np.array([0, 0, 1, 1, 2])

    res1 = mod._np_segment_sum(np, data, segment_ids)
    assert np.array_equal(res1, [3, 7, 5])

    res2 = mod._np_segment_sum(np, data, segment_ids, num_segments=4)
    assert np.array_equal(res2, [3, 7, 5, 0])
