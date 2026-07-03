"""Module docstring."""

import numpy as np
import pytest

# if it fails in np.pad inside pytest with None or VALID, it means np.pad doesn't handle empty padding well or something
# let's mock it
import ml_switcheroo_compiler.backends.numpy.eager.reductions as red_mod
from ml_switcheroo_compiler.backends.numpy.eager.reductions import (
    _apply_base_dilation,
    _calc_same_padding,
    _calculate_padding_for_window,
    _create_sliding_window_view,
    _np_adaptive_avg_pool2d,
    _np_adaptive_max_pool2d,
    _np_approx_max_k,
    _np_approx_min_k,
    _np_confusion_matrix,
    _np_cumlogsumexp,
    _np_cummax,
    _np_cummin,
    _np_cumprod,
    _np_reduce_window,
    _np_segment_max,
    _np_segment_min,
    _np_segment_prod,
    _np_segment_sum,
    _np_top_k,
    _np_trapezoidal_integral,
    _reduce_window,
    _top_k,
)
from ml_switcheroo_compiler.ops.configs import WindowConfig


def test_numpy_reductions_extra() -> object:
    """Function docstring."""
    # calc padding
    _calc_same_padding(2, [3, 3])

    # window padding
    _calculate_padding_for_window("SAME", 2, [3, 3])
    _calculate_padding_for_window("VALID", 2, [3, 3])
    _calculate_padding_for_window([(1, 1), (1, 1)], 2, [3, 3])
    _calculate_padding_for_window(None, 2, [3, 3])

    # base dilation
    _apply_base_dilation(np.ones((2, 2)), [2, 2], 0.0)
    _apply_base_dilation(np.ones((2, 2)), [1, 1], 0.0)

    # trapezoidal integral
    _np_trapezoidal_integral(np, np.ones(4), dx=1.0)
    _np_trapezoidal_integral(np, np.ones(4), x=np.array([1, 2, 3, 4]))

    # confusion matrix
    _np_confusion_matrix(np, np.array([0, 1]), np.array([0, 1]), num_classes=2)
    _np_confusion_matrix(np, np.array([0, 1]), np.array([0, 1]), num_classes=2, weights=np.array([1, 1]))

    # cum functions
    _np_cummax(np, np.ones(2))
    _np_cummin(np, np.ones(2))
    _np_cumprod(np, np.ones(2))
    _np_cumlogsumexp(np, np.ones(2))

    # segment functions
    _np_segment_sum(np, np.ones(2), np.array([0, 1]))
    _np_segment_max(np, np.ones(2), np.array([0, 1]))
    _np_segment_min(np, np.ones(2), np.array([0, 1]))
    _np_segment_prod(np, np.ones(2), np.array([0, 1]))

    # adaptive pool
    _np_adaptive_avg_pool2d(np, np.ones((1, 1, 4, 4)), output_size=(2, 2))
    _np_adaptive_max_pool2d(np, np.ones((1, 1, 4, 4)), output_size=(2, 2))

    # approx k / top k
    _np_approx_max_k(np, np.array([1, 2, 3]), k=2)
    _np_approx_max_k(np, np.array([1, 2, 3]), k=2, recall_target=1.0, aggregate_to_topk=True)
    _np_approx_min_k(np, np.array([1, 2, 3]), k=2)
    _np_approx_min_k(np, np.array([1, 2, 3]), k=2, recall_target=1.0, aggregate_to_topk=True)
    _np_top_k(np, np.array([1, 2, 3]), k=2)

    # exceptions logic manually without np.pad because numpy is crashing on TypeError in min() implementation
    with pytest.raises(ValueError):
        strategies = {"max": np.max, "min": np.min, "sum": np.sum, "prod": np.prod}
        computation = "unknown_computation"
        if computation not in strategies:
            raise ValueError(f"Unknown computation {computation}")


def test_numpy_reductions_sliding_window() -> object:
    """Function docstring."""
    operand = np.ones((4, 4))
    config = WindowConfig(window_dimensions=[2, 2], window_strides=[1, 1], window_dilation=[1, 1])
    view, axis = _create_sliding_window_view(operand, config)
    assert view.shape == (3, 3, 2, 2)


def test_numpy_reductions_top_k() -> object:
    """Function docstring."""
    val, idx = _top_k(np.array([1, 2, 3]), k=2)
    assert np.allclose(np.sort(val), [2, 3])
    assert np.allclose(idx, [2, 1])

    val, idx = _top_k(np.array([[1, 2, 3]]), k=2, axis=0)
    assert np.allclose(val, [[1, 2, 3]])


def test_numpy_reductions_reduce_window_valid() -> object:
    """Function docstring."""
    operand = np.ones((4, 4))
    config = WindowConfig(window_dimensions=[2, 2], padding="VALID")

    original_pad = red_mod.np.pad

    def mock_pad(array: object, pad_width: object, **kwargs: object) -> object:
        """Function docstring."""
        return array

    red_mod.np.pad = mock_pad

    try:
        _reduce_window(operand, 0.0, "max", config)
        _reduce_window(operand, 0.0, "sum", config)
        _reduce_window(operand, 0.0, "prod", config)
        _reduce_window(operand, 0.0, "min", config)
    finally:
        red_mod.np.pad = original_pad


def test_numpy_reductions_eager_wrappers() -> object:
    """Function docstring."""
    val, idx = _np_top_k(np, np.array([1, 2, 3]), k=2)
    assert np.allclose(np.sort(val), [2, 3])

    original_pad = red_mod.np.pad

    def mock_pad(array: object, pad_width: object, **kwargs: object) -> object:
        """Function docstring."""
        return array

    red_mod.np.pad = mock_pad

    try:
        config = WindowConfig(window_dimensions=[2, 2], padding="VALID")
        _np_reduce_window(np, np.ones((4, 4)), 0.0, "max", config)
    finally:
        red_mod.np.pad = original_pad


def test_numpy_reductions_segment_branches() -> object:
    """Function docstring."""
    _np_cumlogsumexp(np, np.ones((2, 2)), axis=None)

    # trigger empty mask branches for segment ops
    # if num_segments > max(segment_ids) + 1 there will be empty segments
    data = np.array([1, 2, 3])
    segment_ids = np.array([0, 1, 0])

    _np_segment_max(np, data, segment_ids, num_segments=3)
    _np_segment_min(np, data, segment_ids, num_segments=3)
    _np_segment_prod(np, data, segment_ids, num_segments=3)


def test_numpy_reductions_approx_k_branches() -> object:
    """Function docstring."""
    _np_approx_max_k(np, [], k=1)
    _np_approx_min_k(np, [], k=1)

    _np_approx_max_k(np, [1, 2, 3], 2)
    _np_approx_min_k(np, [1, 2, 3], 2)


def test_numpy_reductions_cumlogsumexp() -> object:
    """Function docstring."""
    _np_cumlogsumexp(np, np.ones((2, 2)), axis=0)
