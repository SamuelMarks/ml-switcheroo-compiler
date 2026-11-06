"""Tests for test_window_reductions."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.window_reductions as wr
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_window_reductions_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_window_reductions_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = wr._np_adaptive_avg_pool2d(ensure_eager_tensor(np), np.zeros((1, 1, 2, 2)), (1, 1))
    assert ensure_eager_tensor(res).shape == (1, 1, 1, 1)
    res = wr._np_adaptive_avg_pool2d(ensure_eager_tensor(np), 1.0, (1, 1))
    assert res == 1.0

    # Verify mathematical correctness of 2D adaptive pooling
    input_2d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # shape (1, 1, 2, 2)
    # Target (1, 1) output should average the entire 2x2 spatial area to 2.5
    res_math = wr._np_adaptive_avg_pool2d(ensure_eager_tensor(np), input_2d, (1, 1))
    assert ensure_callable(np.allclose)(res_math, [[[[2.5]]]])

    # Target (2, 2) output should map exactly to 1x1 cells, preserving the input
    res_math_2x2 = wr._np_adaptive_avg_pool2d(ensure_eager_tensor(np), input_2d, (2, 2))
    assert ensure_callable(np.allclose)(res_math_2x2, input_2d)

    # AdaptiveAvgPool3D
    res = wr._np_adaptive_avg_pool3d(ensure_eager_tensor(np), np.zeros((1, 1, 2, 2, 2)), (1, 1, 1))
    assert ensure_eager_tensor(res).shape == (1, 1, 1, 1, 1)
    res = wr._np_adaptive_avg_pool3d(ensure_eager_tensor(np), 1.0, (1, 1, 1))
    assert res == 1.0

    # Verify mathematical correctness of 3D adaptive pooling
    input_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])  # shape (1, 1, 2, 2, 2)
    res_math_3d = wr._np_adaptive_avg_pool3d(ensure_eager_tensor(np), input_3d, (1, 1, 1))
    assert ensure_callable(np.allclose)(res_math_3d, [[[[[4.5]]]]])

    # AdaptiveMaxPool3D
    res = wr._np_adaptive_max_pool3d(ensure_eager_tensor(np), np.zeros((1, 1, 2, 2, 2)), (1, 1, 1))
    assert ensure_eager_tensor(res).shape == (1, 1, 1, 1, 1)
    res = wr._np_adaptive_max_pool3d(ensure_eager_tensor(np), 1.0, (1, 1, 1))
    assert res == 1.0

    # AdaptiveMaxPool3D_Indices
    res = wr._np_adaptive_max_pool3d_indices(ensure_eager_tensor(np), np.zeros((1, 1, 2, 2, 2)), (1, 1, 1))
    assert ensure_callable(len)(res) == 2
    _state.update(ensure_tensor_dict({"input_2d": input_2d, "input_3d": input_3d, "res": res, "res_math": res_math}))
    _state.update(ensure_tensor_dict({"res_math_2x2": res_math_2x2, "res_math_3d": res_math_3d}))


def _test_window_reductions_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_window_reductions_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    input_2d, res = _state["input_2d"], _state["res"]
    assert res[0].shape == (1, 1, 1, 1, 1)

    # AdaptiveMaxPool2D
    res = ensure_eager_tensor(wr._np_adaptive_max_pool2d(ensure_eager_tensor(np), np.zeros((1, 1, 2, 2)), (1, 1)))
    assert res.shape == (1, 1, 1, 1)
    res = ensure_eager_tensor(wr._np_adaptive_max_pool2d(ensure_eager_tensor(np), 1.0, (1, 1)))
    assert res == 1.0

    # Verify mathematical correctness of 2D adaptive max pooling
    res_max_math = wr._np_adaptive_max_pool2d(ensure_eager_tensor(np), input_2d, (1, 1))
    assert ensure_callable(np.allclose)(res_max_math, [[[[4.0]]]])

    # FractionalAvgPool
    res = ensure_eager_tensor(wr._np_fractional_avg_pool(ensure_eager_tensor(np), np.zeros((1, 1, 2, 2))))
    assert res.shape == (1, 1, 2, 2)

    # FractionalMaxPool
    res = ensure_eager_tensor(wr._np_fractional_max_pool(ensure_eager_tensor(np), np.zeros((1, 1, 2, 2))))
    assert res.shape == (1, 1, 2, 2)

    # SegmentMax
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    segment_ids = np.array([0, 0])
    res = ensure_eager_tensor(wr._np_segment_max(ensure_eager_tensor(np), data, segment_ids))
    assert res.shape == (1, 2)
    res = ensure_eager_tensor(wr._np_segment_max(ensure_eager_tensor(np), data, segment_ids, num_segments=2))
    assert res.shape == (2, 2)

    # SegmentMin
    res = ensure_eager_tensor(wr._np_segment_min(ensure_eager_tensor(np), data, segment_ids))
    assert res.shape == (1, 2)
    res = ensure_eager_tensor(wr._np_segment_min(ensure_eager_tensor(np), data, segment_ids, num_segments=2))
    assert res.shape == (2, 2)

    # SegmentProd
    res = ensure_eager_tensor(wr._np_segment_prod(ensure_eager_tensor(np), data, segment_ids))
    _state.update(ensure_tensor_dict({"data": data, "res": res, "res_max_math": res_max_math, "segment_ids": segment_ids}))
    _state.update(ensure_tensor_dict({"input_2d": input_2d}))


def _test_window_reductions_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_window_reductions_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    data, res, segment_ids = _state["data"], _state["res"], _state["segment_ids"]
    assert res.shape == (1, 2)
    res = ensure_eager_tensor(wr._np_segment_prod(ensure_eager_tensor(np), data, segment_ids, num_segments=2))
    assert res.shape == (2, 2)

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.backends.numpy.eager.window_reductions._reduce_window") as mock_rw:
        mock_rw.return_value = 1
        res = ensure_eager_tensor(wr._np_reduce_window(ensure_eager_tensor(np)))
        assert res == 1

    with patch("ml_switcheroo_compiler.backends.numpy.eager.window_reductions._segment_sum") as mock_ss:
        mock_ss.return_value = 1
        res = ensure_eager_tensor(wr._np_segment_sum(ensure_eager_tensor(np)))
        assert res == 1
    _state.update(ensure_tensor_dict({"mock_rw": mock_rw, "mock_ss": mock_ss, "patch": patch, "res": res}))


def test_window_reductions_coverage() -> None:
    # AdaptiveAvgPool2D
    """Test function test_window_reductions_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_window_reductions_coverage_part_1(_state)
    _test_window_reductions_coverage_part_2(_state)
    _test_window_reductions_coverage_part_3(_state)
    assert len(_state) > 0
