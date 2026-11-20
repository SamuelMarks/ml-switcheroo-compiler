"""Tests for test_window_reductions."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.backends.eager.reference_math.window_reductions as wr
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import EagerTensorProtocol, adapt_backend_module, coerce_operand_list, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


class _TestWindowReductionsCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_window_reductions_coverage` split helpers."""

    data: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]
    input_2d: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]
    res: EagerTensorProtocol | OpValue
    segment_ids: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]


def _test_window_reductions_coverage_part_1(_state: _TestWindowReductionsCoverageState) -> None:
    """Execute part 1 of test_window_reductions_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    res = wr._np_adaptive_avg_pool2d(np, np.zeros((1, 1, 2, 2)), (1, 1))
    assert extract_test_shape(res) == (1, 1, 1, 1)
    res = wr._np_adaptive_avg_pool2d(np, 1.0, (1, 1))
    assert res == 1.0

    # Verify mathematical correctness of 2D adaptive pooling
    input_2d = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # shape (1, 1, 2, 2)
    # Target (1, 1) output should average the entire 2x2 spatial area to 2.5
    res_math = wr._np_adaptive_avg_pool2d(np, input_2d, (1, 1))
    assert np.allclose(ensure_ndarray(res_math), [[[[2.5]]]])

    # Target (2, 2) output should map exactly to 1x1 cells, preserving the input
    res_math_2x2 = wr._np_adaptive_avg_pool2d(np, input_2d, (2, 2))
    assert np.allclose(ensure_ndarray(res_math_2x2), input_2d)

    # AdaptiveAvgPool3D
    res = wr._np_adaptive_avg_pool3d(np, np.zeros((1, 1, 2, 2, 2)), (1, 1, 1))
    assert extract_test_shape(res) == (1, 1, 1, 1, 1)
    res = wr._np_adaptive_avg_pool3d(np, 1.0, (1, 1, 1))
    assert res == 1.0

    # Verify mathematical correctness of 3D adaptive pooling
    input_3d = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])  # shape (1, 1, 2, 2, 2)
    res_math_3d = wr._np_adaptive_avg_pool3d(np, input_3d, (1, 1, 1))
    assert np.allclose(ensure_ndarray(res_math_3d), [[[[[4.5]]]]])

    # AdaptiveMaxPool3D
    res = wr._np_adaptive_max_pool3d(np, np.zeros((1, 1, 2, 2, 2)), (1, 1, 1))
    assert extract_test_shape(res) == (1, 1, 1, 1, 1)
    res = wr._np_adaptive_max_pool3d(np, 1.0, (1, 1, 1))
    assert res == 1.0

    # AdaptiveMaxPool3D_Indices
    res = wr._np_adaptive_max_pool3d_indices(np, np.zeros((1, 1, 2, 2, 2)), (1, 1, 1))
    assert len(coerce_operand_list(res)) == 2
    _state.update({"input_2d": input_2d, "res": res})


def _test_window_reductions_coverage_part_2(_state: _TestWindowReductionsCoverageState) -> None:
    """Execute part 2 of test_window_reductions_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    input_2d, res = _state["input_2d"], _state["res"]
    assert isinstance(res, (tuple, list))
    assert adapt_backend_module(res[0]).require_symbol("shape") == (1, 1, 1, 1, 1)

    # AdaptiveMaxPool2D
    res = wr._np_adaptive_max_pool2d(np, np.zeros((1, 1, 2, 2)), (1, 1))
    assert adapt_backend_module(res).require_symbol("shape") == (1, 1, 1, 1)
    res = wr._np_adaptive_max_pool2d(np, 1.0, (1, 1))
    assert res == 1.0

    # Verify mathematical correctness of 2D adaptive max pooling
    res_max_math = wr._np_adaptive_max_pool2d(np, input_2d, (1, 1))
    assert np.allclose(ensure_ndarray(res_max_math), [[[[4.0]]]])

    # FractionalAvgPool
    res = wr._np_fractional_avg_pool(np, np.zeros((1, 1, 2, 2)))
    assert adapt_backend_module(res).require_symbol("shape") == (1, 1, 2, 2)

    # FractionalMaxPool
    res = wr._np_fractional_max_pool(np, np.zeros((1, 1, 2, 2)))
    assert adapt_backend_module(res).require_symbol("shape") == (1, 1, 2, 2)

    # SegmentMax
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    segment_ids = np.array([0, 0])
    res = wr._np_segment_max(np, data, segment_ids)
    assert adapt_backend_module(res).require_symbol("shape") == (1, 2)
    res = wr._np_segment_max(np, data, segment_ids, num_segments=2)
    assert adapt_backend_module(res).require_symbol("shape") == (2, 2)

    # SegmentMin
    res = wr._np_segment_min(np, data, segment_ids)
    assert adapt_backend_module(res).require_symbol("shape") == (1, 2)
    res = wr._np_segment_min(np, data, segment_ids, num_segments=2)
    assert adapt_backend_module(res).require_symbol("shape") == (2, 2)

    # SegmentProd
    res = wr._np_segment_prod(np, data, segment_ids)
    _state.update({"data": data, "res": res, "segment_ids": segment_ids})
    _state.update({"input_2d": input_2d})


def _test_window_reductions_coverage_part_3(_state: _TestWindowReductionsCoverageState) -> None:
    """Execute part 3 of test_window_reductions_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    data, res, segment_ids = _state["data"], _state["res"], _state["segment_ids"]
    assert adapt_backend_module(res).require_symbol("shape") == (1, 2)
    res = wr._np_segment_prod(np, data, segment_ids, num_segments=2)
    assert adapt_backend_module(res).require_symbol("shape") == (2, 2)

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.backends.eager.reference_math.window_reductions._reduce_window") as mock_rw:
        mock_rw.return_value = 1
        res = wr._np_reduce_window(np)
        assert res == 1

    with patch("ml_switcheroo_compiler.backends.eager.reference_math.window_reductions._segment_sum") as mock_ss:
        mock_ss.return_value = 1
        res = wr._np_segment_sum(np)
        assert res == 1
    _state.update({"res": res})


def test_window_reductions_coverage() -> None:
    # AdaptiveAvgPool2D
    """Test function test_window_reductions_coverage."""
    _state: _TestWindowReductionsCoverageState = {}
    _test_window_reductions_coverage_part_1(_state)
    _test_window_reductions_coverage_part_2(_state)
    _test_window_reductions_coverage_part_3(_state)
    assert len(_state) > 0
