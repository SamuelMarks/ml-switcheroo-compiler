"""Tests for numpy eager math stats ops."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.reference_math.math_stats import (
    _np_average,
    _np_confusion_matrix,
    _np_corrcoef,
    _np_correlate,
    _np_cov,
    _np_descriptive,
    _np_distributions,
    _np_histogram2d_,
    _np_histogram_,
    _np_histogram_bin_edges_,
    _np_histogramdd_,
    _np_median_,
    _np_randombernoulli,
    _np_randomcategorical,
    _np_randompermutation,
    _np_randomtruncatednormal,
    _np_randomuniform,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import extract_test_shape, unpack_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_average() -> None:
    """Test function test_average."""
    res = _np_average(np, [1, 2, 3])
    assert res == 2.0


def test_corrcoef() -> None:
    """Test function test_corrcoef."""
    x = np.array(
        [
            [0.1, 0.32, 0.2, 0.4, 0.8],
            [0.23, 0.18, 0.56, 0.61, 0.12],
            [0.9, 0.3, 0.6, 0.5, 0.3],
            [0.34, 0.75, 0.91, 0.19, 0.21],
        ]
    )
    res = _np_corrcoef(np, x)
    assert extract_test_shape(res) == (4, 4)


def test_correlate() -> None:
    """Test function test_correlate."""
    res = _np_correlate(np, [1, 2, 3], [0, 1, 0.5])
    assert extract_test_shape(res) == (1,)


def test_cov() -> None:
    """Test function test_cov."""
    x = np.array(
        [
            [0.1, 0.32, 0.2, 0.4, 0.8],
            [0.23, 0.18, 0.56, 0.61, 0.12],
            [0.9, 0.3, 0.6, 0.5, 0.3],
            [0.34, 0.75, 0.91, 0.19, 0.21],
        ]
    )
    res = _np_cov(np, x)
    assert extract_test_shape(res) == (4, 4)


def test_histogram() -> None:
    """Test function test_histogram."""
    (res, bins) = unpack_pair(_np_histogram_(np, [1, 2, 1], bins=[0, 1, 2, 3]))
    np.testing.assert_array_equal(ensure_ndarray(res), [0, 2, 1])


def test_histogram2d() -> None:
    """Test function test_histogram2d."""
    hist2d_out = _np_histogram2d_(np, [1, 2, 1], [1, 2, 1], bins=[[0, 1, 2, 3], [0, 1, 2, 3]])
    assert isinstance(hist2d_out, tuple)
    (res, xedges, yedges) = hist2d_out
    del xedges, yedges
    assert extract_test_shape(res) == (3, 3)


def test_histogram_bin_edges() -> None:
    """Test function test_histogram_bin_edges."""
    res = _np_histogram_bin_edges_(np, [1, 2, 1], bins=[0, 1, 2, 3])
    np.testing.assert_array_equal(ensure_ndarray(res), [0, 1, 2, 3])


def test_histogramdd() -> None:
    """Test function test_histogramdd."""
    (res, edges) = unpack_pair(_np_histogramdd_(np, ([1, 2, 1], [1, 2, 1]), bins=[[0, 1, 2, 3], [0, 1, 2, 3]]))
    assert extract_test_shape(res) == (3, 3)


def test_median() -> None:
    """Test function test_median."""
    res = _np_median_(np, [1, 2, 3])
    assert res == 2.0


def test_confusion_matrix() -> None:
    """Test function test_confusion_matrix."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])
    res = _np_confusion_matrix(np, y_true, y_pred)
    assert extract_test_shape(res) == (3, 3)
    assert ensure_ndarray(res)[(0, 0)] == 2
    res_kwargs = _np_confusion_matrix(np, labels=y_true, predictions=y_pred, num_classes=3)
    assert extract_test_shape(res_kwargs) == (3, 3)
    with pytest.raises(ValueError, match="Expected labels and predictions"):
        _np_confusion_matrix(np)

    class MockData:
        """Test class MockData."""

        def __init__(self, data: BackendArg) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

    res_mock = _np_confusion_matrix(np, MockData(y_true), MockData(y_pred))
    assert extract_test_shape(res_mock) == (3, 3)


def test_descriptive() -> None:
    """Test function test_descriptive."""
    a = np.array([1, 2, 3, 4, 5])
    res = _np_descriptive(np, a)
    assert isinstance(res, dict)
    assert res["mean"] == 3.0
    assert res["std"] == pytest.approx(1.41421356)
    assert res["min"] == 1
    assert res["max"] == 5
    res_kwargs = _np_descriptive(np, a=a)
    assert isinstance(res_kwargs, dict)
    assert res_kwargs["mean"] == 3.0
    with pytest.raises(ValueError, match="Expected 1 argument"):
        _np_descriptive(np)

    class MockData:
        """Test class MockData."""

        def __init__(self, data: BackendArg) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

    res_mock = _np_descriptive(np, MockData(a))
    assert isinstance(res_mock, dict)
    assert res_mock["mean"] == 3.0


def test_distributions() -> None:
    """Test function test_distributions."""
    res = _np_distributions(np)
    assert ensure_ndarray(res).tolist() == [0.0]


def test_randomcategorical() -> None:
    """Test function disabled_test_randomcategorical."""
    res = _np_randomcategorical(np, "x")
    assert res == "x"


def test_randompermutation() -> None:
    """Test function disabled_test_randompermutation."""
    res = _np_randompermutation(np, "y")
    assert res == "y"


def test_randomtruncatednormal() -> None:
    """Test function test_randomtruncatednormal."""
    res = _np_randomtruncatednormal(np, (2, 2))
    assert extract_test_shape(res) == (2, 2)
    res_kwargs = _np_randomtruncatednormal(np, shape=(2, 2))
    assert extract_test_shape(res_kwargs) == (2, 2)


def test_randombernoulli() -> None:
    """Test function test_randombernoulli."""
    res = _np_randombernoulli(np, (2, 2), 0.5)
    assert extract_test_shape(res) == (2, 2)
    res_kwargs = _np_randombernoulli(np, shape=(2, 2), p=0.5)
    assert extract_test_shape(res_kwargs) == (2, 2)


def test_randomuniform() -> None:
    """Test function test_randomuniform."""
    res = _np_randomuniform(np, (2, 2), 0.0, 1.0)
    assert extract_test_shape(res) == (2, 2)
    res_kwargs = _np_randomuniform(np, shape=(2, 2), minval=0.0, maxval=1.0)
    assert extract_test_shape(res_kwargs) == (2, 2)
