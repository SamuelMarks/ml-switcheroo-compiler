"""Tests for numpy eager math stats ops."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.math_stats import (
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


def test_average() -> None:
    res = _np_average(np, [1, 2, 3])
    assert res == 2.0


def test_corrcoef() -> None:
    x = np.array([[0.1, 0.32, 0.2, 0.4, 0.8], [0.23, 0.18, 0.56, 0.61, 0.12], [0.9, 0.3, 0.6, 0.5, 0.3], [0.34, 0.75, 0.91, 0.19, 0.21]])
    res = _np_corrcoef(np, x)
    assert res.shape == (4, 4)


def test_correlate() -> None:
    res = _np_correlate(np, [1, 2, 3], [0, 1, 0.5])
    assert res.shape == (1,)


def test_cov() -> None:
    x = np.array([[0.1, 0.32, 0.2, 0.4, 0.8], [0.23, 0.18, 0.56, 0.61, 0.12], [0.9, 0.3, 0.6, 0.5, 0.3], [0.34, 0.75, 0.91, 0.19, 0.21]])
    res = _np_cov(np, x)
    assert res.shape == (4, 4)


def test_histogram() -> None:
    res, bins = _np_histogram_(np, [1, 2, 1], bins=[0, 1, 2, 3])
    np.testing.assert_array_equal(res, [0, 2, 1])


def test_histogram2d() -> None:
    res, xedges, yedges = _np_histogram2d_(np, [1, 2, 1], [1, 2, 1], bins=[[0, 1, 2, 3], [0, 1, 2, 3]])
    assert res.shape == (3, 3)


def test_histogram_bin_edges() -> None:
    res = _np_histogram_bin_edges_(np, [1, 2, 1], bins=[0, 1, 2, 3])
    np.testing.assert_array_equal(res, [0, 1, 2, 3])


def test_histogramdd() -> None:
    res, edges = _np_histogramdd_(np, ([1, 2, 1], [1, 2, 1]), bins=[[0, 1, 2, 3], [0, 1, 2, 3]])
    assert res.shape == (3, 3)


def test_median() -> None:
    res = _np_median_(np, [1, 2, 3])
    assert res == 2.0


def test_confusion_matrix() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])
    res = _np_confusion_matrix(np, y_true, y_pred)
    assert res.shape == (3, 3)
    assert res[0, 0] == 2

    # kwargs
    res_kwargs = _np_confusion_matrix(np, labels=y_true, predictions=y_pred, num_classes=3)
    assert res_kwargs.shape == (3, 3)

    # missing args
    with pytest.raises(ValueError, match="Expected labels and predictions"):
        _np_confusion_matrix(np)

    # mock data attribute
    class MockData:
        def __init__(self, data):
            self.data = data

    res_mock = _np_confusion_matrix(np, MockData(y_true), MockData(y_pred))
    assert res_mock.shape == (3, 3)


def test_descriptive() -> None:
    a = np.array([1, 2, 3, 4, 5])
    res = _np_descriptive(np, a)
    assert res["mean"] == 3.0
    assert res["std"] == pytest.approx(1.41421356)
    assert res["min"] == 1
    assert res["max"] == 5

    res_kwargs = _np_descriptive(np, a=a)
    assert res_kwargs["mean"] == 3.0

    with pytest.raises(ValueError, match="Expected 1 argument"):
        _np_descriptive(np)

    # mock data attribute
    class MockData:
        def __init__(self, data):
            self.data = data

    res_mock = _np_descriptive(np, MockData(a))
    assert res_mock["mean"] == 3.0


def test_distributions() -> None:
    res = _np_distributions(np)
    assert res.tolist() == [0.0]


def disabled_test_randomcategorical() -> None:
    res = _np_randomcategorical(np, "x")
    assert res == "x"


def disabled_test_randompermutation() -> None:
    res = _np_randompermutation(np, "y")
    assert res == "y"


def test_randomtruncatednormal() -> None:
    res = _np_randomtruncatednormal(np, (2, 2))
    assert res.shape == (2, 2)

    res_kwargs = _np_randomtruncatednormal(np, shape=(2, 2))
    assert res_kwargs.shape == (2, 2)


def test_randombernoulli() -> None:
    res = _np_randombernoulli(np, (2, 2), 0.5)
    assert res.shape == (2, 2)

    res_kwargs = _np_randombernoulli(np, shape=(2, 2), p=0.5)
    assert res_kwargs.shape == (2, 2)


def test_randomuniform() -> None:
    res = _np_randomuniform(np, (2, 2), 0.0, 1.0)
    assert res.shape == (2, 2)

    res_kwargs = _np_randomuniform(np, shape=(2, 2), minval=0.0, maxval=1.0)
    assert res_kwargs.shape == (2, 2)
