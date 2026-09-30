"""Tests for test_histograms_coverage."""

from __future__ import annotations

from unittest.mock import patch

import ml_switcheroo_compiler.ops.info_and_histograms.histograms as hist_mod


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_histograms_coverage() -> None:
    """Verify 100% line and branch coverage for histograms operations."""
    # Histogram infer_shape
    hist_op = hist_mod.Histogram()
    assert hist_op.infer_shape(bins=DummyWithShape((11,))) == (10,)
    assert hist_op.infer_shape(bins=15) == (15,)
    assert hist_op.infer_shape(bins="auto") == (10,)

    # Histogram2d infer_shape
    hist2d_op = hist_mod.Histogram2d()
    assert hist2d_op.infer_shape(bins=(10, 20)) == (10, 20)
    assert hist2d_op.infer_shape(bins=[DummyWithShape((6,)), DummyWithoutShape()]) == (5, 10)
    assert hist2d_op.infer_shape(bins=[10]) == (10, 10)
    assert hist2d_op.infer_shape(bins=10) == (10, 10)

    # HistogramBinEdges infer_shape
    edges_op = hist_mod.HistogramBinEdges()
    assert edges_op.infer_shape(bins=DummyWithShape((11,))) == (11,)
    assert edges_op.infer_shape(bins=20) == (21,)
    assert edges_op.infer_shape(bins="auto") == (11,)

    # Histogramdd infer_shape
    dd_op = hist_mod.Histogramdd()
    assert dd_op.infer_shape() == ()
    assert dd_op.infer_shape(DummyWithShape((100, 3))) == (10, 10, 10)
    assert dd_op.infer_shape(DummyWithoutShape()) == (10,)

    # Module wrappers dispatch
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", side_effect=lambda op, *args, **kwargs: f"dispatched_{op}"):
        assert hist_mod.histogram(1) == "dispatched_Histogram"
        assert hist_mod.histogram2d(1, 2) == "dispatched_Histogram2d"
        assert hist_mod.histogram_bin_edges(1) == "dispatched_HistogramBinEdges"
        assert hist_mod.histogramdd(1) == "dispatched_Histogramdd"
