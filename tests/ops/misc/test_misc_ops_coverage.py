"""Tests for test_misc_ops_coverage."""

from __future__ import annotations

from unittest.mock import patch

import ml_switcheroo_compiler.ops.info_and_histograms.misc_ops as misc_ops_mod


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy object with shape.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize dummy object without shape."""
        self.val: int = 42


def test_misc_ops_full_coverage() -> None:
    """Test full coverage for miscellaneous information and infeed ops."""
    infeed_op = misc_ops_mod.Infeed()
    assert infeed_op.infer_shape(shape=(5, 10)) == (5, 10)
    assert infeed_op.infer_shape(shape=[5, 10]) == (5, 10)
    assert infeed_op.infer_shape() == ()

    vectorize_op = misc_ops_mod.Vectorize()
    assert vectorize_op.infer_shape() == ()

    axis_index_op = misc_ops_mod.AxisIndex()
    assert axis_index_op.infer_shape() == ()
    assert axis_index_op.infer_shape(DummyWithShape((3, 7))) == (3, 7)
    assert axis_index_op.infer_shape(DummyWithShape([3, 7])) == (3, 7)  # type: ignore[arg-type]
    assert axis_index_op.infer_shape(DummyWithoutShape()) == ()

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="infeed_val") as mock_disp:
        res_infeed = misc_ops_mod.infeed(shape=(2, 2))
        assert res_infeed == "infeed_val"
        mock_disp.assert_called_once_with("Infeed", shape=(2, 2))

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="vec_val") as mock_disp:
        res_vec = misc_ops_mod.vectorize(dummy_arg=1)
        assert res_vec == "vec_val"
        mock_disp.assert_called_once_with("Vectorize", dummy_arg=1)
