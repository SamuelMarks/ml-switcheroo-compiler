"""Tests for test_info_ops_coverage."""

from __future__ import annotations

from unittest.mock import patch

import ml_switcheroo_compiler.ops.info_and_histograms.info as info_mod


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


def test_info_operations_coverage() -> None:
    """Verify 100% line and branch coverage for info operations."""
    # Test OpDef infer_shape methods
    assert info_mod.Finfo().infer_shape() == ()
    assert info_mod.Iinfo().infer_shape() == ()
    assert info_mod.GetPrintoptions().infer_shape() == ()
    assert info_mod.Isscalar().infer_shape() == ()
    assert info_mod.Iterable().infer_shape() == ()
    assert info_mod.PromoteTypes().infer_shape() == ()
    assert info_mod.ResultType().infer_shape() == ()

    # Test module-level wrapper dispatchers
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", side_effect=lambda op, *args, **kwargs: f"dispatched_{op}"):
        assert info_mod.iinfo("int32") == "dispatched_Iinfo"
        assert info_mod.isscalar(5) == "dispatched_Isscalar"
        assert info_mod.iterable([1, 2]) == "dispatched_Iterable"
        assert info_mod.promote_types("int32", "float32") == "dispatched_PromoteTypes"
        assert info_mod.result_type("int32", "float32") == "dispatched_ResultType"
        assert info_mod.get_printoptions() == "dispatched_GetPrintoptions"
