"""Tests for test_ragged_core_coverage."""

from __future__ import annotations

import typing

import ml_switcheroo_compiler.ops.ragged.core as ragged_core_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


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


def test_ragged_core_coverage() -> None:
    """Verify 100% line and branch coverage for ragged core operations."""
    dot_op = ragged_core_mod.RaggedDot()

    # len(args) >= 2 with shapes
    t1 = DummyWithShape((2, 4, 8))
    t2 = DummyWithShape((8, 16))
    out1 = dot_op.infer_shape(t1, t2)
    assert out1 == (2, 4, 16)

    # len(args) < 2 using kwargs rt_a and b
    out2 = dot_op.infer_shape(rt_a=t1, b=t2)
    assert out2 == (2, 4, 16)

    # a is None -> returns ()
    assert dot_op.infer_shape(None, None) == ()

    # shape_a variations:
    # 1. a is raw list/tuple
    assert typing.cast("EagerTensorProtocol", getattr(dot_op.infer_shape([3, 5], None), "__getitem__")(0)) == 3

    # 2. len(shape_a) <= 1 (ragged_dim uses SymVar fallback)
    t_1d = DummyWithShape((7,))
    out_1d = dot_op.infer_shape(t_1d, None)
    assert typing.cast("EagerTensorProtocol", getattr(out_1d, "__getitem__")(0)) == 7
    assert str(typing.cast("EagerTensorProtocol", getattr(out_1d, "__getitem__")(1))) == "ragged_dim"
    assert typing.cast("EagerTensorProtocol", getattr(out_1d, "__getitem__")(2)) == 1  # len(shape_a) <= 2, len(shape_b) == 0 fallback

    # 3. len(shape_a) == 0 (batch fallback to 1)
    t_0d = DummyWithShape(())
    out_0d = dot_op.infer_shape(t_0d, None)
    assert typing.cast("EagerTensorProtocol", getattr(out_0d, "__getitem__")(0)) == 1

    # 4. len(shape_b) == 0, len(shape_a) > 2 (out_features falls back to shape_a[-1])
    t_3d = DummyWithShape((2, 3, 9))
    out_3d = dot_op.infer_shape(t_3d, None)
    assert typing.cast("EagerTensorProtocol", getattr(out_3d, "__getitem__")(2)) == 9
