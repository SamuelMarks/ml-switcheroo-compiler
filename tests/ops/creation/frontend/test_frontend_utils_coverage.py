"""Tests for test_frontend_utils_coverage."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.ops.creation.frontend_utils import (
    Frompyfunc,
    Geometric,
    Geomspace,
)


def _side_effect_type_error(obj: Any, dtype: Any = None) -> list[Any]:
    """Simulate backend array error when dtype is provided.

    Args:
        obj (Any): Input object.
        dtype (Any): Desired dtype.

    Returns:
        list[Any]: Nested list wrapping obj.
    """
    if dtype is not None:
        raise TypeError("unsupported dtype")
    return [obj]


def _side_effect_value_error(obj: Any, dtype: Any = None) -> list[Any]:
    """Simulate backend array error when dtype is missing.

    Args:
        obj (Any): Input object.
        dtype (Any): Desired dtype.

    Returns:
        list[Any]: List with obj and dtype.
    """
    if dtype is None:
        raise ValueError("needs dtype")
    return [obj, dtype]


class _DummyInput:
    """Dummy tensor object holding a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy input.

        Args:
            shape (tuple[int, ...]): Tensor shape.
        """
        self.shape = shape


def test_frontend_utils_frompyfunc_infer_shape() -> None:
    """Cover Frompyfunc.infer_shape with broadcast and empty inputs."""
    fpf = Frompyfunc()
    assert fpf.infer_shape() == ()

    in1 = _DummyInput((2, 1))
    in2 = _DummyInput((1, 3))
    # Frompyfunc with args[3:] slicing or args <= 3
    res1 = fpf.infer_shape("a", "b", "c", in1, in2)
    assert res1 == (2, 3)

    res2 = fpf.infer_shape(in1, in2)
    assert res2 == (2, 3)


def test_frontend_utils_geomspace_infer_shape() -> None:
    """Cover Geomspace.infer_shape with negative axis and valid shapes."""
    gs = Geomspace()

    start = _DummyInput((4,))
    stop = _DummyInput((4,))

    # axis < 0 branch
    res_neg_axis = gs.infer_shape(start, stop, num=10, axis=-1)
    assert res_neg_axis == (4, 10)

    # axis >= 0 branch
    res_pos_axis = gs.infer_shape(start, stop, num=10, axis=0)
    assert res_pos_axis == (10, 4)

    # scalar start/stop
    res_scalar = gs.infer_shape(1.0, 10.0, num=5)
    assert res_scalar == (5,)

    # invalid num / axis exception handling
    res_invalid = gs.infer_shape(start, stop, num="invalid", axis="invalid")
    assert res_invalid == (50, 4)


def test_frontend_utils_geometric_infer_shape() -> None:
    """Cover Geometric.infer_shape with size tuple/list/int."""
    geo = Geometric()

    class DummyP:
        """Dummy parameter tensor with shape."""

        shape = (3, 4)

    # size as tuple
    assert geo.infer_shape(DummyP(), size=(2, 5)) == (2, 5)
    # size as int
    assert geo.infer_shape(DummyP(), size=7) == (7,)
    # size None, defaults to p.shape
    assert geo.infer_shape(DummyP()) == (3, 4)
