"""Tests for test_unary_special_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.unary.special as unary_special


class _MockDimWithId:
    """Mock dimension exposing an id attribute."""

    def __init__(self, dim_id: str) -> None:
        """Initialize mock dimension.

        Args:
            dim_id (str): Identifier for dimension.
        """
        self.id = dim_id


class _MockShapeContainer:
    """Mock container exposing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock shape container.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape = shape


class _MockShapeMetaContainer:
    """Mock container exposing a shape_metadata attribute."""

    def __init__(self, shape_metadata: tuple[int, ...]) -> None:
        """Initialize mock shape metadata container.

        Args:
            shape_metadata (tuple[int, ...]): Shape metadata tuple.
        """
        self.shape_metadata = shape_metadata


class _MockGenerator:
    """Mock generator for variable AST visitor."""

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix for AST emissions.

        Returns:
            str: Fallback prefix string.
        """
        return "mock_backend"


def test_unary_special_exhaustive() -> None:
    """Verify shape inference and operators in ops/unary/special.py."""
    # 1. Cast
    cast_op = unary_special.Cast()
    assert cast_op.infer_shape() == ()
    assert cast_op.infer_shape(x="sym_input") == "sym_input"
    assert cast_op.infer_shape("tensor_symbol") == "tensor_symbol"
    assert cast_op.infer_shape((2, 3)) == (2, 3)
    assert cast_op.infer_shape([4, 5]) == (4, 5)
    assert cast_op.infer_shape(_MockShapeContainer((6, 7))) == (6, 7)
    assert cast_op.infer_shape(_MockShapeMetaContainer((8, 9))) == (8, 9)

    # 2. Bitcast
    bitcast_op = unary_special.Bitcast()
    assert isinstance(bitcast_op, unary_special.Cast)
    assert bitcast_op.infer_shape((10,)) == (10,)

    # 3. CanCast
    can_cast_op = unary_special.CanCast()
    assert can_cast_op.infer_shape() == ()
    assert can_cast_op.infer_shape("x", "float32") == ()

    # 4. Frexp
    frexp_op = unary_special.Frexp()
    assert frexp_op.infer_shape() == ()
    assert frexp_op.infer_shape(x="sym_input") == "sym_input"
    assert frexp_op.infer_shape("tensor_symbol") == "tensor_symbol"
    assert frexp_op.infer_shape((3, 4)) == (3, 4)
    assert frexp_op.infer_shape([5, 6]) == (5, 6)
    assert frexp_op.infer_shape(_MockShapeContainer((7, 8))) == (7, 8)
    assert frexp_op.infer_shape(_MockShapeMetaContainer((9, 10))) == (9, 10)

    # 5. Lbeta
    lbeta_op = unary_special.Lbeta()
    assert lbeta_op.infer_shape(()) == ()
    assert lbeta_op.infer_shape((5,)) == ()
    assert lbeta_op.infer_shape((2, 3)) == (2,)
    assert lbeta_op.infer_shape((4, 5, 6)) == (4, 5)
    assert lbeta_op.infer_shape([7, 8]) == (7,)
    assert lbeta_op.infer_shape(_MockShapeContainer((10, 20))) == (10,)
    assert lbeta_op.infer_shape(99) == ()

    # 6. All registered UnaryMathOp subclasses
    unary_math_classes = [
        unary_special.Erf,
        unary_special.BesselI0e,
        unary_special.BesselI1e,
        unary_special.Erfc,
        unary_special.Erfinv,
        unary_special.Lgamma,
        unary_special.Digamma,
        unary_special.Mvlgamma,
        unary_special.SpecialGamma,
        unary_special.BesselI0,
        unary_special.BesselI1,
        unary_special.Erfcinv,
        unary_special.Ndtri,
        unary_special.BesselJ0,
        unary_special.BesselJ1,
        unary_special.BesselK0,
        unary_special.BesselK0e,
        unary_special.BesselK1,
        unary_special.BesselK1e,
        unary_special.BesselY0,
        unary_special.BesselY1,
        unary_special.Dawsn,
        unary_special.Expint,
        unary_special.FresnelCos,
        unary_special.FresnelSin,
        unary_special.Spence,
        unary_special.ModifiedBesselI0,
        unary_special.ModifiedBesselI1,
        unary_special.ModifiedBesselK0,
        unary_special.ModifiedBesselK1,
    ]
    for cls in unary_math_classes:
        op_inst = cls()
        assert op_inst.infer_shape((2, 4)) == (2, 4)
        assert bool(op_inst.op_name)
