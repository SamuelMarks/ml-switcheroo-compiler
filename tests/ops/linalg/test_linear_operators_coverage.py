"""Tests for test_linear_operators_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.linalg.linear_operator as linop_mod


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


def test_linear_operators_coverage() -> None:
    """Test full coverage for base and derived linear operators."""
    base_op = linop_mod.BaseLinearOperator()

    assert base_op.infer_shape(DummyWithShape((3, 3))) == (3, 3)
    assert base_op.infer_shape(DummyWithShape([3, 3])) == (3, 3)  # type: ignore[arg-type]
    assert base_op.infer_shape(operand=DummyWithShape((4, 4))) == (4, 4)
    assert base_op.infer_shape(operator=DummyWithShape((5, 5))) == (5, 5)
    assert base_op.infer_shape(DummyWithoutShape()) == ()
    assert base_op.infer_shape() == ()

    derived_classes: list[type[linop_mod.BaseLinearOperator]] = [
        linop_mod.LinearOperator,
        linop_mod.LinearOperatorAdjoint,
        linop_mod.LinearOperatorBlockDiag,
        linop_mod.LinearOperatorBlockLowerTriangular,
        linop_mod.LinearOperatorCirculant,
        linop_mod.LinearOperatorCirculant2D,
        linop_mod.LinearOperatorCirculant3D,
        linop_mod.LinearOperatorComposition,
        linop_mod.LinearOperatorDiag,
        linop_mod.LinearOperatorFullMatrix,
        linop_mod.LinearOperatorHouseholder,
        linop_mod.LinearOperatorIdentity,
        linop_mod.LinearOperatorInversion,
        linop_mod.LinearOperatorKronecker,
        linop_mod.LinearOperatorLowRankUpdate,
        linop_mod.LinearOperatorLowerTriangular,
        linop_mod.LinearOperatorPermutation,
        linop_mod.LinearOperatorScaledIdentity,
        linop_mod.LinearOperatorToeplitz,
        linop_mod.LinearOperatorTridiag,
        linop_mod.LinearOperatorZeros,
    ]

    for op_cls in derived_classes:
        instance = op_cls()
        assert isinstance(instance, linop_mod.BaseLinearOperator)
        assert instance.infer_shape(DummyWithShape((2, 2))) == (2, 2)
