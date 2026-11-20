"""Behavioral unit tests for linear operator."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.linalg.linear_operator as linop_mod
from tests.testing_utils import MockTensorStub


def test_linear_operator_shape_inference() -> None:
    """Test function test_linear_operator_shape_inference."""
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperator

    assert LinearOperator().infer_shape() == ()

    class MockOperand(MockTensorStub):
        """Test class MockOperand."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.shape = (1, 2)

    assert LinearOperator().infer_shape(MockOperand()) == (1, 2)
    assert LinearOperator().infer_shape(operand=MockOperand()) == (1, 2)


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize dummy object with shape.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape(MockTensorStub):
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize dummy object without shape."""
        self.val: int = 42


def test_linear_operators_coverage() -> None:
    """Test full coverage for base and derived linear operators."""
    base_op = linop_mod.BaseLinearOperator()

    assert base_op.infer_shape(DummyWithShape((3, 3))) == (3, 3)
    assert base_op.infer_shape(DummyWithShape((2, 3))) == (2, 3)
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
