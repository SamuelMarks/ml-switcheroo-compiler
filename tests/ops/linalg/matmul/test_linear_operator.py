"""Test linear operators."""

from ml_switcheroo_compiler.ops.linalg.linear_operator import (
    LinearOperator,
    LinearOperatorAdjoint,
    LinearOperatorBlockDiag,
    LinearOperatorBlockLowerTriangular,
    LinearOperatorCirculant,
    LinearOperatorCirculant2D,
    LinearOperatorCirculant3D,
    LinearOperatorComposition,
    LinearOperatorDiag,
    LinearOperatorFullMatrix,
    LinearOperatorHouseholder,
    LinearOperatorIdentity,
    LinearOperatorInversion,
    LinearOperatorKronecker,
    LinearOperatorLowerTriangular,
    LinearOperatorLowRankUpdate,
    LinearOperatorPermutation,
    LinearOperatorScaledIdentity,
    LinearOperatorToeplitz,
    LinearOperatorTridiag,
    LinearOperatorZeros,
)


def test_linear_operators() -> None:
    """Test linear operators."""
    op = LinearOperator()
    LinearOperatorAdjoint(op)
    LinearOperatorBlockDiag([op])
    LinearOperatorBlockLowerTriangular([[op]])
    LinearOperatorCirculant(1)
    LinearOperatorCirculant2D(1)
    LinearOperatorCirculant3D(1)
    LinearOperatorComposition([op])
    LinearOperatorDiag(1)
    LinearOperatorFullMatrix(1)
    LinearOperatorHouseholder(1)
    LinearOperatorIdentity(1)
    LinearOperatorInversion(op)
    LinearOperatorKronecker([op])
    LinearOperatorLowRankUpdate(op, 1)
    LinearOperatorLowerTriangular(1)
    LinearOperatorPermutation(1)
    LinearOperatorScaledIdentity(1, 1)
    LinearOperatorToeplitz(1, 1)
    LinearOperatorTridiag(1)
    LinearOperatorZeros(1, 1)
