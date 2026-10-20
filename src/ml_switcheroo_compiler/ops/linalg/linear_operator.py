"""Implementation of linear operator operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.linear_operator`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.base import OpDef, register_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


class HasShape(typing.Protocol):
    """Protocol for objects providing a shape attribute."""

    shape: tuple[int, ...]


ShapeOperand = typing.Union["Tensor", HasShape, int, float, str, None]


class BaseLinearOperator(OpDef):
    """Base for linear operators."""

    def infer_shape(
        self,
        *args: ShapeOperand,
        **kwargs: ShapeOperand,
    ) -> tuple[int, ...]:
        """Infer output shape for linear operator.

        Args:
            *args (ShapeOperand): Positional arguments or operands.
            **kwargs (ShapeOperand): Keyword arguments including operand or operator.

        Returns:
            tuple[int, ...]: Inferred output shape.
        """
        # Default shape inference attempts to find a shape or operand
        operand = args[0] if len(args) > 0 else kwargs.get("operand", kwargs.get("operator"))
        if hasattr(operand, "shape"):
            shape_val = getattr(operand, "shape", ())
            if isinstance(shape_val, tuple):
                return shape_val
            return tuple(shape_val)
        return ()


@register_op("LinearOperator")
class LinearOperator(BaseLinearOperator):
    """LinearOperator class."""


@register_op("LinearOperatorAdjoint")
class LinearOperatorAdjoint(BaseLinearOperator):
    """LinearOperatorAdjoint class."""


@register_op("LinearOperatorBlockDiag")
class LinearOperatorBlockDiag(BaseLinearOperator):
    """LinearOperatorBlockDiag class."""


@register_op("LinearOperatorBlockLowerTriangular")
class LinearOperatorBlockLowerTriangular(BaseLinearOperator):
    """LinearOperatorBlockLowerTriangular class."""


@register_op("LinearOperatorCirculant")
class LinearOperatorCirculant(BaseLinearOperator):
    """LinearOperatorCirculant class."""


@register_op("LinearOperatorCirculant2D")
class LinearOperatorCirculant2D(BaseLinearOperator):
    """LinearOperatorCirculant2D class."""


@register_op("LinearOperatorCirculant3D")
class LinearOperatorCirculant3D(BaseLinearOperator):
    """LinearOperatorCirculant3D class."""


@register_op("LinearOperatorComposition")
class LinearOperatorComposition(BaseLinearOperator):
    """LinearOperatorComposition class."""


@register_op("LinearOperatorDiag")
class LinearOperatorDiag(BaseLinearOperator):
    """LinearOperatorDiag class."""


@register_op("LinearOperatorFullMatrix")
class LinearOperatorFullMatrix(BaseLinearOperator):
    """LinearOperatorFullMatrix class."""


@register_op("LinearOperatorHouseholder")
class LinearOperatorHouseholder(BaseLinearOperator):
    """LinearOperatorHouseholder class."""


@register_op("LinearOperatorIdentity")
class LinearOperatorIdentity(BaseLinearOperator):
    """LinearOperatorIdentity class."""


@register_op("LinearOperatorInversion")
class LinearOperatorInversion(BaseLinearOperator):
    """LinearOperatorInversion class."""


@register_op("LinearOperatorKronecker")
class LinearOperatorKronecker(BaseLinearOperator):
    """LinearOperatorKronecker class."""


@register_op("LinearOperatorLowRankUpdate")
class LinearOperatorLowRankUpdate(BaseLinearOperator):
    """LinearOperatorLowRankUpdate class."""


@register_op("LinearOperatorLowerTriangular")
class LinearOperatorLowerTriangular(BaseLinearOperator):
    """LinearOperatorLowerTriangular class."""


@register_op("LinearOperatorPermutation")
class LinearOperatorPermutation(BaseLinearOperator):
    """LinearOperatorPermutation class."""


@register_op("LinearOperatorScaledIdentity")
class LinearOperatorScaledIdentity(BaseLinearOperator):
    """LinearOperatorScaledIdentity class."""


@register_op("LinearOperatorToeplitz")
class LinearOperatorToeplitz(BaseLinearOperator):
    """LinearOperatorToeplitz class."""


@register_op("LinearOperatorTridiag")
class LinearOperatorTridiag(BaseLinearOperator):
    """LinearOperatorTridiag class."""


@register_op("LinearOperatorZeros")
class LinearOperatorZeros(BaseLinearOperator):
    """LinearOperatorZeros class."""
