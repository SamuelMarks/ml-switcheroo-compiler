from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for linear_operator.py."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class BaseLinearOperator(OpDef):
    """Base for linear operators."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        # Default shape inference attempts to find a shape or operand
        operand = args[0] if len(args) > 0 else kwargs.get("operand", kwargs.get("operator"))
        if hasattr(operand, "shape"):
            return operand.shape
        return ()


@register_op("LinearOperator")
class LinearOperator(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorAdjoint")
class LinearOperatorAdjoint(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorBlockDiag")
class LinearOperatorBlockDiag(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorBlockLowerTriangular")
class LinearOperatorBlockLowerTriangular(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorCirculant")
class LinearOperatorCirculant(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorCirculant2D")
class LinearOperatorCirculant2D(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorCirculant3D")
class LinearOperatorCirculant3D(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorComposition")
class LinearOperatorComposition(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorDiag")
class LinearOperatorDiag(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorFullMatrix")
class LinearOperatorFullMatrix(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorHouseholder")
class LinearOperatorHouseholder(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorIdentity")
class LinearOperatorIdentity(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorInversion")
class LinearOperatorInversion(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorKronecker")
class LinearOperatorKronecker(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorLowRankUpdate")
class LinearOperatorLowRankUpdate(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorLowerTriangular")
class LinearOperatorLowerTriangular(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorPermutation")
class LinearOperatorPermutation(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorScaledIdentity")
class LinearOperatorScaledIdentity(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorToeplitz")
class LinearOperatorToeplitz(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorTridiag")
class LinearOperatorTridiag(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass


@register_op("LinearOperatorZeros")
class LinearOperatorZeros(BaseLinearOperator):
    def __init__(self, *args, **kwargs):
        pass
