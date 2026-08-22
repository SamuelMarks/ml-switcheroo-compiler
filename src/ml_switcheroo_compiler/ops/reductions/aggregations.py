"""Module aggregations.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Reductions."""
from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Sum")
class Sum(ReductionOp):
    """Sum reduction operation.

    Computes the sum of elements across specified dimensions of an input tensor
    """

    op_name = "Sum"


@register_op("Max")
class Max(ReductionOp):
    """Max reduction operation.

    Computes the maximum value of elements across specified dimensions of an input
    tensor
    """

    op_name = "Max"


@register_op("Min")
class Min(ReductionOp):
    """Min reduction operation.

    Computes the minimum value of elements across specified dimensions of an input
    tensor
    """

    op_name = "Min"


@register_op("Prod")
class Prod(ReductionOp):
    """Product reduction operation.

    Computes the product of elements across specified dimensions of an input tensor
    """

    op_name = "Prod"
    np_op_name = "prod"


@register_op("Argmax")
class Argmax(ReductionOp):
    """Argmax reduction operation.

    Computes the indices of the maximum values across specified dimensions of an
    input tensor
    """

    op_name = "Argmax"
    np_op_name = "argmax"


@register_op("Argmin")
class Argmin(ReductionOp):
    """Argmin reduction operation.

    Computes the indices of the minimum values across specified dimensions of an
    input tensor
    """

    op_name = "Argmin"
    np_op_name = "argmin"


@register_op("Logsumexp")
class Logsumexp(ReductionOp):
    """Log-sum-exp reduction operation.

    Computes the logarithm of the sum of exponentials of elements across specified
    dimensions
    """

    op_name = "Logsumexp"
    np_op_name = "logsumexp"


@register_op("CountNonzero")
class CountNonzero(ReductionOp):
    """Count non-zero elements reduction operation.

    Counts the number of non-zero elements across specified dimensions of an input
    tensor
    """

    op_name = "CountNonzero"
    np_op_name = "count_nonzero"


@register_op("Norm")
class Norm(ReductionOp):
    """Matrix or vector norm reduction operation.

    Computes the norm of elements across specified dimensions of an input tensor
    """

    op_name = "Norm"
    np_op_name = "norm"


@register_op("Cumsum")
class Cumsum(ReductionOp):
    """Cumulative sum reduction operation.

    Computes the cumulative sum of elements across specified dimensions of an input
    tensor
    """

    op_name = "Cumsum"
    np_op_name = "cumsum"


class NaryMathOp(OpDef):
    """Define base class for N-ary mathematical operations (operations taking a list of tensors)."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        inputs = args[0] if len(args) > 0 else kwargs.get("inputs")
        # Assume all inputs have the same shape
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            return getattr(inputs[0], "shape", ())
        return ()


@register_op("AddN")
class AddN(NaryMathOp):
    """AddN operation."""

    op_name = "AddN"


@register_op("AccumulateN")
class AccumulateN(NaryMathOp):
    """AccumulateN operation."""

    op_name = "AccumulateN"


@register_op("CumulativeLogsumexp")
class CumulativeLogsumexp(ReductionOp):
    """Cumulative log-sum-exp reduction operation."""

    op_name = "CumulativeLogsumexp"


@register_op("ReduceEuclideanNorm")
class ReduceEuclideanNorm(ReductionOp):
    """ReduceEuclideanNorm operation."""

    op_name = "ReduceEuclideanNorm"


@register_op("Cummax")
class Cummax(ReductionOp):
    """Cummax."""

    op_name = "Cummax"


@register_op("Cummin")
class Cummin(ReductionOp):
    """Cummin."""

    op_name = "Cummin"


@register_op("Cumprod")
class Cumprod(ReductionOp):
    """Cumprod."""

    op_name = "Cumprod"
    np_op_name = "cumprod"


@register_op("Cumlogsumexp")
class Cumlogsumexp(ReductionOp):
    """Cumlogsumexp."""

    op_name = "Cumlogsumexp"


@register_op("Logcumsumexp")
class Logcumsumexp(ReductionOp):
    """Logcumsumexp operation."""

    op_name = "Logcumsumexp"
    np_op_name = "logcumsumexp"  # fake numpy op, backend should handle
