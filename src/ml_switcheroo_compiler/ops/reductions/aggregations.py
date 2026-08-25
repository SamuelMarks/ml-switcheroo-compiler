"""Module aggregations.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Reductions."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


@register_op("Sum")
class Sum(ReductionOp):
    """Sum reduction operation.

    Computes the sum of elements across specified dimensions of an input tensor
    """

    op_name: object = "Sum"


@register_op("Max")
class Max(ReductionOp):
    """Max reduction operation.

    Computes the maximum value of elements across specified dimensions of an input
    tensor
    """

    op_name: object = "Max"


@register_op("Min")
class Min(ReductionOp):
    """Min reduction operation.

    Computes the minimum value of elements across specified dimensions of an input
    tensor
    """

    op_name: object = "Min"


@register_op("Prod")
class Prod(ReductionOp):
    """Product reduction operation.

    Computes the product of elements across specified dimensions of an input tensor
    """

    op_name: object = "Prod"
    np_op_name: object = "prod"


@register_op("Argmax")
class Argmax(ReductionOp):
    """Argmax reduction operation.

    Computes the indices of the maximum values across specified dimensions of an
    input tensor
    """

    op_name: object = "Argmax"
    np_op_name: object = "argmax"


@register_op("Argmin")
class Argmin(ReductionOp):
    """Argmin reduction operation.

    Computes the indices of the minimum values across specified dimensions of an
    input tensor
    """

    op_name: object = "Argmin"
    np_op_name: object = "argmin"


@register_op("Logsumexp")
class Logsumexp(ReductionOp):
    """Log-sum-exp reduction operation.

    Computes the logarithm of the sum of exponentials of elements across specified
    dimensions
    """

    op_name: object = "Logsumexp"
    np_op_name: object = "logsumexp"


@register_op("CountNonzero")
class CountNonzero(ReductionOp):
    """Count non-zero elements reduction operation.

    Counts the number of non-zero elements across specified dimensions of an input
    tensor
    """

    op_name: object = "CountNonzero"
    np_op_name: object = "count_nonzero"


@register_op("Norm")
class Norm(ReductionOp):
    """Matrix or vector norm reduction operation.

    Computes the norm of elements across specified dimensions of an input tensor
    """

    op_name: object = "Norm"
    np_op_name: object = "norm"


@register_op("Cumsum")
class Cumsum(ReductionOp):
    """Cumulative sum reduction operation.

    Computes the cumulative sum of elements across specified dimensions of an input
    tensor
    """

    op_name: object = "Cumsum"
    np_op_name: object = "cumsum"


class NaryMathOp(OpDef):
    """Define base class for N-ary mathematical operations (operations taking a list of tensors)."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        inputs: object = args[0] if len(args) > 0 else kwargs.get("inputs")
        # Assume all inputs have the same shape
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            return getattr(inputs[0], "shape", ())
        return ()


@register_op("AddN")
class AddN(NaryMathOp):
    """AddN operation."""

    op_name: object = "AddN"


@register_op("AccumulateN")
class AccumulateN(NaryMathOp):
    """AccumulateN operation."""

    op_name: object = "AccumulateN"


@register_op("CumulativeLogsumexp")
class CumulativeLogsumexp(ReductionOp):
    """Cumulative log-sum-exp reduction operation."""

    op_name: object = "CumulativeLogsumexp"


@register_op("ReduceEuclideanNorm")
class ReduceEuclideanNorm(ReductionOp):
    """ReduceEuclideanNorm operation."""

    op_name: object = "ReduceEuclideanNorm"


@register_op("Cummax")
class Cummax(ReductionOp):
    """Cummax."""

    op_name: object = "Cummax"


@register_op("Cummin")
class Cummin(ReductionOp):
    """Cummin."""

    op_name: object = "Cummin"


@register_op("Cumprod")
class Cumprod(ReductionOp):
    """Cumprod."""

    op_name: object = "Cumprod"
    np_op_name: object = "cumprod"


@register_op("Cumlogsumexp")
class Cumlogsumexp(ReductionOp):
    """Cumlogsumexp."""

    op_name: object = "Cumlogsumexp"


@register_op("Logcumsumexp")
class Logcumsumexp(ReductionOp):
    """Logcumsumexp operation."""

    op_name: object = "Logcumsumexp"
    np_op_name: object = "logcumsumexp"  # fake numpy op, backend should handle
