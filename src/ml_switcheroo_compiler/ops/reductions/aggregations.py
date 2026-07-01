"""Reductions."""

from __future__ import annotations


from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.configs import WindowConfig


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


class SegmentOp(OpDef):
    """Base class for Segment Operations."""

    def infer_shape(
        self,
        data: object,
        segment_ids: object,
        num_segments: object = None,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        # Simple heuristic: replace first dimension with num_segments
        return ()

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax."""
        return f"Not implemented {self.__class__.__name__}"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras."""
        return f"Not implemented {self.__class__.__name__}"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx."""
        return f"Not implemented {self.__class__.__name__}"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch."""
        return f"Not implemented {self.__class__.__name__}"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow."""
        return f"Not implemented {self.__class__.__name__}"


@register_op("SegmentSum")
class SegmentSum(SegmentOp):
    """SegmentSum operation."""

    op_name = "SegmentSum"


@register_op("SegmentMax")
class SegmentMax(SegmentOp):
    """SegmentMax operation."""

    op_name = "SegmentMax"


@register_op("SegmentMean")
class SegmentMean(SegmentOp):
    """SegmentMean operation."""

    op_name = "SegmentMean"


@register_op("SegmentMin")
class SegmentMin(SegmentOp):
    """SegmentMin operation."""

    op_name = "SegmentMin"


@register_op("SegmentProd")
class SegmentProd(SegmentOp):
    """SegmentProd operation."""

    op_name = "SegmentProd"


@register_op("UnsortedSegmentMax")
class UnsortedSegmentMax(SegmentOp):
    """UnsortedSegmentMax operation."""

    op_name = "UnsortedSegmentMax"


@register_op("UnsortedSegmentMean")
class UnsortedSegmentMean(SegmentOp):
    """UnsortedSegmentMean operation."""

    op_name = "UnsortedSegmentMean"


@register_op("UnsortedSegmentMin")
class UnsortedSegmentMin(SegmentOp):
    """UnsortedSegmentMin operation."""

    op_name = "UnsortedSegmentMin"


@register_op("UnsortedSegmentProd")
class UnsortedSegmentProd(SegmentOp):
    """UnsortedSegmentProd operation."""

    op_name = "UnsortedSegmentProd"


@register_op("UnsortedSegmentSqrtN")
class UnsortedSegmentSqrtN(SegmentOp):
    """UnsortedSegmentSqrtN operation."""

    op_name = "UnsortedSegmentSqrtN"


@register_op("UnsortedSegmentSum")
class UnsortedSegmentSum(SegmentOp):
    """UnsortedSegmentSum operation."""

    op_name = "UnsortedSegmentSum"


class ApproxKOp(OpDef):
    """Base class for ApproxK Operations."""

    def infer_shape(
        self,
        operand: object,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        return ()

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax."""
        return f"Not implemented {self.__class__.__name__}"  # pragma: no cover

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras."""
        return f"Not implemented {self.__class__.__name__}"  # pragma: no cover

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx."""
        return f"Not implemented {self.__class__.__name__}"  # pragma: no cover

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch."""
        return f"Not implemented {self.__class__.__name__}"  # pragma: no cover

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow."""
        return f"Not implemented {self.__class__.__name__}"  # pragma: no cover


@register_op("ApproxMaxK")
class ApproxMaxK(ApproxKOp):
    """ApproxMaxK operation."""

    op_name = "ApproxMaxK"


@register_op("ApproxMinK")
class ApproxMinK(ApproxKOp):
    """ApproxMinK operation."""

    op_name = "ApproxMinK"


@register_op("ApproxMaxKIndices")
class ApproxMaxKIndices(ApproxKOp):
    """ApproxMaxKIndices operation."""

    op_name = "ApproxMaxKIndices"


@register_op("ApproxMinKIndices")
class ApproxMinKIndices(ApproxKOp):
    """ApproxMinKIndices operation."""

    op_name = "ApproxMinKIndices"


@register_op("ReduceWindow")
class ReduceWindow(ReductionOp):
    """ReduceWindow operation.

    Applies a reduction function over a sliding window of the input.
    """

    op_name = "ReduceWindow"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): operand, init_value, computation, config.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        operand, config = self._extract_reduce_window_args(args, kwargs)

        if not hasattr(operand, "shape") or not operand.shape:
            return ()

        in_shape = operand.shape
        window_dimensions, window_strides, padding, base_dilation, window_dilation = (
            self._normalize_config(config)
        )

        out_shape = []
        for i, dim in enumerate(in_shape):
            if i >= len(window_dimensions):
                out_shape.append(dim)
                continue

            out_dim = self._compute_reduce_window_dim(dim, i, config)
            out_shape.append(out_dim)

        return tuple(out_shape)

    def _extract_reduce_window_args(self, args: tuple, kwargs: dict) -> tuple:
        """Execute _extract_reduce_window_args.

        Args:
            args (Any): Argument args.
            kwargs (Any): Argument kwargs.

        Returns:
        Any: The result.
        """
        operand = args[0] if len(args) > 0 else kwargs["operand"]
        config = args[3] if len(args) > MAGIC_VAL_3 else kwargs.get("config", None)
        if config is None:
            config = WindowConfig(window_dimensions=[])
        return operand, config

    def _normalize_config(self, config: WindowConfig) -> tuple:
        """Execute _normalize_config.

        Args:
            config (Any): Argument config.

        Returns:
        Any: The result.
        """
        window_dimensions = config.window_dimensions
        n = len(window_dimensions)
        window_strides = config.window_strides if config.window_strides is not None else [1] * n
        padding = config.padding if config.padding is not None else [(0, 0)] * n
        base_dilation = config.base_dilation if config.base_dilation is not None else [1] * n
        window_dilation = config.window_dilation if config.window_dilation is not None else [1] * n
        return window_dimensions, window_strides, padding, base_dilation, window_dilation

    @staticmethod
    def _get_axis_param(param_list: list[int], axis: int, default: int = 1) -> int:
        """Function docstring.

        Args:
        param_list: Arg.
        axis: Arg.
        default: Arg.
        """
        return param_list[axis] if axis < len(param_list) else default

    @staticmethod
    def _get_axis_pad(padding: list[tuple[int, int]], axis: int) -> tuple[int, int]:
        """Function docstring.

        Args:
        padding: Arg.
        axis: Arg.
        """
        pad = (
            padding[axis] if isinstance(padding, (list, tuple)) and axis < len(padding) else (0, 0)
        )
        return pad if isinstance(pad, tuple) else (0, 0)

    def _extract_window_params(
        self, axis: int, config: WindowConfig
    ) -> tuple[int, int, int, int, int, int]:
        """Function docstring.

        Args:
        axis: Arg.
        config: Arg.
        """
        window_dimensions, window_strides, padding, base_dilation, window_dilation = (
            self._normalize_config(config)
        )
        pad_low, pad_high = self._get_axis_pad(padding, axis)  # type: ignore
        base_dil = self._get_axis_param(base_dilation, axis, 1)
        win_dil = self._get_axis_param(window_dilation, axis, 1)
        stride = self._get_axis_param(window_strides, axis, 1)
        win_dim = self._get_axis_param(window_dimensions, axis, 1)

        return pad_low, pad_high, base_dil, win_dil, stride, win_dim

    def _compute_reduce_window_dim(self, dim: int, axis: int, config: WindowConfig) -> int:
        """Execute _compute_reduce_window_dim.

        Args:
            dim (int): Argument dim.
            axis (int): Argument axis.
            config (WindowConfig): Argument config.

        Returns:
        Any: The result.
        """
        pad_low, pad_high, base_dil, win_dil, stride, win_dim = self._extract_window_params(
            axis, config
        )
        eff_in_dim = (dim - 1) * base_dil + 1 + pad_low + pad_high
        eff_win_dim = (win_dim - 1) * win_dil + 1
        return 0 if eff_in_dim < eff_win_dim else (eff_in_dim - eff_win_dim) // stride + 1

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented ReduceWindow"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented ReduceWindow"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented ReduceWindow"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented ReduceWindow"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented ReduceWindow"


@register_op("Bincount")
class Bincount(ReductionOp):
    """Count number of occurrences of each value in array of non-negative ints."""

    op_name = "Bincount"
    np_op_name = "bincount"


class NaryMathOp(OpDef):
    """Base class for N-ary mathematical operations (operations taking a list of tensors)."""

    def infer_shape(self, inputs: object, **kwargs: object) -> object:
        """Infer shape."""
        # Assume all inputs have the same shape
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            return getattr(inputs[0], "shape", ())
        return ()  # pragma: no cover


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


@register_op("Cumlogsumexp")
class Cumlogsumexp(ReductionOp):
    """Cumlogsumexp."""

    op_name = "Cumlogsumexp"


@register_op("Logcumsumexp")
class Logcumsumexp(ReductionOp):
    """Logcumsumexp operation."""

    op_name = "Logcumsumexp"
    np_op_name = "logcumsumexp"  # fake numpy op, backend should handle
