"""Define tensor creation operations for the ML Switcheroo framework.

This module contains operations that generate new tensors, such as zeros, ones, full,
and arange, along with their shape inference and NumPy evaluation implementations
"""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core.constants import MIN_CONV_RANK
from ml_switcheroo_compiler.ops.base import OpArgValue, OpDef, register_op


def _coerce_shape(val: OpArgValue) -> tuple[int, ...] | None:
    """Coerce a raw shape argument into a tuple of integers or None.

    Args:
        val: Raw shape specification (tuple, list, int, or None).

    Returns:
        tuple[int, ...] | None: Coerced shape tuple or None.
    """
    return (val,) if isinstance(val, int) else (tuple(int(x) for x in val if isinstance(x, int)) if isinstance(val, (tuple, list)) else None)


class CreationOp(OpDef):
    """Define base class for tensor creation operations.

    Provides common implementations for shape inference and NumPy evaluation
    for operations that create tensors of a specified shape (e.g., Zeros, Ones).
    """

    op_name: str = ""

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer the output shape of the tensor creation operation.

        Args:
            *args: Positional arguments containing the target shape.
            **kwargs: Keyword arguments containing optional shape specification.

        Returns:
            tuple[int, ...] | None: The inferred output tensor shape.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        return _coerce_shape(shape)


@register_op("Zeros")
class Zeros(CreationOp):
    """Provide an operation that creates a tensor of a specified shape filled with zeros."""

    op_name = "Zeros"


@register_op("Ones")
class Ones(CreationOp):
    """Provide an operation that creates a tensor of a specified shape filled with ones."""

    op_name = "Ones"


@register_op("Full")
class Full(CreationOp):
    """Provide an operation that creates a tensor of a specified shape filled with a constant value."""

    op_name = "Full"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer the output shape of the Full operation.

        Args:
            *args: Positional arguments containing shape and fill_value.
            **kwargs: Keyword arguments containing optional shape specification.

        Returns:
            tuple[int, ...] | None: The inferred output tensor shape.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        return _coerce_shape(shape)


@register_op("Arange")
class Arange(OpDef):
    """Provide an operation that creates a 1-D tensor containing evenly spaced values."""

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer the output shape of the Arange operation.

        Args:
            *args: Positional arguments (start, stop, step).
            **kwargs: Keyword arguments.

        Returns:
            tuple[int, ...] | None: None because Arange output length depends on values.
        """
        del args, kwargs
        return None


@register_op("Rand")
class Rand(CreationOp):
    """Create a tensor with random numbers from a uniform distribution [0, 1)."""

    op_name = "Rand"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer the output shape of the Rand operation.

        Args:
            *args: Dimensions or shape tuple.
            **kwargs: Keyword arguments containing optional size.

        Returns:
            tuple[int, ...]: The inferred output shape.
        """
        if "size" in kwargs:
            return _coerce_shape(kwargs["size"]) or ()
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return _coerce_shape(args[0]) or ()
        return _coerce_shape(args) or ()


@register_op("Randn")
class Randn(Rand):
    """Create a tensor with random numbers from a standard normal distribution."""

    op_name = "Randn"


@register_op("Randint")
class Randint(CreationOp):
    """Create a tensor with random integers from [low, high)."""

    op_name = "Randint"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer the output shape of the Randint operation.

        Args:
            *args: Positional arguments (low, high, size).
            **kwargs: Keyword arguments containing optional size.

        Returns:
            tuple[int, ...]: The inferred output shape.
        """
        if "size" in kwargs:
            return _coerce_shape(kwargs["size"]) or ()
        if len(args) == MIN_CONV_RANK:
            return _coerce_shape(args[2]) or ()
        return ()


@register_op("ManualSeed")
class ManualSeed(OpDef):
    """Set the seed for generating random numbers."""

    op_name = "ManualSeed"

    def infer_shape(self, seed: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the output shape of the operation.

        Args:
            seed (Any): The seed to process.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return ()


@register_op("ConstantOfShape")
class ConstantOfShape(OpDef):
    """ConstantOfShape operator."""

    def infer_shape(self, *shapes: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *shapes: Input shapes
            **kwargs: Extra kwargs

        Returns:
            The output shape
        """
        return shapes[0] if shapes else ()


@register_op("Range")
class Range(OpDef):
    """Range operator."""

    def infer_shape(self, *shapes: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *shapes: Input shapes
            **kwargs: Extra kwargs

        Returns:
            The output shape
        """
        # This is a bit tricky, but in eager mode we just rely on eager_eval.
        return shapes[0] if shapes else ()


@register_op("Blackman")
class Blackman(OpDef):
    """Blackman window."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the output shape.

        Args:
            M (Any): The M parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Bartlett")
class Bartlett(OpDef):
    """Bartlett window."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the output shape.

        Args:
            M (Any): The M parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Hamming")
class Hamming(OpDef):
    """Hamming window."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the output shape.

        Args:
            M (Any): The M parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Hanning")
class Hanning(OpDef):
    """Hanning window."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the output shape.

        Args:
            M (Any): The M parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Kaiser")
class Kaiser(OpDef):
    """Kaiser window."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer the output shape.

        Args:
            M (Any): The M parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("TrilIndices")
class TrilIndices(OpDef):
    """TrilIndices operator definition."""

    op_name = "TrilIndices"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for TrilIndices.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            tuple[int, ...]: Computed shape.
        """
        del kwargs
        return (_coerce_shape(args[0]) or ()) if args else ()


@register_op("TrilIndicesFrom")
class TrilIndicesFrom(OpDef):
    """TrilIndicesFrom operator definition."""

    op_name = "TrilIndicesFrom"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for TrilIndicesFrom.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            tuple[int, ...]: Computed shape.
        """
        del kwargs
        return (_coerce_shape(args[0]) or ()) if args else ()


@register_op("TriuIndices")
class TriuIndices(OpDef):
    """TriuIndices operator definition."""

    op_name = "TriuIndices"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for TriuIndices.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            tuple[int, ...]: Computed shape.
        """
        del kwargs
        return (_coerce_shape(args[0]) or ()) if args else ()


@register_op("TriuIndicesFrom")
class TriuIndicesFrom(OpDef):
    """TriuIndicesFrom operator definition."""

    op_name = "TriuIndicesFrom"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for TriuIndicesFrom.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            tuple[int, ...]: Computed shape.
        """
        del kwargs
        return (_coerce_shape(args[0]) or ()) if args else ()


@register_op("Logspace")
class Logspace(CreationOp):
    """Logspace op."""

    op_name = "Logspace"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for Logspace.

        Args:
            *args: Positional arguments (start, stop, num).
            **kwargs: Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred 1-D output shape.
        """
        num_val = kwargs.get("num", args[2] if len(args) > 2 and isinstance(args[2], int) else 50)
        return (int(num_val) if isinstance(num_val, int) else 50,)


@register_op("Frombuffer")
class Frombuffer(OpDef):
    """Operator Frombuffer."""

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer shape for Frombuffer.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments containing optional count.

        Returns:
            tuple[int, ...] | None: Inferred 1-D shape if count is specified, else None.
        """
        del args
        count_val = kwargs.get("count", -1)
        if isinstance(count_val, int) and count_val != -1:
            return (count_val,)
        return None
