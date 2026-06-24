"""Defines tensor creation operations for the ML Switcheroo framework.

This module contains operations that generate new tensors, such as zeros, ones, full,
and arange, along with their shape inference and NumPy evaluation implementations
"""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class CreationOp(OpDef):
    """Base class for tensor creation operations.

    Provides common implementations for shape inference and NumPy evaluation
    for operations that create tensors of a specified shape (e.g., Zeros, Ones)
    """

    op_name: str = ""

    def infer_shape(self, shape: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            shape (object): The shape of the tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return shape


@register_op("Zeros")
class Zeros(CreationOp):
    """An operation that creates a tensor of a specified shape filled with zeros."""

    op_name = "Zeros"


@register_op("Ones")
class Ones(CreationOp):
    """An operation that creates a tensor of a specified shape filled with ones."""

    op_name = "Ones"


@register_op("Full")
class Full(CreationOp):
    """An operation that creates a tensor of a specified shape filled with a constant.

    value
    """

    op_name = "Full"

    def infer_shape(
        self,
        shape: object,
        fill_value: object,
        **kwargs: object,
    ) -> object:
        """Infer the output shape of the operation.

        Args:
            shape (object): The shape of the tensor.
            fill_value (object): The fill_value to process.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return shape


@register_op("Arange")
class Arange(OpDef):
    """An operation that creates a 1-D tensor containing a sequence of evenly spaced.

    values

    within a given interval
    """

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return None  # Dynamic shape depending on values


@register_op("Rand")
class Rand(CreationOp):
    """Creates a tensor with random numbers from a uniform distribution [0, 1)."""

    op_name = "Rand"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if "size" in kwargs:
            return kwargs["size"]
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return args[0]
        return args


@register_op("Randn")
class Randn(Rand):
    """Creates a tensor with random numbers from a standard normal distribution."""

    op_name = "Randn"


@register_op("Randint")
class Randint(CreationOp):
    """Creates a tensor with random integers from [low, high)."""

    op_name = "Randint"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if "size" in kwargs:
            return kwargs["size"]
        if len(args) == MAGIC_VAL_3:
            return args[2]
        return ()


@register_op("ManualSeed")
class ManualSeed(OpDef):
    """Sets the seed for generating random numbers."""

    op_name = "ManualSeed"

    def infer_shape(self, seed: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            seed (object): The seed to process.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return ()


@register_op("ConstantOfShape")
class ConstantOfShape(OpDef):
    """ConstantOfShape operator."""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
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

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
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

    def infer_shape(self, M: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Bartlett")
class Bartlett(OpDef):
    """Bartlett window."""

    def infer_shape(self, M: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Hamming")
class Hamming(OpDef):
    """Hamming window."""

    def infer_shape(self, M: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Hanning")
class Hanning(OpDef):
    """Hanning window."""

    def infer_shape(self, M: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Kaiser")
class Kaiser(OpDef):
    """Kaiser window."""

    def infer_shape(self, M: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (M,) if isinstance(M, int) else (M.item(),)
