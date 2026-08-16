# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define tensor creation operations for the ML Switcheroo framework.

This module contains operations that generate new tensors, such as zeros, ones, full,
and arange, along with their shape inference and NumPy evaluation implementations
"""

from typing import Any

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3
from ml_switcheroo_compiler.ops.base import OpDef, register_op


class CreationOp(OpDef):
    """Define base class for tensor creation operations.

    Provides common implementations for shape inference and NumPy evaluation
    for operations that create tensors of a specified shape (e.g., Zeros, Ones)
    """

    op_name: str = ""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        """Infer shape.

        Args:
            shape (object): The shape parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return shape


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
    """Provide an operation that creates a tensor of a specified shape filled with a constant.

    value
    """

    op_name = "Full"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
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
    """Provide an operation that creates a 1-D tensor containing a sequence of evenly spaced.

    values

    within a given interval
    """

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
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
    """Create a tensor with random numbers from a uniform distribution [0, 1)."""

    op_name = "Rand"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
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
    """Create a tensor with random numbers from a standard normal distribution."""

    op_name = "Randn"


@register_op("Randint")
class Randint(CreationOp):
    """Create a tensor with random integers from [low, high)."""

    op_name = "Randint"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
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
    """Set the seed for generating random numbers."""

    op_name = "ManualSeed"

    def infer_shape(self, seed: Any, **kwargs: Any) -> Any:
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

    def infer_shape(self, *shapes: Any, **kwargs: Any) -> Any:
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

    def infer_shape(self, *shapes: Any, **kwargs: Any) -> Any:
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

    def infer_shape(self, M: Any, **kwargs: Any) -> Any:
        """Infer the output shape.

        Args:
            M (object): The M parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Bartlett")
class Bartlett(OpDef):
    """Bartlett window."""

    def infer_shape(self, M: Any, **kwargs: Any) -> Any:
        """Infer the output shape.

        Args:
            M (object): The M parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Hamming")
class Hamming(OpDef):
    """Hamming window."""

    def infer_shape(self, M: Any, **kwargs: Any) -> Any:
        """Infer the output shape.

        Args:
            M (object): The M parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Hanning")
class Hanning(OpDef):
    """Hanning window."""

    def infer_shape(self, M: Any, **kwargs: Any) -> Any:
        """Infer the output shape.

        Args:
            M (object): The M parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Kaiser")
class Kaiser(OpDef):
    """Kaiser window."""

    def infer_shape(self, M: Any, **kwargs: Any) -> Any:
        """Infer the output shape.

        Args:
            M (object): The M parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("TrilIndices")
class TrilIndices(OpDef):
    """TrilIndices operator definition."""

    op_name = "TrilIndices"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("TrilIndicesFrom")
class TrilIndicesFrom(OpDef):
    """TrilIndicesFrom operator definition."""

    op_name = "TrilIndicesFrom"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("TriuIndices")
class TriuIndices(OpDef):
    """TriuIndices operator definition."""

    op_name = "TriuIndices"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("TriuIndicesFrom")
class TriuIndicesFrom(OpDef):
    """TriuIndicesFrom operator definition."""

    op_name = "TriuIndicesFrom"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("Logspace")
class Logspace(CreationOp):
    """Logspace op."""

    op_name = "Logspace"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (kwargs.get("num", args[2] if len(args) > 2 and isinstance(args[2], int) else 50),)


@register_op("Frombuffer")
class Frombuffer(OpDef):
    """Operator Frombuffer."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            args: Positional args.
            kwargs: Keyword args.

        Args:
            message (str): The message.
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.

        Returns:
        Any: Result.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        # Typically 1D array of elements depending on dtype and buffer size
        if "count" in kwargs and kwargs["count"] != -1:
            return (kwargs["count"],)
        return None  # Dynamic otherwise
