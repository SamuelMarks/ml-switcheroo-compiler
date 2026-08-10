# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for products.py."""

from typing import Any

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.ir.shape_system import matmul_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("BandPart")
class BandPart(OpDef):
    """BandPart operator.

    Extracts a central band of a tensor.
    """

    def infer_shape(self, input: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            input (object): The input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return input if isinstance(input, tuple) else None


@register_op("Diag")
class Diag(OpDef):
    """Diag operator.

    Extracts a diagonal or constructs a diagonal array.
    """

    def infer_shape(self, input: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            input (object): The input parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if isinstance(input, tuple):
            if len(input) == 1:
                return (input[0], input[0])
            elif len(input) >= MAGIC_VAL_2:
                return input[:-1]
        return None


@register_op("Matmul")
class Matmul(OpDef):
    """Matrix multiplication operator.

    Computes the matrix product of two arrays
    """

    def infer_shape(self, a: Any, b: Any, **kwargs: Any) -> Any:
        """Infer the output shape of the operation.

        Args:
            a (object): The first input tensor.
            b (object): The second input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            try:
                return matmul_shape(a, b)
            except ValueError:
                return None
        return None


def _has_valid_shape(obj: Any) -> bool:
    """Evaluate _has_valid_shape operation.

    Args:
        obj (object): The obj parameter.

    Returns:
        bool: Result.
    """
    return hasattr(obj, "shape") and bool(obj.shape)


@register_op("MatrixPower")
class MatrixPower(OpDef):
    """Matrix power operator.

    Computes the matrix power of a square matrix.
    """

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            a (object): The a parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer the output shape of the operation.

        Args:
            a (object): The input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns: Any: The evaluated output resulting from this operation.
        """
        if hasattr(a, "shape"):
            return a.shape
        return ()


@register_op("Trace")
class Trace(OpDef):
    """Trace operator."""

    op_name = "Trace"

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            a (object): The a parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        shape = list(a.shape)
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        if len(shape) >= 2:
            shape.pop(max(axis1, axis2))
            shape.pop(min(axis1, axis2))
        return tuple(shape)


@register_op("MatrixRank")
class MatrixRank(OpDef):
    """MatrixRank operator."""

    op_name = "MatrixRank"

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            a (object): The a parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        shape = list(a.shape)
        if len(shape) >= 2:
            shape.pop()
            shape.pop()
        return tuple(shape)


@register_op("MatrixTranspose")
class MatrixTranspose(OpDef):
    """MatrixTranspose operator."""

    op_name = "MatrixTranspose"

    def infer_shape(self, a: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            a (object): The a parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        shape = list(a.shape)
        if len(shape) >= 2:
            shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)


@register_op("Adjoint")
class Adjoint(OpDef):
    """Adjoint operator."""

    op_name = "Adjoint"

    def infer_shape(self, matrix: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            matrix (object): The matrix parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        shape = list(matrix.shape)
        if len(shape) >= 2:
            shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)


@register_op("Diagonal")
class Diagonal(OpDef):
    """Diagonal operator definition."""

    op_name = "Diagonal"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("EinsumPath")
class EinsumPath(OpDef):
    """EinsumPath operator definition."""

    op_name = "EinsumPath"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        # Typically returns a tuple representing the path and a string representation.
        return ()


@register_op("MultiDot")
class MultiDot(OpDef):
    """MultiDot operator definition."""

    op_name = "MultiDot"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()
