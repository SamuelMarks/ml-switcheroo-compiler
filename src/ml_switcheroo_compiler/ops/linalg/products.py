"""Implementation of products operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.products`)."""

from typing import cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ir.shape_system import matmul_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("BandPart")
class BandPart(OpDef):
    """BandPart operator.

    Extracts a central band of a tensor.
    """

    def infer_shape(self, input: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            input (EagerTensorProtocol): The input parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return input if isinstance(input, tuple) else None


@register_op("Diag")
class Diag(OpDef):
    """Diag operator.

    Extracts a diagonal or constructs a diagonal array.
    """

    def infer_shape(self, input: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            input (EagerTensorProtocol): The input parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        if isinstance(input, tuple):
            if len(input) == 1:
                return (input[0], input[0])
            elif len(input) >= MATRIX_RANK:
                return input[:-1]
        return None


@register_op("Matmul")
class Matmul(OpDef):
    """Matrix multiplication operator.

    Computes the matrix product of two arrays
    """

    def infer_shape(self, a: EagerTensorProtocol, b: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape of the operation.

        Args:
            a (EagerTensorProtocol): The first input tensor.
            b (EagerTensorProtocol): The second input tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            try:
                return matmul_shape(a, b)
            except ValueError:
                return None
        return None


def _has_valid_shape(obj: TensorOrScalar) -> bool:
    """Compute the has valid shape tensor operation.

    Args:
        obj (TensorOrScalar): The obj parameter.

    Returns:
        bool: Result.
    """
    return hasattr(obj, "shape") and bool(getattr(obj, "shape"))


@register_op("MatrixPower")
class MatrixPower(OpDef):
    """Matrix power operator.

    Computes the matrix power of a square matrix.
    """

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer the output shape of the MatrixPower operation.

        Args:
            a: The input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape of the matrix power result.
        """
        del kwargs
        if hasattr(a, "shape"):
            return a.shape
        return ()


@register_op("Trace")
class Trace(OpDef):
    """Trace operator."""

    op_name = "Trace"

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            a (EagerTensorProtocol): The a parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = list(a.shape)
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        if len(shape) >= 2:
            shape.pop(max(int(cast(int, axis1)), int(cast(int, axis2))))
            shape.pop(min(int(cast(int, axis1)), int(cast(int, axis2))))
        return tuple(shape)


@register_op("MatrixRank")
class MatrixRank(OpDef):
    """MatrixRank operator."""

    op_name = "MatrixRank"

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            a (EagerTensorProtocol): The a parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
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

    def infer_shape(self, a: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            a (EagerTensorProtocol): The a parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = list(a.shape)
        if len(shape) >= 2:
            shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)


@register_op("Adjoint")
class Adjoint(OpDef):
    """Adjoint operator."""

    op_name = "Adjoint"

    def infer_shape(self, matrix: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            matrix (EagerTensorProtocol): The matrix parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = list(matrix.shape)
        if len(shape) >= 2:
            shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)


@register_op("Diagonal")
class Diagonal(OpDef):
    """Diagonal operator definition."""

    op_name = "Diagonal"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("EinsumPath")
class EinsumPath(OpDef):
    """EinsumPath operator definition."""

    op_name = "EinsumPath"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        # Typically returns a tuple representing the path and a string representation.
        return ()


@register_op("MultiDot")
class MultiDot(OpDef):
    """MultiDot operator definition."""

    op_name = "MultiDot"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()
