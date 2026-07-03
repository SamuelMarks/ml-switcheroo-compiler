"""Module docstring."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.ir.shape_system import matmul_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("BandPart")
class BandPart(OpDef):
    """BandPart operator.

    Extracts a central band of a tensor.
    """

    def infer_shape(self, input: object, **kwargs: object) -> object:
        """Infer shape."""
        return input if isinstance(input, tuple) else None


@register_op("Diag")
class Diag(OpDef):
    """Diag operator.

    Extracts a diagonal or constructs a diagonal array.
    """

    def infer_shape(self, input: object, **kwargs: object) -> object:
        """Infer shape."""
        if isinstance(input, tuple):  # pragma: no branch
            if len(input) == 1:  # pragma: no branch
                return (input[0], input[0])  # pragma: no cover
            elif len(input) >= MAGIC_VAL_2:  # pragma: no branch
                return input[:-1]
        return None  # pragma: no cover


@register_op("Matmul")
class Matmul(OpDef):
    """Matrix multiplication operator.

    Computes the matrix product of two arrays
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
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


def _has_valid_shape(obj: object) -> bool:
    """Function docstring.

    Args:
        obj: Arg.
    """
    return hasattr(obj, "shape") and bool(obj.shape)


@register_op("MatrixPower")
class MatrixPower(OpDef):
    """Matrix power operator.

    Computes the matrix power of a square matrix.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
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

        Returns:
            object: The evaluated output resulting from this operation.
        """
        if hasattr(a, "shape"):
            return a.shape
        return ()


@register_op("Trace")
class Trace(OpDef):
    """Trace operator."""

    op_name = "Trace"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
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

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
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

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        shape = list(a.shape)
        if len(shape) >= 2:
            shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)


@register_op("Adjoint")
class Adjoint(OpDef):
    """Adjoint operator."""

    op_name = "Adjoint"

    def infer_shape(self, matrix: object, **kwargs: object) -> object:
        """Infer shape."""
        shape = list(matrix.shape)
        if len(shape) >= 2:
            shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)


@register_op("Diagonal")
class Diagonal(OpDef):
    """Diagonal operator definition."""

    op_name = "Diagonal"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("MultiDot")
class MultiDot(OpDef):
    """MultiDot operator definition."""

    op_name = "MultiDot"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()
