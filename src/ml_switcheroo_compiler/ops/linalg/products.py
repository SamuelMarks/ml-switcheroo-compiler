"""Operations and utilities for `ml_switcheroo_compiler.ops.linalg.products`."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import coerce_int
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.shape import OpValue, ShapeProtocol, extract_shape
from ml_switcheroo_compiler.ir.shape_system import matmul_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("BandPart")
class BandPart(OpDef):
    """BandPart operator.

    Extracts a central band of a tensor.
    """

    def infer_shape(self, input: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            input (OpValue): Input `input` argument (input).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return input if isinstance(input, tuple) else None


@register_op("Diag")
class Diag(OpDef):
    """Diag operator.

    Extracts a diagonal or constructs a diagonal array.
    """

    def infer_shape(self, input: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            input (OpValue): Input `input` argument (input).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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

    def infer_shape(self, a: OpValue, b: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape of the operation.

        Args:
            a (OpValue): The first input tensor.
            b (OpValue): The second input tensor.
            **kwargs (OpValue): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if isinstance(a, tuple) and isinstance(b, tuple):
            try:
                return matmul_shape(a, b)
            except ValueError:
                return None
        return None


def _has_valid_shape(obj: OpValue) -> bool:
    """Compute the has valid shape tensor operation.

    Args:
        obj (OpValue): Input `obj` argument (obj).

    Returns:
        bool: Computed `bool` output.
    """
    return isinstance(obj, ShapeProtocol) and bool(obj.shape)


@register_op("MatrixPower")
class MatrixPower(OpDef):
    """Matrix power operator.

    Computes the matrix power of a square matrix.
    """

    def infer_shape(self, a: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer the output shape of the MatrixPower operation.

        Args:
            a: The input tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape of the matrix power result.
        """
        del kwargs
        return extract_shape(a)


@register_op("Trace")
class Trace(OpDef):
    """Trace operator."""

    op_name = "Trace"

    def infer_shape(self, a: ShapeProtocol, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            a (ShapeProtocol): Input `a` argument (a).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        shape = list(extract_shape(a))
        axis1 = kwargs.get("axis1", 0)
        axis2 = kwargs.get("axis2", 1)
        if len(shape) >= 2:
            shape.pop(max(int(coerce_int(axis1)), int(coerce_int(axis2))))
            shape.pop(min(int(coerce_int(axis1)), int(coerce_int(axis2))))
        return tuple(shape)


@register_op("MatrixRank")
class MatrixRank(OpDef):
    """MatrixRank operator."""

    op_name = "MatrixRank"

    def infer_shape(self, a: ShapeProtocol, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            a (ShapeProtocol): Input `a` argument (a).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        shape = list(extract_shape(a))
        if len(shape) >= 2:
            shape.pop()
            shape.pop()
        return tuple(shape)


@register_op("MatrixTranspose")
class MatrixTranspose(OpDef):
    """MatrixTranspose operator."""

    op_name = "MatrixTranspose"

    def infer_shape(self, a: ShapeProtocol, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            a (ShapeProtocol): Input `a` argument (a).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        shape = list(extract_shape(a))
        if len(shape) >= 2:
            shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)


@register_op("Adjoint")
class Adjoint(OpDef):
    """Adjoint operator."""

    op_name = "Adjoint"

    def infer_shape(self, matrix: ShapeProtocol, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            matrix (ShapeProtocol): Input `matrix` argument (matrix).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        shape = list(extract_shape(matrix))
        if len(shape) >= 2:
            shape[-1], shape[-2] = shape[-2], shape[-1]
        return tuple(shape)


@register_op("Diagonal")
class Diagonal(OpDef):
    """Diagonal operator definition."""

    op_name = "Diagonal"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("EinsumPath")
class EinsumPath(OpDef):
    """EinsumPath operator definition."""

    op_name = "EinsumPath"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        # Typically returns a tuple representing the path and a string representation.
        return ()


@register_op("MultiDot")
class MultiDot(OpDef):
    """MultiDot operator definition."""

    op_name = "MultiDot"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("Sqrtm")
class Sqrtm(OpDef):
    """Principal matrix square-root operator definition."""

    op_name = "Sqrtm"

    def infer_shape(self, a: ShapeProtocol, **kwargs: OpValue) -> OpValue:
        """Infer output shape for `Sqrtm`.

        Args:
            a (ShapeProtocol): Input square matrix tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Output shape matching `a.shape`.
        """
        return extract_shape(a)


@register_op("EighTridiagonal")
class EighTridiagonal(OpDef):
    """Real symmetric tridiagonal eigensolver operator definition."""

    op_name = "EighTridiagonal"

    def infer_shape(
        self,
        alpha: ShapeProtocol,
        beta: ShapeProtocol,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer `(eigenvalues_shape, eigenvectors_shape)` for `EighTridiagonal`.

        Args:
            alpha (ShapeProtocol): Main diagonal elements.
            beta (ShapeProtocol): Off-diagonal elements.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Pair of `(eigvals_shape, eigvecs_shape)`.
        """
        return (extract_shape(alpha), list(extract_shape(alpha)) + [extract_shape(alpha)[-1]])


@register_op("TridiagonalMatmul")
class TridiagonalMatmul(OpDef):
    """Tridiagonal matrix multiplication operator definition."""

    op_name = "TridiagonalMatmul"

    def infer_shape(
        self,
        dl: ShapeProtocol,
        d: ShapeProtocol,
        du: ShapeProtocol,
        b: ShapeProtocol,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer output shape for tridiagonal matrix-matrix multiplication.

        Args:
            dl (ShapeProtocol): Subdiagonal vector/tensor.
            d (ShapeProtocol): Main diagonal vector/tensor.
            du (ShapeProtocol): Superdiagonal vector/tensor.
            b (ShapeProtocol): Right-hand side matrix tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Output shape matching `b.shape`.
        """
        return extract_shape(b)
