"""Type & Shape System for IR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.core import TensorSpec


class SymInt:
    """Symbolic Integer to trace graphs with dynamic dimensions."""

    def __init__(self, name_or_expr: str) -> None:
        """Initialize SymInt.

        name_or_expr (str): Argument name_or_expr

        Args:
            name_or_expr (str): Argument name_or_expr
        """
        self.expr = str(name_or_expr)

    def __add__(self, other: SymInt | int) -> SymInt:
        """Evaluate add.

        Args:
            other (Union['SymInt', int]): Argument other

        Returns:
            'SymInt': The result of the operation
        """
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} + {other.expr})")
        return SymInt(f"({self.expr} + {other})")

    def __radd__(self, other: int) -> SymInt:
        """Evaluate reverse add.

        Args:
            other (int): Argument other

        Returns:
            'SymInt': The result of the operation
        """
        return SymInt(f"({other} + {self.expr})")

    def __sub__(self, other: SymInt | int) -> SymInt:
        """Evaluate sub.

        Args:
            other (Union['SymInt', int]): Argument other

        Returns:
            'SymInt': The result of the operation
        """
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} - {other.expr})")
        return SymInt(f"({self.expr} - {other})")

    def __rsub__(self, other: int) -> SymInt:
        """Evaluate reverse sub.

        Args:
            other (int): Argument other

        Returns:
            'SymInt': The result of the operation
        """
        return SymInt(f"({other} - {self.expr})")

    def __mul__(self, other: SymInt | int) -> SymInt:
        """Evaluate mul.

        Args:
            other (Union['SymInt', int]): Argument other

        Returns:
            'SymInt': The result of the operation
        """
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} * {other.expr})")
        return SymInt(f"({self.expr} * {other})")

    def __rmul__(self, other: int) -> SymInt:
        """Evaluate reverse mul.

        Args:
            other (int): Argument other

        Returns:
            'SymInt': The result of the operation
        """
        return SymInt(f"({other} * {self.expr})")

    def __floordiv__(self, other: SymInt | int) -> SymInt:
        """Evaluate floordiv.

        Args:
            other (Union['SymInt', int]): Argument other

        Returns:
            'SymInt': The result of the operation
        """
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} // {other.expr})")
        return SymInt(f"({self.expr} // {other})")

    def __str__(self) -> str:
        """Evaluate str.

        Returns:
            str: The computed result.
        """
        return str(self.expr)

    def __repr__(self) -> str:
        """Evaluate repr.

        Returns:
            str: The computed result.
        """
        return f"SymInt({self.expr})"

    def __eq__(self, other: object) -> bool:
        """Evaluate eq.

        Args:
            other (object): Argument other


        Returns:
            bool: The computed result.
        """
        if isinstance(other, SymInt):
            return self.expr == other.expr
        return False


class SymbolicSolver:
    """Symbolic Expression Solver to validate shape consistency."""

    @staticmethod
    def is_consistent(expr1: SymInt | int, expr2: SymInt | int) -> bool:
        """Check if two symbolic expressions are mathematically equivalent.

        expr1 (Union[SymInt, int]): Argument expr1
            expr2 (Union[SymInt, int]): Argument expr2

        Args:
            expr1 (Union[SymInt, int]): Argument expr1
            expr2 (Union[SymInt, int]): Argument expr2


        Returns:
            bool: The computed result.
        """
        if isinstance(expr1, int) and isinstance(expr2, int):
            return expr1 == expr2

        sym1 = expr1.expr if isinstance(expr1, SymInt) else str(expr1)
        sym2 = expr2.expr if isinstance(expr2, SymInt) else str(expr2)

        # Naive string comparison for now since we can't use sympy
        return sym1 == sym2


class ShapeTracker:
    """Calculates exact output shapes given input TensorSpecs."""

    @staticmethod
    def infer_elementwise(inputs: list[TensorSpec]) -> tuple[int | SymInt, ...]:
        """Infer shape for elementwise operations requiring broadcasting.

        inputs (list[TensorSpec]): Argument inputs

        Returns:
            tuple[Union[int, SymInt], ...]: The result of the operation

        Args:
            inputs (list[TensorSpec]): Argument inputs
        """
        if not inputs:
            return ()

        shape = tuple(str(x) if isinstance(x, SymInt) else x for x in inputs[0].shape)
        for i in range(1, len(inputs)):
            s2 = tuple(str(x) if isinstance(x, SymInt) else x for x in inputs[i].shape)
            shape = broadcast_shapes(shape, s2)

        # Reconstruct SymInts
        out_shape = []
        for dim in shape:
            if isinstance(dim, str) and not dim.isdigit():
                out_shape.append(SymInt(dim))
            else:
                out_shape.append(int(dim))
        return tuple(out_shape)

    @staticmethod
    def infer_matmul(
        input1: TensorSpec,
        input2: TensorSpec,
    ) -> tuple[int | SymInt, ...]:
        """Infer shape for matrix multiplication.

        input1 (TensorSpec): Argument input1
            input2 (TensorSpec): Argument input2

        Returns:
            tuple[Union[int, SymInt], ...]: The result of the operation

        Args:
            input1 (TensorSpec): Argument input1
            input2 (TensorSpec): Argument input2
        """
        s1 = tuple(str(x) if isinstance(x, SymInt) else x for x in input1.shape)
        s2 = tuple(str(x) if isinstance(x, SymInt) else x for x in input2.shape)

        out_shape_str = matmul_shape(s1, s2)

        out_shape = []
        for dim in out_shape_str:
            if isinstance(dim, str) and not dim.isdigit():
                out_shape.append(SymInt(dim))
            else:
                out_shape.append(int(dim))
        return tuple(out_shape)


ShapeType = tuple[Union[int, str], ...]


def broadcast_shapes(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Broadcast two shapes together according to numpy rules.

    shape_a (ShapeType): First shape
    shape_b (ShapeType): Second shape

    Returns:
    ShapeType: The broadcasted shape

    Raises:
    ValueError: If shapes are not compatible for broadcasting

    Args:
    shape_a (ShapeType): The shape of the tensor._a
    shape_b (ShapeType): The shape of the tensor._b
    """
    out_shape = []

    max_len = max(len(shape_a), len(shape_b))
    pad_a = (1,) * (max_len - len(shape_a)) + shape_a
    pad_b = (1,) * (max_len - len(shape_b)) + shape_b

    for a, b in zip(pad_a, pad_b):
        if a == b:
            out_shape.append(a)
        elif a == 1:
            out_shape.append(b)
        elif b == 1:
            out_shape.append(a)
        elif isinstance(a, str) or isinstance(b, str):
            # Simple symbolic equality handled above; complex logic (e.g., 'B' vs 'T')
            # usually errors out or requires explicit constraints. Here we assume
            # unmatched symbols don't broadcast unless one is 1
            msg = f"Cannot broadcast symbolic dimensions {a} and {b}"
            raise ValueError(msg)
        else:
            msg = f"Incompatible dimensions for broadcasting: {a} and {b}"
            raise ValueError(msg)

    return tuple(out_shape)


def matmul_shape(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a matrix multiplication.

    shape_a (ShapeType): LHS shape
    shape_b (ShapeType): RHS shape

    Returns:
    ShapeType: Output shape

    Raises:
    ValueError: If shapes are incompatible

    Args:
    shape_a (ShapeType): The shape of the tensor._a
    shape_b (ShapeType): The shape of the tensor._b
    """
    if len(shape_a) == 0 or len(shape_b) == 0:
        msg = "Scalars cannot be matrix multiplied."
        raise ValueError(msg)

    # 1D dot product
    if len(shape_a) == 1 and len(shape_b) == 1:
        if shape_a[0] != shape_b[0]:
            msg = f"Incompatible 1D dot product shapes: {shape_a}, {shape_b}"
            raise ValueError(
                msg,
            )
        return ()

    # Standard 2D matrix multiplication
    if len(shape_a) == 2 and len(shape_b) == 2:
        if shape_a[1] != shape_b[0]:
            msg = f"Incompatible 2D matmul shapes: {shape_a}, {shape_b}"
            raise ValueError(msg)
        return (shape_a[0], shape_b[1])

    # Batched matrix multiplication (numpy style)
    # LHS: (..., M, K), RHS: (..., K, N) -> (..., M, N)
    # Broadcast batch dimensions
    batch_a = shape_a[:-2] if len(shape_a) > 2 else ()
    batch_b = shape_b[:-2] if len(shape_b) > 2 else ()

    out_batch = broadcast_shapes(batch_a, batch_b)

    m_dim = shape_a[-2] if len(shape_a) > 1 else 1
    k_dim_a = shape_a[-1]
    k_dim_b = shape_b[-2] if len(shape_b) > 1 else shape_b[-1]
    n_dim = shape_b[-1] if len(shape_b) > 1 else 1

    if k_dim_a != k_dim_b:
        msg = f"Incompatible inner dimensions for matmul: {k_dim_a} and {k_dim_b}"
        raise ValueError(
            msg,
        )

    out_shape = list(out_batch)
    if len(shape_a) > 1:
        out_shape.append(m_dim)
    if len(shape_b) > 1:
        out_shape.append(n_dim)

    return tuple(out_shape)


def normalize_axis(
    axis: int | tuple[int, ...],
    ndim: int,
) -> int | tuple[int, ...]:
    """Normalize a negative axis or tuple of axes to be positive.

    axis: The axis or axes to normalize
    ndim: The number of dimensions of the tensor

    Returns:
    The normalized axis or axes

    Raises:
    ValueError: If an axis is out of bounds [-ndim, ndim-1]

    Args:
    axis (Union[int, tuple[int, ...]]): The axis to operate along
    ndim (int): Argument ndim
    """
    if isinstance(axis, int):
        if axis < -ndim or axis >= ndim:
            msg = f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
            raise ValueError(
                msg,
            )
        if axis < 0:
            return axis + ndim
        return axis
    if isinstance(axis, (tuple, list)):
        normalized = []
        for ax in axis:
            if ax < -ndim or ax >= ndim:
                msg = f"Axis {ax} is out of bounds for tensor of dimension {ndim}"
                raise ValueError(
                    msg,
                )
            if ax < 0:
                normalized.append(ax + ndim)
            else:
                normalized.append(ax)
        return tuple(normalized)
    msg = f"Invalid type for axis: {type(axis)}"
    raise TypeError(msg)
