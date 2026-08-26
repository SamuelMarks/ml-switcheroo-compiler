"""Shape System."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Type & Shape System for IR."""


from typing import TYPE_CHECKING, Union

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.errors import ShapeMismatchError

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.core import TensorSpec

MAX_DIMENSIONS = 4
MAX_RANK = 5


class SymInt:
    """Symbolic Integer to trace graphs with dynamic dimensions."""

    def __init__(self, name_or_expr: str) -> None:
        """Initialize SymInt.

        Args:
            name_or_expr (str): The name_or_expr parameter.
        """
        self.expr = str(name_or_expr)

    def __add__(self, other: SymInt | int) -> SymInt:
        """Evaluate __add__ operation.

        Args:
        other (object): The other parameter.

        Returns:
        SymInt: Result.
        """
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} + {other.expr})")
        return SymInt(f"({self.expr} + {other})")

    def __radd__(self, other: int) -> SymInt:
        """Evaluate __radd__ operation.

        Args:
        other (int): The other parameter.

        Returns:
        SymInt: Result.
        """
        return SymInt(f"({other} + {self.expr})")

    def __sub__(self, other: SymInt | int) -> SymInt:
        """Evaluate __sub__ operation.

        Args:
        other (object): The other parameter.

        Returns:
        SymInt: Result.
        """
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} - {other.expr})")
        return SymInt(f"({self.expr} - {other})")

    def __rsub__(self, other: int) -> SymInt:
        """Evaluate __rsub__ operation.

        Args:
        other (int): The other parameter.

        Returns:
        SymInt: Result.
        """
        return SymInt(f"({other} - {self.expr})")

    def __mul__(self, other: SymInt | int) -> SymInt:
        """Evaluate __mul__ operation.

        Args:
        other (object): The other parameter.

        Returns:
        SymInt: Result.
        """
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} * {other.expr})")
        return SymInt(f"({self.expr} * {other})")

    def __rmul__(self, other: int) -> SymInt:
        """Evaluate __rmul__ operation.

        Args:
        other (int): The other parameter.

        Returns:
        SymInt: Result.
        """
        return SymInt(f"({other} * {self.expr})")

    def __floordiv__(self, other: SymInt | int) -> SymInt:
        """Evaluate __floordiv__ operation.

        Args:
        other (object): The other parameter.

        Returns:
        SymInt: Result.
        """
        if isinstance(other, SymInt):
            return SymInt(f"({self.expr} // {other.expr})")
        return SymInt(f"({self.expr} // {other})")

    def __str__(self) -> str:
        """Evaluate __str__ operation.

        Returns:
        str: Result.
        """
        return str(self.expr)

    def __repr__(self) -> str:
        """Evaluate __repr__ operation.

        Returns:
        str: Result.
        """
        return f"SymInt({self.expr})"

    def __hash__(self) -> int:
        """Evaluate __hash__ operation.

        Returns:
        int: Result.
        """
        return hash(self.expr)

    def __eq__(self, other) -> bool:
        """Evaluate __eq__ operation.

        Args:
        other (object): The other parameter.

        Returns:
        bool: Result.
        """
        if isinstance(other, SymInt):
            return self.expr == other.expr
        return False


class SymbolicSolver:
    """Symbolic Expression Solver to validate shape consistency."""

    @staticmethod
    def is_consistent(expr1: SymInt | int, expr2: SymInt | int) -> bool:
        """Check if two symbolic expressions are mathematically equivalent.

        Args:
            expr1 (Union[SymInt, int]): The expr1 parameter.
            expr2 (Union[SymInt, int]): The expr2 parameter.

        Returns:
            bool: A boolean indicating the result of the check.
        """
        if isinstance(expr1, int) and isinstance(expr2, int):
            return expr1 == expr2

        sym1 = expr1.expr if isinstance(expr1, SymInt) else str(expr1)
        sym2 = expr2.expr if isinstance(expr2, SymInt) else str(expr2)

        # Naive string comparison for now since we can't use sympy
        return sym1 == sym2


def _to_str_shape(shape):
    """Convert shape to string representation.

    Args:
        shape (tuple): The shape tuple.

    Returns:
        tuple: The string represented shape.
    """
    return tuple(str(x) if isinstance(x, SymInt) else x for x in shape)


def _from_str_shape(shape):
    """Convert shape from string representation.

    Args:
        shape (tuple): The string represented shape tuple.

    Returns:
        tuple: The integer or SymInt shape.
    """
    out_shape = []
    for dim in shape:
        if isinstance(dim, str) and not dim.isdigit():
            out_shape.append(SymInt(dim))
        else:
            out_shape.append(int(dim))
    return tuple(out_shape)


class ShapeTracker:
    """Calculate exact output shapes given input TensorSpecs."""

    @staticmethod
    def infer_elementwise(inputs: list[TensorSpec]) -> tuple[int | SymInt, ...]:
        """Infer shape for elementwise operations requiring broadcasting.

        Returns:
            tuple[Union[int, SymInt], ...]: The inferred shape or computed result

        Args:
            inputs (list[TensorSpec]): The inputs for the operation.
        """
        if not inputs:
            return ()

        shape = _to_str_shape(inputs[0].shape)
        for i in range(1, len(inputs)):
            shape = broadcast_shapes(shape, _to_str_shape(inputs[i].shape))

        return _from_str_shape(shape)

    @staticmethod
    def infer_matmul(
        input1: TensorSpec,
        input2: TensorSpec,
    ) -> tuple[int | SymInt, ...]:
        """Infer shape for matrix multiplication.

        Returns:
            tuple[Union[int, SymInt], ...]: The inferred shape or computed result

        Args:
            input1 (TensorSpec): The input1 parameter.
            input2 (TensorSpec): The input2 parameter.
        """
        s1 = _to_str_shape(input1.shape)
        s2 = _to_str_shape(input2.shape)

        out_shape_str = matmul_shape(s1, s2)

        return _from_str_shape(out_shape_str)


ShapeType = tuple[Union[int, str], ...]


def _broadcast_dim(a: int | str, b: int | str) -> int | str:
    """Broadcast two dimensions.

    Args:
        a (int | str): The first dimension.
        b (int | str): The second dimension.

    Returns:
        int | str: The broadcasted dimension.

    Raises:
        ShapeMismatchError: If the dimensions are incompatible.
    """
    if a == b:
        return a
    if a == 1:
        return b
    if b == 1:
        return a
    if isinstance(a, str) or isinstance(b, str):
        msg = f"Cannot broadcast symbolic dimensions {a} and {b}"
        raise ShapeMismatchError(msg)
    msg = f"Incompatible dimensions for broadcasting: {a} and {b}"
    raise ShapeMismatchError(msg)


def broadcast_shapes(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Broadcast two shapes together according to numpy rules.

    Args:
        shape_a (ShapeType): First shape.
        shape_b (ShapeType): Second shape.

    Returns:
        ShapeType: The broadcasted shape.
    """
    max_len = max(len(shape_a), len(shape_b))
    pad_a = (1,) * (max_len - len(shape_a)) + shape_a
    pad_b = (1,) * (max_len - len(shape_b)) + shape_b
    return tuple(_broadcast_dim(a, b) for a, b in zip(pad_a, pad_b))


def _matmul_shape_1d(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a 1D dot product.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ValueError: If the shapes are incompatible.
    """
    if shape_a[0] != shape_b[0]:
        msg = f"Incompatible 1D dot product shapes: {shape_a}, {shape_b}"
        raise ValueError(
            msg,
        )
    return ()


def _matmul_shape_2d(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a 2D matrix multiplication.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ShapeMismatchError: If the shapes are incompatible.
    """
    if shape_a[1] != shape_b[0]:
        msg = f"Incompatible 2D matmul shapes: {shape_a}, {shape_b}"
        raise ShapeMismatchError(msg)
    return (shape_a[0], shape_b[1])


def _get_matmul_dims(shape_a: ShapeType, shape_b: ShapeType):
    """Evaluate _get_matmul_dims operation.

    Args:
        shape_a (ShapeType): The shape_a parameter.
        shape_b (ShapeType): The shape_b parameter.

    Returns:
        tuple: Result.
    """
    m_dim = shape_a[-2] if len(shape_a) > 1 else 1
    k_dim_a = shape_a[-1]
    k_dim_b = shape_b[-2] if len(shape_b) > 1 else shape_b[-1]
    n_dim = shape_b[-1] if len(shape_b) > 1 else 1
    return m_dim, k_dim_a, k_dim_b, n_dim


def _get_batch_dims(shape: ShapeType) -> ShapeType:
    """Evaluate _get_batch_dims operation.

    Args:
        shape (ShapeType): The shape parameter.

    Returns:
        ShapeType: Result.
    """
    return shape[:-2] if len(shape) > MAGIC_VAL_2 else ()


def _matmul_shape_batched(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a batched matrix multiplication.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ShapeMismatchError: If the inner dimensions are incompatible.
    """
    batch_a = _get_batch_dims(shape_a)
    batch_b = _get_batch_dims(shape_b)
    out_batch = broadcast_shapes(batch_a, batch_b)

    m_dim, k_dim_a, k_dim_b, n_dim = _get_matmul_dims(shape_a, shape_b)

    if k_dim_a != k_dim_b:
        msg = f"Incompatible inner dimensions for matmul: {k_dim_a} and {k_dim_b}"
        raise ShapeMismatchError(msg)

    out_shape = list(out_batch)
    if len(shape_a) > 1:
        out_shape.append(m_dim)
    if len(shape_b) > 1:
        out_shape.append(n_dim)

    return tuple(out_shape)


def matmul_shape(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a matrix multiplication.

    Args:
        shape_a (ShapeType): First shape.
        shape_b (ShapeType): Second shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ShapeMismatchError: If shapes are incompatible or scalars.
    """
    if len(shape_a) == 0 or len(shape_b) == 0:
        msg = "Scalars cannot be matrix multiplied."
        raise ShapeMismatchError(msg)

    # 1D dot product
    if len(shape_a) == 1 and len(shape_b) == 1:
        return _matmul_shape_1d(shape_a, shape_b)

    # Standard 2D matrix multiplication
    if len(shape_a) == MAGIC_VAL_2 and len(shape_b) == MAGIC_VAL_2:
        return _matmul_shape_2d(shape_a, shape_b)

    # Batched matrix multiplication (numpy style)
    return _matmul_shape_batched(shape_a, shape_b)


def _normalize_single_axis(axis: int, ndim: int) -> int:
    """Normalize a single negative axis to be positive.

    Args:
        axis (int): The axis.
        ndim (int): Number of dimensions.

    Returns:
        int: Normalized axis.

    Raises:
        ShapeMismatchError: If axis is out of bounds.
    """
    if axis < -ndim or axis >= ndim:
        msg = f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
        raise ShapeMismatchError(msg)
    if axis < 0:
        return axis + ndim
    return axis


def normalize_axis(
    axis: int | tuple[int, ...],
    ndim: int,
) -> int | tuple[int, ...]:
    """Normalize a negative axis or tuple of axes to be positive.

    Args:
        axis (Union[int, tuple[int, ...]]): The axis to operate along.
        ndim (int): Number of dimensions.

    Returns:
        int | tuple[int, ...]: Normalized axis or axes.

    Raises:
        TypeError: If axis has an invalid type.
    """
    if isinstance(axis, int):
        return _normalize_single_axis(axis, ndim)
    if isinstance(axis, (tuple, list)):
        return tuple(_normalize_single_axis(ax, ndim) for ax in axis)
    msg = f"Invalid type for axis: {type(axis)}"
    raise TypeError(msg)
