"""Core shape utility and broadcast verification functions."""

from __future__ import annotations

from typing import overload


@overload
def _broadcast_dim(
    d1: int,
    d2: int,
    shape1: tuple[int, ...],
    shape2: tuple[int, ...],
) -> int: ...


@overload
def _broadcast_dim(
    d1: int | str,
    d2: int | str,
    shape1: tuple[int | str, ...],
    shape2: tuple[int | str, ...],
) -> int | str: ...


def _broadcast_dim(
    d1: int | str,
    d2: int | str,
    shape1: tuple[int | str, ...],
    shape2: tuple[int | str, ...],
) -> int | str:
    """Broadcast a single dimension supporting static and dynamic/symbolic dimensions.

    Args:
        d1 (int | str): Dimension size from first shape.
        d2 (int | str): Dimension size from second shape.
        shape1 (tuple[int | str, ...]): First complete shape tuple.
        shape2 (tuple[int | str, ...]): Second complete shape tuple.

    Returns:
        int | str: Resulting broadcasted dimension.

    Raises:
        ShapeMismatchError: If the two dimensions cannot broadcast together.
    """
    if d1 == d2:
        return d1
    if d1 in (1, "?", -1):
        return d2
    if d2 in (1, "?", -1):
        return d1
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    raise ShapeMismatchError(f"Shapes {shape1} and {shape2} are incompatible at dimension {d1} vs {d2}.")


@overload
def broadcast_shapes(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...],
) -> tuple[int, ...]: ...


@overload
def broadcast_shapes(
    shape1: tuple[int | str, ...],
    shape2: tuple[int | str, ...],
) -> tuple[int | str, ...]: ...


def broadcast_shapes(
    shape1: tuple[int | str, ...],
    shape2: tuple[int | str, ...],
) -> tuple[int | str, ...]:
    """Calculate the broadcasted shape of two tuples with symbolic and dynamic dimension support.

    Args:
        shape1 (tuple[int | str, ...]): First operand shape tuple.
        shape2 (tuple[int | str, ...]): Second operand shape tuple.

    Returns:
        tuple[int | str, ...]: Unified broadcasted shape tuple.
    """
    ndim: int = max(len(shape1), len(shape2))
    shape1_pad: tuple[int | str, ...] = (1,) * (ndim - len(shape1)) + tuple(shape1)
    shape2_pad: tuple[int | str, ...] = (1,) * (ndim - len(shape2)) + tuple(shape2)

    result: list[int | str] = []
    for d1, d2 in zip(shape1_pad, shape2_pad):
        result.append(_broadcast_dim(d1, d2, shape1, shape2))
    return tuple(result)  # type: ignore[return-value]


def _validate_strict_2d_matmul(
    shape_a: tuple[int | str, ...],
    shape_b: tuple[int | str, ...],
) -> tuple[int | str, ...]:
    """Validate strict 2-D matrix multiplication dimensions.

    Args:
        shape_a (tuple[int | str, ...]): Left-hand operand shape.
        shape_b (tuple[int | str, ...]): Right-hand operand shape.

    Returns:
        tuple[int | str, ...]: Resulting output 2-D shape.

    Raises:
        ShapeMismatchError: If operand ranks are not 2 or contracting dims mismatch.
    """
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    if len(shape_a) != 2 or len(shape_b) != 2:
        raise ShapeMismatchError(f"Strict 2-D matrix multiplication requires 2-D tensors, got {shape_a} and {shape_b}.")
    if shape_a[1] != shape_b[0] and shape_a[1] not in ("?", -1) and shape_b[0] not in ("?", -1):
        raise ShapeMismatchError(f"Incompatible contracting dimensions for matrix multiply: {shape_a} and {shape_b}.")
    return (shape_a[0], shape_b[1])


def _validate_vector_or_batched_matmul(
    shape_a: tuple[int | str, ...],
    shape_b: tuple[int | str, ...],
) -> tuple[int | str, ...]:
    """Validate 1-D vector products and batched matrix multiplication shapes.

    Args:
        shape_a (tuple[int | str, ...]): Left-hand operand shape.
        shape_b (tuple[int | str, ...]): Right-hand operand shape.

    Returns:
        tuple[int | str, ...]: Resulting output tensor shape.

    Raises:
        ShapeMismatchError: If contracting dimensions or ranks mismatch.
    """
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    if len(shape_a) == 1 and len(shape_b) == 1:
        if shape_a[0] != shape_b[0] and shape_a[0] not in ("?", -1) and shape_b[0] not in ("?", -1):
            raise ShapeMismatchError(f"Incompatible 1-D dot product dimensions: {shape_a[0]} and {shape_b[0]}.")
        return ()

    if len(shape_a) == 1:
        if shape_a[0] != shape_b[-2] and shape_a[0] not in ("?", -1) and shape_b[-2] not in ("?", -1):
            raise ShapeMismatchError(f"Incompatible contracting dimensions for vector-matrix multiply: {shape_a[0]} and {shape_b[-2]}.")
        return shape_b[:-2] + (shape_b[-1],)

    if len(shape_b) == 1:
        if shape_a[-1] != shape_b[0] and shape_a[-1] not in ("?", -1) and shape_b[0] not in ("?", -1):
            raise ShapeMismatchError(f"Incompatible contracting dimensions for matrix-vector multiply: {shape_a[-1]} and {shape_b[0]}.")
        return shape_a[:-1]

    k_a = shape_a[-1]
    k_b = shape_b[-2]
    if k_a != k_b and k_a not in ("?", -1) and k_b not in ("?", -1):
        raise ShapeMismatchError(f"Incompatible contracting dimensions for batched matrix multiply: {k_a} and {k_b}.")

    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    batch_out = broadcast_shapes(batch_a, batch_b)
    return batch_out + (shape_a[-2], shape_b[-1])


@overload
def validate_matrix_multiply_shapes(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    strict_2d: bool = ...,
) -> tuple[int, ...]: ...


@overload
def validate_matrix_multiply_shapes(
    shape_a: tuple[int | str, ...],
    shape_b: tuple[int | str, ...],
    strict_2d: bool = ...,
) -> tuple[int | str, ...]: ...


def validate_matrix_multiply_shapes(
    shape_a: tuple[int | str, ...],
    shape_b: tuple[int | str, ...],
    strict_2d: bool = False,
) -> tuple[int | str, ...]:
    """Validate matrix multiplication dimensions and compute the resulting output shape.

    Args:
        shape_a (tuple[int | str, ...]): Left-hand operand shape.
        shape_b (tuple[int | str, ...]): Right-hand operand shape.
        strict_2d (bool): If True, require strictly 2-D matrix operands. Defaults to False.

    Returns:
        tuple[int | str, ...]: Resulting output tensor shape after matrix multiplication.

    Raises:
        ShapeMismatchError: If contracting dimensions do not match or operands violate rank rules.
    """
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    if len(shape_a) == 0 or len(shape_b) == 0:
        raise ShapeMismatchError("Matrix multiplication requires at least 1-D tensors.")

    if strict_2d:
        return _validate_strict_2d_matmul(shape_a, shape_b)

    return _validate_vector_or_batched_matmul(shape_a, shape_b)
