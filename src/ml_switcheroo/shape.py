"""Shape inference utilities mimicking numpy broadcasting rules."""

from typing import Tuple, Union

ShapeType = Tuple[Union[int, str], ...]


def broadcast_shapes(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Broadcast two shapes together according to numpy rules.

    Args:
        shape_a (ShapeType): First shape.
        shape_b (ShapeType): Second shape.

    Returns:
        ShapeType: The broadcasted shape.

    Raises:
        ValueError: If shapes are not compatible for broadcasting.
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
            # unmatched symbols don't broadcast unless one is 1.
            raise ValueError(f"Cannot broadcast symbolic dimensions {a} and {b}")
        else:
            raise ValueError(f"Incompatible dimensions for broadcasting: {a} and {b}")

    return tuple(out_shape)


def matmul_shape(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Calculate the output shape for a matrix multiplication.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: Output shape.

    Raises:
        ValueError: If shapes are incompatible.
    """
    if len(shape_a) == 0 or len(shape_b) == 0:
        raise ValueError("Scalars cannot be matrix multiplied.")

    # 1D dot product
    if len(shape_a) == 1 and len(shape_b) == 1:
        if shape_a[0] != shape_b[0]:
            raise ValueError(
                f"Incompatible 1D dot product shapes: {shape_a}, {shape_b}"
            )
        return ()

    # Standard 2D matrix multiplication
    if len(shape_a) == 2 and len(shape_b) == 2:
        if shape_a[1] != shape_b[0]:
            raise ValueError(f"Incompatible 2D matmul shapes: {shape_a}, {shape_b}")
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
        raise ValueError(
            f"Incompatible inner dimensions for matmul: {k_dim_a} and {k_dim_b}"
        )

    out_shape = list(out_batch)
    if len(shape_a) > 1:
        out_shape.append(m_dim)
    if len(shape_b) > 1:
        out_shape.append(n_dim)

    return tuple(out_shape)


def normalize_axis(
    axis: Union[int, Tuple[int, ...]], ndim: int
) -> Union[int, Tuple[int, ...]]:
    """Normalize a negative axis or tuple of axes to be positive.

    Args:
        axis: The axis or axes to normalize.
        ndim: The number of dimensions of the tensor.

    Returns:
        The normalized axis or axes.

    Raises:
        ValueError: If an axis is out of bounds [-ndim, ndim-1].
    """
    if isinstance(axis, int):
        if axis < -ndim or axis >= ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor of dimension {ndim}"
            )
        if axis < 0:
            return axis + ndim
        return axis
    elif isinstance(axis, (tuple, list)):
        normalized = []
        for ax in axis:
            if ax < -ndim or ax >= ndim:
                raise ValueError(
                    f"Axis {ax} is out of bounds for tensor of dimension {ndim}"
                )
            if ax < 0:
                normalized.append(ax + ndim)
            else:
                normalized.append(ax)
        return tuple(normalized)
    else:
        raise TypeError(f"Invalid type for axis: {type(axis)}")
