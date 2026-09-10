"""Declarative Dynamic Op Shape Inference Engine.

This module provides Pydantic schema models, symbolic dimension representations,
and declarative shape transfer functions for all operations in the compiler.
"""

import math
import os
from collections.abc import Callable, Sequence
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict

ShapeInferFn = Callable[..., "ShapeType"]
_SHAPE_INFERENCE_REGISTRY: dict[str, ShapeInferFn] = {}


def register_shape_inference(op_type: str) -> Callable[[ShapeInferFn], ShapeInferFn]:
    """Decorate to register a custom shape inference function for an op.

    Args:
        op_type (str): The op_type identifier.

    Returns:
        Callable[[ShapeInferFn], ShapeInferFn]: Decorator function.
    """

    def decorator(func: ShapeInferFn) -> ShapeInferFn:
        """Register the wrapped function in _SHAPE_INFERENCE_REGISTRY.

        Args:
            func (ShapeInferFn): Shape inference function to register.

        Returns:
            ShapeInferFn: Registered shape inference function.
        """
        _SHAPE_INFERENCE_REGISTRY[op_type] = func
        return func

    return decorator


class S:
    """Symbolic dimension variable (e.g., dynamic batch size S('batch'))."""

    name: str

    def __init__(self, name: str) -> None:
        """Initialize a symbolic dimension variable.

        Args:
            name (str): Identifier for the symbolic dimension.
        """
        self.name = name

    def __repr__(self) -> str:
        """Return developer representation of symbolic dimension.

        Returns:
            str: String representation in S('name') format.
        """
        return f"S({self.name!r})"

    def __str__(self) -> str:
        """Return human-readable dimension name.

        Returns:
            str: Symbolic name.
        """
        return self.name

    def __eq__(self, other: object) -> bool:
        """Compare two symbolic dimensions for equality.

        Args:
            other (object): Object to compare against.

        Returns:
            bool: True if both are S with matching names.
        """
        if isinstance(other, S):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        """Hash code for symbolic dimension.

        Returns:
            int: Hash integer.
        """
        return hash(self.name)


DimType = Union[int, S]
ShapeType = tuple[DimType, ...]


class ShapeOpSignature(BaseModel):
    """Declarative shape signature specification for an operation."""

    category: str
    pattern: Optional[str] = None
    description: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class ShapeSignaturesConfig(BaseModel):
    """Container schema for all declarative shape signatures."""

    operations: dict[str, ShapeOpSignature]
    model_config = ConfigDict(extra="allow")


_CACHED_SIGNATURES: Optional[ShapeSignaturesConfig] = None


def load_shape_signatures() -> ShapeSignaturesConfig:
    """Load and cache the declarative shape signatures from YAML.

    Returns:
        ShapeSignaturesConfig: Validated declarative shape signatures.
    """
    global _CACHED_SIGNATURES
    if _CACHED_SIGNATURES is not None:
        return _CACHED_SIGNATURES

    yaml_path = os.path.join(os.path.dirname(__file__), "shape_signatures.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    else:
        raw_data = {"operations": {}}

    _CACHED_SIGNATURES = ShapeSignaturesConfig.model_validate(raw_data)
    return _CACHED_SIGNATURES


def compute_contiguous_strides(shape: ShapeType) -> tuple[int, ...]:
    """Compute standard row-major contiguous strides for a tensor shape.

    Args:
        shape (ShapeType): Dimensions of the tensor.

    Returns:
        tuple[int, ...]: Contiguous stride tuple.
    """
    if not shape:
        return ()

    strides: list[int] = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        next_dim = shape[i + 1]
        if isinstance(next_dim, int):
            strides[i] = strides[i + 1] * next_dim
        else:
            strides[i] = strides[i + 1]
    return tuple(strides)


def broadcast_shapes(*shapes: ShapeType) -> ShapeType:
    """Compute multi-operand broadcast shape according to standard rules.

    Args:
        *shapes (ShapeType): One or more shapes to broadcast together.

    Returns:
        ShapeType: Resulting broadcasted shape tuple.

    Raises:
        ValueError: If shapes are incompatible for broadcasting.
    """
    if not shapes:
        return ()

    valid_shapes: list[ShapeType] = [s for s in shapes if s is not None]
    if not valid_shapes:
        return ()
    if len(valid_shapes) == 1:
        return valid_shapes[0]

    result_len = max(len(s) for s in valid_shapes)
    result_dims: list[DimType] = []

    for i in range(1, result_len + 1):
        current_dim: DimType = 1
        for s in valid_shapes:
            if i <= len(s):
                dim = s[-i]
                if dim == 1:
                    continue
                if current_dim == 1:
                    current_dim = dim
                elif current_dim != dim:
                    raise ValueError(f"Shape mismatch: cannot broadcast dimension {dim} with {current_dim}")
        result_dims.append(current_dim)

    result_dims.reverse()
    return tuple(result_dims)


def infer_matmul(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Infer the output shape of matrix multiplication.

    Supports vector-vector, matrix-vector, vector-matrix, matrix-matrix,
    and batched matrix multiplication with broadcasting.

    Args:
        shape_a (ShapeType): LHS operand shape.
        shape_b (ShapeType): RHS operand shape.

    Returns:
        ShapeType: Inferred result shape.

    Raises:
        ValueError: If input dimensions are empty or contracting dimensions mismatch.
    """
    if not shape_a or not shape_b:
        raise ValueError(f"Matrix multiplication requires non-empty operands, got {shape_a} and {shape_b}")

    # 1D @ 1D -> scalar ()
    if len(shape_a) == 1 and len(shape_b) == 1:
        if shape_a[0] != shape_b[0]:
            raise ValueError(f"Contracting dimension mismatch: {shape_a[0]} vs {shape_b[0]}")
        return ()

    # 2D @ 1D -> 1D (M,)
    if len(shape_a) == 2 and len(shape_b) == 1:
        if shape_a[1] != shape_b[0]:
            raise ValueError(f"Contracting dimension mismatch: {shape_a[1]} vs {shape_b[0]}")
        return (shape_a[0],)

    # 1D @ 2D -> 1D (N,)
    if len(shape_a) == 1 and len(shape_b) == 2:
        if shape_a[0] != shape_b[0]:
            raise ValueError(f"Contracting dimension mismatch: {shape_a[0]} vs {shape_b[0]}")
        return (shape_b[1],)

    # Multi-dimensional / Batched Matmul: (*B, M, K) @ (*B, K, N) -> (*B, M, N)
    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    batch_out = broadcast_shapes(batch_a, batch_b)

    k_a = shape_a[-1]
    k_b = shape_b[-2]
    if k_a != k_b:
        raise ValueError(f"Contracting dimension mismatch in matmul: {k_a} vs {k_b}")

    m = shape_a[-2]
    n = shape_b[-1]
    return (*batch_out, m, n)


def infer_dot(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Infer shape for dot or inner products.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: Inferred result shape.

    Raises:
        ValueError: If contracting dimensions mismatch.
    """
    if len(shape_a) == 1 and len(shape_b) == 1:
        if shape_a[0] != shape_b[0]:
            raise ValueError(f"Dot product dimension mismatch: {shape_a[0]} vs {shape_b[0]}")
        return ()
    return infer_matmul(shape_a, shape_b)


def infer_outer(shape_a: ShapeType, shape_b: ShapeType) -> ShapeType:
    """Infer shape of outer product between two tensors.

    Args:
        shape_a (ShapeType): LHS shape.
        shape_b (ShapeType): RHS shape.

    Returns:
        ShapeType: (shape_a[0], shape_b[0]) or concatenated shapes.
    """
    m = shape_a[0] if len(shape_a) == 1 else math.prod([d for d in shape_a if isinstance(d, int)])
    n = shape_b[0] if len(shape_b) == 1 else math.prod([d for d in shape_b if isinstance(d, int)])
    return (m, n)


def infer_conv(
    input_shape: ShapeType,
    weight_shape: ShapeType,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int], str] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    spatial_dims: int = 2,
) -> ShapeType:
    """Infer shape for 1D, 2D, or 3D convolutions.

    Formula:
        out_dim = floor((in_dim + 2 * pad - dilation * (kernel - 1) - 1) / stride) + 1

    Args:
        input_shape (ShapeType): Shape of input tensor (N, C_in, ...spatial).
        weight_shape (ShapeType): Shape of kernel weights (C_out, C_in, ...spatial).
        stride (Union[int, Sequence[int]]): Stride along spatial dimensions.
        padding (Union[int, Sequence[int], str]): Padding along spatial dimensions.
        dilation (Union[int, Sequence[int]]): Dilation along spatial dimensions.
        spatial_dims (int): Number of spatial dimensions (1, 2, or 3).

    Returns:
        ShapeType: Output tensor shape (N, C_out, ...spatial_out).

    Raises:
        ValueError: If input or weight dimensions are malformed or invalid.
    """
    expected_rank = spatial_dims + 2
    if len(input_shape) != expected_rank or len(weight_shape) != expected_rank:
        raise ValueError(f"Convolution expects rank {expected_rank}, got input {input_shape} and weight {weight_shape}")

    batch = input_shape[0]
    c_in = input_shape[1]
    c_out = weight_shape[0]
    w_c_in = weight_shape[1]

    if c_in != w_c_in:
        raise ValueError(f"Input channel count {c_in} does not match weight channel count {w_c_in}")

    strides = [stride] * spatial_dims if isinstance(stride, int) else list(stride)
    dilations = [dilation] * spatial_dims if isinstance(dilation, int) else list(dilation)

    pads: list[int] = []
    if isinstance(padding, int):
        pads = [padding] * spatial_dims
    elif isinstance(padding, str):
        pads = [0] * spatial_dims
    else:
        pads = list(padding)

    spatial_out: list[DimType] = []
    for i in range(spatial_dims):
        in_dim = input_shape[2 + i]
        kernel_dim = weight_shape[2 + i]
        st = strides[i]
        pad = pads[i]
        dil = dilations[i]

        if isinstance(in_dim, int) and isinstance(kernel_dim, int):
            effective_kernel = dil * (kernel_dim - 1) + 1
            out_dim = (in_dim + 2 * pad - effective_kernel) // st + 1
            if out_dim <= 0:
                raise ValueError(f"Negative or zero spatial dimension computed: {out_dim}")
            spatial_out.append(out_dim)
        else:
            spatial_out.append(S(f"spatial_out_{i}"))

    return (batch, c_out, *spatial_out)


def infer_pool(
    input_shape: ShapeType,
    kernel_size: Union[int, Sequence[int]],
    stride: Optional[Union[int, Sequence[int]]] = None,
    padding: Union[int, Sequence[int]] = 0,
    spatial_dims: int = 2,
) -> ShapeType:
    """Infer shape for 1D, 2D, or 3D pooling operations.

    Args:
        input_shape (ShapeType): Shape of input tensor (N, C, ...spatial).
        kernel_size (Union[int, Sequence[int]]): Window size.
        stride (Optional[Union[int, Sequence[int]]]): Stride size.
        padding (Union[int, Sequence[int]]): Padding size.
        spatial_dims (int): Number of spatial dimensions.

    Returns:
        ShapeType: Output tensor shape.

    Raises:
        ValueError: If input shape is invalid.
    """
    expected_rank = spatial_dims + 2
    if len(input_shape) != expected_rank:
        raise ValueError(f"Pooling expects rank {expected_rank}, got shape {input_shape}")

    batch = input_shape[0]
    channels = input_shape[1]

    kernels = [kernel_size] * spatial_dims if isinstance(kernel_size, int) else list(kernel_size)
    if stride is None:
        strides = kernels
    else:
        strides = [stride] * spatial_dims if isinstance(stride, int) else list(stride)

    pads = [padding] * spatial_dims if isinstance(padding, int) else list(padding)

    spatial_out: list[DimType] = []
    for i in range(spatial_dims):
        in_dim = input_shape[2 + i]
        k = kernels[i]
        st = strides[i]
        pad = pads[i]

        if isinstance(in_dim, int):
            out_dim = (in_dim + 2 * pad - k) // st + 1
            if out_dim <= 0:
                raise ValueError(f"Negative or zero pooled dimension: {out_dim}")
            spatial_out.append(out_dim)
        else:
            spatial_out.append(S(f"pool_out_{i}"))

    return (batch, channels, *spatial_out)


def infer_reshape(input_shape: ShapeType, target_shape: Sequence[Union[int, S]]) -> ShapeType:
    """Infer target shape from a reshape specification, resolving dynamic '-1' axes.

    Args:
        input_shape (ShapeType): Source shape.
        target_shape (Sequence[Union[int, S]]): Requested target shape.

    Returns:
        ShapeType: Resolved target shape tuple.

    Raises:
        ValueError: If shape is incompatible or multiple '-1' dimensions are specified.
    """
    target = list(target_shape)
    neg_count = target.count(-1)
    if neg_count > 1:
        raise ValueError(f"Only one dimension can be -1 in reshape, got {target}")

    has_symbolic = any(isinstance(d, S) for d in input_shape) or any(isinstance(d, S) for d in target)

    if not has_symbolic and all(isinstance(d, int) for d in input_shape) and all(isinstance(d, int) for d in target):
        total_in = math.prod(int(d) for d in input_shape)
        if neg_count == 1:
            neg_idx = target.index(-1)
            other_prod = math.prod([d for i, d in enumerate(target) if i != neg_idx])
            if other_prod == 0 or total_in % other_prod != 0:
                raise ValueError(f"Cannot reshape from {input_shape} (numel={total_in}) to {target_shape}")
            target[neg_idx] = total_in // other_prod
        else:
            total_out = math.prod(int(d) for d in target)
            if total_in != total_out:
                raise ValueError(f"Cannot reshape tensor of size {total_in} into shape {target_shape} (size {total_out})")

    return tuple(target)


def infer_transpose(input_shape: ShapeType, axes: Optional[Sequence[int]] = None) -> ShapeType:
    """Infer transposed shape from axis permutation.

    Args:
        input_shape (ShapeType): Source shape.
        axes (Optional[Sequence[int]]): Dimension permutation.

    Returns:
        ShapeType: Transposed shape tuple.

    Raises:
        ValueError: If axes are invalid for input rank.
    """
    rank = len(input_shape)
    if axes is None:
        return tuple(reversed(input_shape))

    norm_axes = [ax + rank if ax < 0 else ax for ax in axes]
    if len(norm_axes) != rank or set(norm_axes) != set(range(rank)):
        raise ValueError(f"Invalid permutation axes {axes} for rank {rank}")

    return tuple(input_shape[ax] for ax in norm_axes)


def infer_squeeze(input_shape: ShapeType, axis: Optional[Union[int, Sequence[int]]] = None) -> ShapeType:
    """Infer shape after squeezing singleton dimensions.

    Args:
        input_shape (ShapeType): Source shape.
        axis (Optional[Union[int, Sequence[int]]]): Target axis or axes to squeeze.

    Returns:
        ShapeType: Squeezed shape tuple.

    Raises:
        ValueError: If requested squeeze axis is not 1.
    """
    if axis is None:
        return tuple(d for d in input_shape if d != 1)

    rank = len(input_shape)
    axes_list = [axis] if isinstance(axis, int) else list(axis)
    norm_axes = {ax + rank if ax < 0 else ax for ax in axes_list}

    for ax in norm_axes:
        if ax < 0 or ax >= rank:
            raise ValueError(f"Squeeze axis {ax} out of range for rank {rank}")
        if input_shape[ax] != 1 and not isinstance(input_shape[ax], S):
            raise ValueError(f"Cannot squeeze axis {ax} with dimension {input_shape[ax]} != 1")

    return tuple(d for i, d in enumerate(input_shape) if i not in norm_axes)


def infer_expand_dims(input_shape: ShapeType, axis: int) -> ShapeType:
    """Infer shape after inserting a singleton dimension at specified axis.

    Args:
        input_shape (ShapeType): Source shape.
        axis (int): Axis index at which to insert dimension 1.

    Returns:
        ShapeType: Expanded shape tuple.
    """
    rank = len(input_shape)
    norm_axis = axis + rank + 1 if axis < 0 else axis
    norm_axis = max(0, min(norm_axis, rank))
    return (*input_shape[:norm_axis], 1, *input_shape[norm_axis:])


def infer_flatten(input_shape: ShapeType, start_dim: int = 0, end_dim: int = -1) -> ShapeType:
    """Infer shape after flattening dimensions between start_dim and end_dim.

    Args:
        input_shape (ShapeType): Source shape.
        start_dim (int): First dimension to flatten.
        end_dim (int): Last dimension to flatten.

    Returns:
        ShapeType: Flattened shape tuple.
    """
    rank = len(input_shape)
    if rank == 0:
        return ()

    start = start_dim + rank if start_dim < 0 else start_dim
    end = end_dim + rank if end_dim < 0 else end_dim

    start = max(0, min(start, rank - 1))
    end = max(start, min(end, rank - 1))

    prefix = input_shape[:start]
    middle = input_shape[start : end + 1]
    suffix = input_shape[end + 1 :]

    if all(isinstance(d, int) for d in middle):
        flat_dim: DimType = math.prod(int(d) for d in middle)
    else:
        flat_dim = S("flat_dim")

    return (*prefix, flat_dim, *suffix)


def infer_split(input_shape: ShapeType, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> list[ShapeType]:
    """Infer output shapes after splitting a tensor along an axis.

    Args:
        input_shape (ShapeType): Source shape.
        num_or_size_splits (Union[int, Sequence[int]]): Split count or section sizes.
        axis (int): Axis to split along.

    Returns:
        list[ShapeType]: List of output shapes.

    Raises:
        ValueError: If split sizes do not match axis dimension.
    """
    rank = len(input_shape)
    norm_axis = axis + rank if axis < 0 else axis
    dim = input_shape[norm_axis]

    if isinstance(num_or_size_splits, int):
        num = num_or_size_splits
        if isinstance(dim, int):
            if dim % num != 0:
                raise ValueError(f"Cannot evenly divide dimension {dim} into {num} splits")
            sub_dim = dim // num
        else:
            sub_dim = S(f"{dim.name}_split")
        res_shape = (*input_shape[:norm_axis], sub_dim, *input_shape[norm_axis + 1 :])
        return [res_shape for _ in range(num)]

    sizes = list(num_or_size_splits)
    if isinstance(dim, int) and sum(sizes) != dim:
        raise ValueError(f"Sum of split sizes {sum(sizes)} != dimension {dim}")

    results: list[ShapeType] = []
    for sz in sizes:
        results.append((*input_shape[:norm_axis], sz, *input_shape[norm_axis + 1 :]))
    return results


def infer_concat(shapes: Sequence[ShapeType], axis: int = 0) -> ShapeType:
    """Infer shape from concatenating multiple tensors along an axis.

    Args:
        shapes (Sequence[ShapeType]): Shapes to concatenate.
        axis (int): Concatenation axis.

    Returns:
        ShapeType: Concatenated shape tuple.

    Raises:
        ValueError: If ranks mismatch or non-concatenation dimensions differ.
    """
    if not shapes:
        return ()
    first = shapes[0]
    rank = len(first)
    norm_axis = axis + rank if axis < 0 else axis

    total_dim = 0
    has_symbolic = False

    for s in shapes:
        if len(s) != rank:
            raise ValueError(f"Concatenation rank mismatch: {len(s)} vs {rank}")
        for i in range(rank):
            if i != norm_axis and s[i] != first[i]:
                raise ValueError(f"Dimension mismatch along non-concat axis {i}: {s[i]} vs {first[i]}")
        d = s[norm_axis]
        if isinstance(d, int):
            total_dim += d
        else:
            has_symbolic = True

    final_axis_dim: DimType = S("concat_dim") if has_symbolic else total_dim
    return (*first[:norm_axis], final_axis_dim, *first[norm_axis + 1 :])


def infer_stack(shapes: Sequence[ShapeType], axis: int = 0) -> ShapeType:
    """Infer shape from stacking multiple tensors along a new axis.

    Args:
        shapes (Sequence[ShapeType]): Shapes to stack.
        axis (int): Stacking axis.

    Returns:
        ShapeType: Stacked shape tuple.

    Raises:
        ValueError: If input shapes are not identical.
    """
    if not shapes:
        return ()
    first = shapes[0]
    for s in shapes:
        if s != first:
            raise ValueError(f"Stack requires identical shapes, got {s} and {first}")

    rank = len(first)
    norm_axis = axis + rank + 1 if axis < 0 else axis
    norm_axis = max(0, min(norm_axis, rank))
    return (*first[:norm_axis], len(shapes), *first[norm_axis:])


def infer_reduction(input_shape: ShapeType, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> ShapeType:
    """Infer shape resulting from a reduction operation.

    Args:
        input_shape (ShapeType): Source shape.
        axis (Optional[Union[int, Sequence[int]]]): Reduced axis or axes.
        keepdims (bool): Whether to preserve reduced dimensions as size 1.

    Returns:
        ShapeType: Reduced output shape.
    """
    if axis is None:
        return (1,) * len(input_shape) if keepdims else ()

    rank = len(input_shape)
    axes_list = [axis] if isinstance(axis, int) else list(axis)
    norm_axes = {ax + rank if ax < 0 else ax for ax in axes_list}

    if keepdims:
        return tuple(1 if i in norm_axes else d for i, d in enumerate(input_shape))
    return tuple(d for i, d in enumerate(input_shape) if i not in norm_axes)


def _dispatch_elementwise_and_reduction(
    category: str,
    valid_shapes: list[ShapeType],
    kwargs: dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]],
) -> Optional[ShapeType]:
    """Dispatch elementwise and reduction operations.

    Args:
        category (str): Op category.
        valid_shapes (list[ShapeType]): Input shapes.
        kwargs (dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]]): Op attributes.

    Returns:
        Optional[ShapeType]: Inferred shape or None.
    """
    if category == "elementwise":
        return broadcast_shapes(*valid_shapes) if valid_shapes else ()
    if category == "reduction":
        if not valid_shapes:
            return ()
        axis = kwargs.get("axis", kwargs.get("axes"))
        keepdims = bool(kwargs.get("keepdims", False))
        return infer_reduction(valid_shapes[0], axis=axis, keepdims=keepdims)
    return None


def _dispatch_view_ops(
    category: str,
    valid_shapes: list[ShapeType],
    kwargs: dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]],
) -> Optional[ShapeType]:
    """Dispatch view transformations (reshape, transpose, squeeze, expand_dims).

    Args:
        category (str): Op category.
        valid_shapes (list[ShapeType]): Input shapes.
        kwargs (dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]]): Op attributes.

    Returns:
        Optional[ShapeType]: Inferred shape or None.
    """
    if not valid_shapes:
        return () if category in ("reshape", "transpose", "squeeze", "expand_dims") else None

    if category == "reshape":
        target = kwargs.get("shape", kwargs.get("newshape", kwargs.get("target_shape")))
        return infer_reshape(valid_shapes[0], target) if isinstance(target, (list, tuple)) else valid_shapes[0]
    if category == "transpose":
        axes = kwargs.get("axes", kwargs.get("permutation"))
        return infer_transpose(valid_shapes[0], axes if isinstance(axes, (list, tuple)) else None)
    if category == "squeeze":
        axis = kwargs.get("axis")
        return infer_squeeze(valid_shapes[0], axis=axis if isinstance(axis, (int, list, tuple)) else None)
    if category == "expand_dims":
        axis = kwargs.get("axis", 0)
        return infer_expand_dims(valid_shapes[0], axis=int(axis) if isinstance(axis, (int, float)) else 0)
    return None


def _dispatch_group_ops(
    category: str,
    valid_shapes: list[ShapeType],
    kwargs: dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]],
) -> Optional[ShapeType]:
    """Dispatch tensor grouping operations (flatten, split, concat, stack).

    Args:
        category (str): Op category.
        valid_shapes (list[ShapeType]): Input shapes.
        kwargs (dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]]): Op attributes.

    Returns:
        Optional[ShapeType]: Inferred shape or None.
    """
    if not valid_shapes:
        return () if category in ("flatten", "split") else None

    if category == "flatten":
        return infer_flatten(
            valid_shapes[0],
            start_dim=int(kwargs.get("start_dim", 0)),
            end_dim=int(kwargs.get("end_dim", -1)),
        )
    if category == "split":
        splits = kwargs.get("num_or_size_splits", kwargs.get("chunks", 2))
        res = infer_split(valid_shapes[0], splits, axis=int(kwargs.get("axis", 0)))
        return res[0] if res else ()
    if category == "concat":
        return infer_concat(valid_shapes, axis=int(kwargs.get("axis", 0)))
    if category == "stack":
        return infer_stack(valid_shapes, axis=int(kwargs.get("axis", 0)))
    return None


def _dispatch_contracting(
    op_type: str,
    category: str,
    valid_shapes: list[ShapeType],
) -> Optional[ShapeType]:
    """Dispatch contracting operations (e.g. matrix multiplication).

    Args:
        op_type (str): Op name.
        category (str): Op category.
        valid_shapes (list[ShapeType]): Input shapes.

    Returns:
        Optional[ShapeType]: Inferred shape or None.

    Raises:
        ValueError: If fewer than 1 operand is provided.
    """
    if category != "contracting":
        return None
    if len(valid_shapes) >= 2:
        return infer_matmul(valid_shapes[0], valid_shapes[1])
    if len(valid_shapes) == 1:
        return valid_shapes[0]
    raise ValueError(f"Contracting op '{op_type}' requires 2 operands, got {len(valid_shapes)}")


def _dispatch_spatial_and_pooling(
    op_type: str,
    category: str,
    valid_shapes: list[ShapeType],
    kwargs: dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]],
) -> Optional[ShapeType]:
    """Dispatch spatial convolutions and pooling operations.

    Args:
        op_type (str): Op name.
        category (str): Op category.
        valid_shapes (list[ShapeType]): Input shapes.
        kwargs (dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]]): Op attributes.

    Returns:
        Optional[ShapeType]: Inferred shape or None.

    Raises:
        ValueError: On shape constraint violations.
    """
    if category == "spatial":
        if len(valid_shapes) >= 2:
            if "1d" in op_type.lower():
                s_dims = 1
            elif "3d" in op_type.lower():
                s_dims = 3
            else:
                s_dims = 2
            return infer_conv(
                valid_shapes[0],
                valid_shapes[1],
                stride=kwargs.get("stride", kwargs.get("strides", 1)),
                padding=kwargs.get("padding", 0),
                dilation=kwargs.get("dilation", kwargs.get("dilations", 1)),
                spatial_dims=s_dims,
            )
        raise ValueError(f"Spatial op '{op_type}' requires input and weight shapes")
    if category == "pooling":
        if valid_shapes:
            if "1d" in op_type.lower():
                s_dims = 1
            elif "3d" in op_type.lower():
                s_dims = 3
            else:
                s_dims = 2
            return infer_pool(
                valid_shapes[0],
                kernel_size=kwargs.get("kernel_size", kwargs.get("window_size", 2)),
                stride=kwargs.get("stride", kwargs.get("strides")),
                padding=kwargs.get("padding", 0),
                spatial_dims=s_dims,
            )
        raise ValueError(f"Pooling op '{op_type}' requires an input shape")
    return None


def infer_op_shape_declarative(
    op_type: str,
    *shapes: ShapeType,
    **kwargs: Union[int, float, str, bool, ShapeType, Sequence[int], None],
) -> ShapeType:
    """Evaluate shape inference for an operation declaratively against shape_signatures.yaml.

    Args:
        op_type (str): Operation identifier.
        *shapes (ShapeType): Input tensor shapes.
        **kwargs (Union[int, float, str, bool, ShapeType, Sequence[int], None]): Op attributes.

    Returns:
        ShapeType: Resulting shape.

    Raises:
        ValueError: If op shape inference fails or shapes violate declarative constraints.
    """
    cfg = load_shape_signatures()
    op_sig = cfg.operations.get(op_type)
    category = op_sig.category if op_sig else "elementwise"

    valid_shapes = [s for s in shapes if s is not None]

    res = _dispatch_elementwise_and_reduction(category, valid_shapes, kwargs)
    if res is not None:
        return res

    res = _dispatch_view_ops(category, valid_shapes, kwargs)
    if res is not None:
        return res

    res = _dispatch_group_ops(category, valid_shapes, kwargs)
    if res is not None:
        return res

    res = _dispatch_contracting(op_type, category, valid_shapes)
    if res is not None:
        return res

    res = _dispatch_spatial_and_pooling(op_type, category, valid_shapes, kwargs)
    if res is not None:
        return res

    return broadcast_shapes(*valid_shapes) if valid_shapes else ()


def _normalize_inputs_to_shapes(
    args: tuple[Union[ShapeType, Sequence[int], None], ...],
    kwargs: dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]],
) -> list[ShapeType]:
    """Normalize positional and keyword inputs into concrete shape tuples.

    Args:
        args (tuple[Union[ShapeType, Sequence[int], None], ...]): Positional arguments.
        kwargs (dict[str, Union[int, float, str, bool, ShapeType, Sequence[int], None]]): Keyword arguments.

    Returns:
        list[ShapeType]: Extracted shape tuples.
    """
    inputs = kwargs.get("inputs")
    if inputs is None:
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            if any(hasattr(x, "shape") or hasattr(x, "shape_metadata") or isinstance(x, (list, tuple)) for x in args[0]) or not all(isinstance(x, (int, S)) for x in args[0]):
                inputs = list(args[0])
            else:
                inputs = list(args)
        else:
            inputs = list(args)

    shapes: list[ShapeType] = []
    if isinstance(inputs, (list, tuple)):
        for item in inputs:
            if hasattr(item, "shape_metadata") and item.shape_metadata is not None:
                shapes.append(tuple(item.shape_metadata))
            elif hasattr(item, "shape") and item.shape is not None:
                shapes.append(tuple(item.shape))
            elif isinstance(item, (list, tuple)):
                shapes.append(tuple(item))
    return shapes


def infer_shape(
    op_type: str,
    *args: Union[ShapeType, Sequence[int], None],
    **kwargs: Union[int, float, str, bool, ShapeType, Sequence[int], None],
) -> ShapeType:
    """Infer output shape for an operation from positional and keyword inputs.

    Args:
        op_type (str): Operation identifier.
        *args (Union[ShapeType, Sequence[int], None]): Positional inputs/shapes.
        **kwargs (Union[int, float, str, bool, ShapeType, Sequence[int], None]): Keyword arguments.

    Returns:
        ShapeType: Inferred shape tuple.
    """
    if op_type in _SHAPE_INFERENCE_REGISTRY:
        return _SHAPE_INFERENCE_REGISTRY[op_type](*args, **kwargs)

    shapes = _normalize_inputs_to_shapes(args, kwargs)
    cfg = load_shape_signatures()
    if op_type in cfg.operations:
        return infer_op_shape_declarative(op_type, *shapes, **kwargs)

    try:
        from ml_switcheroo_compiler.ops.registry import get_op

        op_cls = get_op(op_type)
        if op_cls is not None:
            if not hasattr(op_cls, "infer_shape"):
                return ()
            if not hasattr(op_cls, "_yaml_data"):
                instance = op_cls()
                try:
                    return instance.infer_shape(*shapes, **kwargs)
                except Exception:
                    return instance.infer_shape(*args, **kwargs)
    except Exception:
        pass

    return infer_op_shape_declarative(op_type, *shapes, **kwargs)
