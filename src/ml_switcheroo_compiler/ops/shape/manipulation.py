# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103, PLR0913
"""Shape operations for Tensor objects."""

from __future__ import annotations
# pylint: disable=duplicate-code


from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import dispatch_eager
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


if TYPE_CHECKING:
    from collections.abc import Sequence


@dispatch_eager("Reshape")
def reshape(input: Tensor, shape: Sequence[int] | None = None) -> Tensor:
    """Reshapes the input tensor to the specified shape.

    Args:
        input (Tensor): The input tensor to reshape
        shape (Sequence[int] | None): The target shape

    Returns:
    Tensor: A reshaped tensor with the specified shape
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = tuple(shape)
    return _emit_shape_node(
        "Reshape",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Flatten")
def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flattens the input tensor into a 1D tensor.

    Args:
        input (Tensor): The input tensor to flatten
        start_dim (int): The first dimension to flatten. Defaults to 0
        end_dim (int): The last dimension to flatten. Defaults to -1

    Returns:
    Tensor: A flattened 1D tensor
    """
    import math

    shape = list(input.shape) if input.shape is not None else []
    if not shape:
        return reshape(input, [-1])
    s_dim = start_dim if start_dim >= 0 else start_dim + len(shape)
    e_dim = end_dim if end_dim >= 0 else end_dim + len(shape)

    new_shape = shape[:s_dim] + [math.prod(shape[s_dim : e_dim + 1])] + shape[e_dim + 1 :]
    return reshape(input, new_shape)


def unflatten(input: Tensor, dim: int, sizes: tuple[int, ...]) -> Tensor:
    """Unflattens a dimension of the input tensor into multiple dimensions.

    Args:
        input (Tensor): The input tensor
        dim (int): The dimension to unflatten
        sizes (tuple[int, ...]): The new shape of the unflattened dimension

    Returns:
    Tensor: The unflattened tensor
    """
    shape = list(input.shape) if input.shape is not None else []
    if not shape:
        return input  # pragma: no cover
    dim = dim if dim >= 0 else dim + len(shape)

    new_shape = shape[:dim] + list(sizes) + shape[dim + 1 :]
    return reshape(input, new_shape)  # pragma: no cover


def view(input: Tensor, shape: tuple[int, ...] | list[int]) -> Tensor:
    """Returns a new tensor with the same data as the input tensor but of a different shape.

    For compiler representation, view is treated identically to reshape.

    Args:
        input (Tensor): The input tensor
        shape (tuple[int, ...] | list[int]): The desired shape

    Returns:
    Tensor: A reshaped tensor
    """
    return reshape(input, list(shape))


def _normalize_dims(dim: int | tuple[int, ...] | list[int] | None) -> list[int] | None:
    """Function docstring.

    Args:
        dim: Arg.
    """
    if dim is None:
        return None
    return [dim] if isinstance(dim, int) else list(dim)


def _compute_squeeze_shape(shape: tuple, dim: int | tuple[int, ...] | list[int] | None) -> tuple:
    """Function docstring.

    Args:
        shape: Arg.
        dim: Arg.
    """
    dims = _normalize_dims(dim)
    if dims is None:
        return tuple(s for s in shape if s != 1)
    return tuple(s for i, s in enumerate(shape) if i not in dims or s != 1)


@dispatch_eager("Squeeze")
def squeeze(
    input: Tensor, dim: int | Sequence[int] | None = None, axis: int | Sequence[int] | None = None
) -> Tensor:
    """Removes dimensions of size 1 from the input tensor.

    Args:
        input (Tensor): The input tensor
        dim (int | Sequence[int] | None): The dimension(s) to squeeze
        axis (int | Sequence[int] | None): Alias for dim
        If None, all dimensions of size 1 are removed. Defaults to None

    Returns:
    Tensor: The squeezed tensor
    """
    if axis is not None:
        dim = axis
    inputs = [input]
    out_shape = _compute_squeeze_shape(input.shape, dim)
    kwargs = {"dim": dim} if dim is not None else {}
    return _emit_shape_node("Squeeze", inputs, kwargs, out_shape, input.dtype)


@dispatch_eager("ExpandDims")
def expand_dims(a: Tensor, axis: int | Sequence[int]) -> Tensor:
    """Expands the shape of an array.

    Args:
        a (Tensor): The input tensor
        axis (Union[int, Sequence[int]]): The position(s) in the expanded axes where the new axis
        (or axes) is placed.

    Returns:
    Tensor: The expanded tensor
    """
    # Calculate new shape
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    out_shape = list(a.shape)

    # Sort axes to handle insertions correctly
    axes_sorted = sorted([ax if ax >= 0 else ax + len(out_shape) + 1 for ax in axes])
    for ax in axes_sorted:
        out_shape.insert(ax, 1)

    out_shape_tup = tuple(out_shape)

    return reshape(a, out_shape_tup)


def unsqueeze(input: Tensor, dim: int) -> Tensor:
    """Inserts a dimension of size 1 at the specified position.

    Args:
        input (Tensor): The input input tensor.
        dim (int): The dim parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    return expand_dims(input, dim)


@dispatch_eager("BroadcastTo")
def expand(input: Tensor, size: Sequence[int]) -> Tensor:
    """Expands the tensor to a new shape by broadcasting singleton dimensions.

    Args:
        input (Tensor): The input tensor
        size (Sequence[int]): The desired expanded shape

    Returns:
    Tensor: The expanded tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Expand",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def _extract_tolist(size: object) -> object:
    """Function docstring.

    Args:
        size: Arg.
    """
    if (
        hasattr(size, "data") and hasattr(size.data, "tolist") and callable(size.data.tolist)
    ):  # pragma: no branch
        return size.data.tolist()  # pragma: no cover
    if hasattr(size, "tolist") and callable(size.tolist):  # pragma: no branch
        return size.tolist()  # pragma: no cover
    return size


def _try_extract_tolist(size_list: list[object]) -> list[object] | None:
    """Function docstring.

    Args:
        size_list: Arg.
    """
    try:
        return [
            int(s.data.tolist() if not isinstance(s.data.tolist(), list) else s.data.tolist()[0])
            for s in size_list
        ]
    except TypeError:
        return None


def _try_extract_item(size_list: list[object]) -> list[object] | None:
    """Function docstring.

    Args:
        size_list: Arg.
    """
    try:
        return [int(s.data.item()) for s in size_list]
    except (TypeError, ValueError, AttributeError):
        return None


def _process_data_list(size_list: list[object], first_data: object) -> list[object]:
    """Function docstring.

    Args:
        size_list: Arg.
        first_data: Arg.
    """
    if hasattr(first_data, "tolist") and callable(first_data.tolist):  # pragma: no cover
        res = _try_extract_tolist(size_list)  # pragma: no cover
        if res is not None:  # pragma: no cover
            return res  # pragma: no cover
    if hasattr(first_data, "item") and callable(first_data.item):  # pragma: no cover
        res = _try_extract_item(size_list)  # pragma: no cover
        if res is not None:  # pragma: no cover
            return res  # pragma: no cover
    return size_list  # pragma: no cover


def _extract_from_list(size_list: list[object]) -> list[object]:
    """Function docstring.

    Args:
        size_list: Arg.
    """
    if not size_list:  # pragma: no branch
        return size_list  # pragma: no cover
    if not hasattr(size_list[0], "data"):  # pragma: no branch
        return size_list
    return _process_data_list(size_list, size_list[0].data)  # pragma: no cover


def _parse_shape_arg(size: object) -> tuple[int, ...] | None:
    """Parses various types of size arguments into a shape tuple.

    Handles scalar tensors, shape tuples, lists of tensors, etc.
    """
    if size is None:
        return None

    size = _extract_tolist(size)

    if isinstance(size, list):
        size = _extract_from_list(size)

    if isinstance(size, (list, tuple)):
        new_size = []
        for s in size:
            if isinstance(s, (int, float, bool)):
                new_size.append(int(s))
            elif hasattr(s, "item") and callable(s.item):
                try:
                    new_size.append(int(s.item()))
                except Exception:
                    new_size.append(s)
            elif hasattr(s, "data") and hasattr(s.data, "item"):
                try:
                    new_size.append(int(s.data.item()))
                except Exception:
                    new_size.append(s)
            else:
                new_size.append(s)
        size = new_size

    return tuple(size)


@dispatch_eager("BroadcastTo")
def broadcast_to(input: Tensor, shape: Sequence[int] = None, **kwargs: object) -> Tensor:
    """Broadcasts the input tensor to a new shape.

    Args:
        input (Tensor): The input tensor
        shape (Sequence[int]): The target shape to broadcast to
        **kwargs (object): Additional keyword arguments.

    Returns:
    Tensor: The broadcasted tensor
    """
    size = shape if shape is not None else kwargs.get("size")
    out_shape = _parse_shape_arg(size)

    inputs = [input]
    return _emit_shape_node(
        "BroadcastTo",
        inputs,
        {},
        out_shape if out_shape is not None else (),
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Swapaxes")
def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Swaps two dimensions of the input tensor.

    Args:
        input (Tensor): The input tensor
        dim0 (int): The first dimension to swap
        dim1 (int): The second dimension to swap

    Returns:
    Tensor: The transposed tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = list(inputs[0].shape)
    if len(out_shape) > max(dim0, dim1):
        out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
    out_shape = tuple(out_shape)

    axes = list(range(len(inputs[0].shape)))
    if len(axes) > max(dim0, dim1):
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]

    return _emit_shape_node(
        "Transpose",
        inputs,
        {"axes": tuple(axes)},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Transpose")
def permute(input: Tensor, dims: Sequence[int]) -> Tensor:
    """Permutes the dimensions of the input tensor according to the specified order.

    Args:
        input (Tensor): The input tensor
        dims (Sequence[int]): The desired ordering of dimensions

    Returns:
    Tensor: The permuted tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = tuple(input.shape[d] for d in dims)
    return _emit_shape_node(
        "Transpose",
        inputs,
        {"axes": tuple(dims)},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Swapaxes")
def swapaxes(input: Tensor, axis1: int, axis2: int) -> Tensor:
    """Swap axes."""
    inputs = [input]
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Swapaxes",
        inputs,
        {"axis1": axis1, "axis2": axis2},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Moveaxis")
def moveaxis(
    input: Tensor,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Tensor:
    """Moves axes of the input tensor from source positions to destination positions.

    Args:
        input (Tensor): The input tensor
        source (int | Sequence[int]): Original positions of the axes to move
        destination (int | Sequence[int]): Target positions for the axes

    Returns:
    Tensor: The tensor with moved axes
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Moveaxis",
        inputs,
        {"source": source, "destination": destination},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("Roll")
def roll(
    input: Tensor,
    shifts: int | Sequence[int],
    dims: int | Sequence[int] | None = None,
) -> Tensor:
    """Rolls the tensor along the specified dimensions.

    Args:
        input (Tensor): The input tensor
        shifts (int | Sequence[int]): The number of places by which elements are
        shifted
        dims (int | Sequence[int] | None): The dimensions along which to roll. Defaults
        to None

    Returns:
    Tensor: The rolled tensor
    """
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Roll",
        inputs,
        {"shift": shifts, "axis": dims},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


@dispatch_eager("BroadcastInDim")
def broadcast_in_dim(
    operand: Tensor,
    shape: Sequence[int],
    broadcast_dimensions: Sequence[int],
) -> Tensor:
    """Broadcasts an array to a target shape by matching specified dimensions.

    Args:
        operand (Tensor): The input tensor
        shape (Sequence[int]): The target shape
        broadcast_dimensions (Sequence[int]): The dimensions of the target shape to match

    Returns:
    Tensor: The broadcasted tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    inputs = [operand]
    attributes = {"shape": shape, "broadcast_dimensions": broadcast_dimensions}
    return _emit_shape_node("BroadcastInDim", inputs, attributes, tuple(shape), operand.dtype)


@dispatch_eager("Reverse")
def reverse(input: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Reverse dimensions."""
    inputs = [input]
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Reverse",
        inputs,
        {"dims": dims},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def atleast_1d(*arys: object) -> object:
    """Convert inputs to arrays with at least one dimension.

    Args:
        *arys (object): One or more input arrays.

    Returns:
        object: An array, or list of arrays, each with a.ndim >= 1.
    """

    def _gen() -> object:  # type: ignore
        for a in arys:
            t = a  # pragma: no cover
            if len(t.shape) == 0:  # pragma: no cover
                yield reshape(t, (1,))  # pragma: no cover
                continue  # pragma: no cover
            yield t  # pragma: no cover

    res = list(_gen())
    if len(res) == 1:  # pragma: no cover
        return res[0]  # pragma: no cover
    return res  # pragma: no cover


def atleast_2d(*arys: object) -> object:
    """Convert inputs to arrays with at least two dimensions.

    Args:
        *arys (object): One or more input arrays.

    Returns:
        object: An array, or list of arrays, each with a.ndim >= MAGIC_VAL_2.
    """

    def _gen() -> object:  # type: ignore
        for a in arys:
            t = a  # pragma: no cover
            if len(t.shape) == 0:  # pragma: no cover
                yield reshape(t, (1, 1))  # pragma: no cover
                continue  # pragma: no cover
            if len(t.shape) == 1:  # pragma: no cover
                yield reshape(t, (1, t.shape[0]))  # pragma: no cover
                continue  # pragma: no cover
            yield t  # pragma: no cover

    res = list(_gen())
    if len(res) == 1:  # pragma: no cover
        return res[0]  # pragma: no cover
    return res  # pragma: no cover


def atleast_3d(*arys: object) -> object:
    """Convert inputs to arrays with at least three dimensions.

    Args:
        *arys (object): One or more input arrays.

    Returns:
        object: An array, or list of arrays, each with a.ndim >= MAGIC_VAL_3.
    """

    def _gen() -> object:  # type: ignore
        for a in arys:
            t = a  # pragma: no cover
            if len(t.shape) == 0:  # pragma: no cover
                yield reshape(t, (1, 1, 1))  # pragma: no cover
                continue  # pragma: no cover
            if len(t.shape) == 1:  # pragma: no cover
                yield reshape(t, (1, t.shape[0], 1))  # pragma: no cover
                continue  # pragma: no cover
            if len(t.shape) == MAGIC_VAL_2:  # pragma: no cover
                yield reshape(t, (t.shape[0], t.shape[1], 1))  # pragma: no cover
                continue  # pragma: no cover
            yield t  # pragma: no cover

    res = list(_gen())
    if len(res) == 1:  # pragma: no cover
        return res[0]  # pragma: no cover
    return res  # pragma: no cover


def broadcast_arrays(*args: object, **kwargs: object) -> object:
    """Broadcast any number of arrays against each other.

    Args:
        *args (object): The arrays to broadcast.
        **kwargs: Keyword arguments.

    Returns:
        object: A list of broadcasted arrays.
    """
    from ml_switcheroo_compiler.core.shape import broadcast_shapes

    tensors = [a for a in args]  # pragma: no cover
    b_shape = tensors[0].shape  # pragma: no cover
    for t in tensors[1:]:  # pragma: no cover
        b_shape = broadcast_shapes(b_shape, t.shape)  # pragma: no cover
    return [broadcast_to(t, b_shape) for t in tensors]  # pragma: no cover


def depth_to_space(input, block_size, data_format="NHWC", name=None):  # pragma: no cover
    # pragma: no cover
    """DepthToSpace for tensors."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig(input.shape, "float32", "cpu"))


def space_to_depth(input, block_size, data_format="NHWC", name=None):  # pragma: no cover
    # pragma: no cover
    """SpaceToDepth for tensors."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig(input.shape, "float32", "cpu"))


def space_to_batch(input, paddings, block_size=None, name=None):  # pragma: no cover
    # pragma: no cover
    """SpaceToBatch for 4-D tensors of shape [batch, height, width, depth]."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig(input.shape, "float32", "cpu"))


def with_space_to_batch(
    input, dilation_rate, padding, op, filter_shape=None, spatial_dims=None, data_format=None
):
    """Performs `op` on the space-to-batch representation of `input`."""
    return op(input)
