"""Shape operations for Tensor objects."""

from __future__ import annotations
# pylint: disable=duplicate-code


from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import dispatch_eager
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if TYPE_CHECKING:
    from collections.abc import Sequence


def tile(input: Tensor, reps: Sequence[int]) -> Tensor:
    """Constructs a new tensor by repeating the input tensor the specified number of times.

    Args:
        input (Tensor): The input tensor
        reps (Sequence[int]): The number of repetitions along each dimension

    Returns:
    Tensor: The tiled tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tile", input.data, reps)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Tile",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def repeat(
    input: Tensor,
    repeats: int | Sequence[int],
    dim: int | None = None,
) -> Tensor:
    """Repeats elements of the input tensor along a specified dimension.

    Args:
        input (Tensor): The input tensor
        repeats (int | Sequence[int]): The number of repetitions for each element
        dim (int | None): The dimension along which to repeat. Defaults to None

    Returns:
    Tensor: The tensor with repeated elements
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Repeat", input.data, repeats, axis=dim)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Repeat",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    """Returns the upper triangular part of a matrix or batch of matrices.

    Args:
        input (Tensor): The input tensor
        diagonal (int): The diagonal to consider. 0 is the main diagonal, positive
        values are above, and negative values are below. Defaults to 0

    Returns:
    Tensor: The upper triangular tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Triu", input.data, k=diagonal)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Triu",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    """Returns the lower triangular part of a matrix or batch of matrices.

    Args:
        input (Tensor): The input tensor
        diagonal (int): The diagonal to consider. 0 is the main diagonal, positive
        values are above, and negative values are below. Defaults to 0

    Returns:
    Tensor: The lower triangular tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Tril", input.data, k=diagonal)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Tril",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def meshgrid(*tensors: Tensor, indexing: str = "ij") -> Sequence[Tensor]:
    """Creates coordinate grids from coordinate vectors.

    Args:
        *tensors (Tensor): Coordinate vectors
        indexing (str): The indexing mode, either "ij" (matrix) or "xy" (Cartesian)
        Defaults to "ij"

    Returns:
    Sequence[Tensor]: A sequence of coordinate grid tensors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        datas = backend.execute_op("Meshgrid", *[t.data for t in tensors], indexing=indexing)
        return tuple(
            Tensor(d, TensorConfig(d.shape, tensors[0].dtype, tensors[0].device)) for d in datas
        )
    inputs = list(tensors)
    # shape calculation placeholder
    out_shape = inputs[0].shape if len(inputs) > 0 else ()
    return tuple(
        _emit_shape_node(
            "Meshgrid",
            inputs,
            {},
            out_shape,
            inputs[0].dtype if len(inputs) > 0 else DType.Float32,
        )
        for _ in inputs
    )


def pad(
    array: object,
    pad_width: object,
    mode: str = "constant",
    **kwargs: object,
) -> object:
    """Pads an array with specified widths and values.

    Args:
        array (object): The array to pad
        pad_width (object): Number of values padded to the edges of each axis
        mode (str): The padding mode (e.g., 'constant'). Defaults to "constant"
        **kwargs (object): Additional keyword arguments for the padding mode

    Returns:
    object: The padded array
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op("Pad", array, pad_width, mode=mode, **kwargs)


@dispatch_eager("TopK")
def top_k(operand: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Returns the top k values and their indices along the last dimension.

    Args:
        operand (Tensor): The input tensor
        k (int): Number of top elements to look for

    Returns:
    tuple[Tensor, Tensor]: Top k values and their indices

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    out_shape = list(operand.shape) if operand.shape else []
    if out_shape:
        out_shape[-1] = k
    out_shape = tuple(out_shape)

    inputs = [operand]
    # We cheat a bit by returning two tensors pointing to the same node for now,
    # as handling multi-output nodes properly requires more IR scaffolding
    val_node = _emit_shape_node("TopK", inputs, {"k": k}, out_shape, operand.dtype)
    idx_node = _emit_shape_node("TopK", inputs, {"k": k}, out_shape, DType.Int32)
    return val_node, idx_node


def argsort(
    operand: Tensor,
    dimension: int = -1,
    is_stable: bool = True,
    axis: int | None = None,
) -> Tensor:
    """Returns the indices that would sort an array along a given dimension.

    Args:
        operand (Tensor): The input tensor
        dimension (int): The dimension to sort along
        is_stable (bool): Whether to use a stable sorting algorithm
        axis (int): Alias for dimension.

    Returns:
    Tensor: The indices that sort the tensor
    """
    if axis is not None:
        dimension = axis

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        kind = "stable" if is_stable else "quicksort"
        data = backend.execute_op("ArgSort", operand.data, axis=dimension, kind=kind)
        from ml_switcheroo_compiler.core.dtype import DType

        return Tensor(data, TensorConfig(operand.shape, DType.Int32, operand.device))

    inputs = [operand]
    attributes = {"dimension": dimension, "is_stable": is_stable}
    from ml_switcheroo_compiler.core.dtype import DType

    return _emit_shape_node("ArgSort", inputs, attributes, operand.shape, DType.Int32)


def sort(
    operand: Tensor,
    dimension: int = -1,
    is_stable: bool = True,
    axis: int | None = None,
) -> Tensor:
    """Sorts the elements of an array along a given dimension.

    Args:
        operand (Tensor): The input tensor
        dimension (int): The dimension to sort along
        is_stable (bool): Whether to use a stable sorting algorithm
        axis (int): Alias for dimension.

    Returns:
    Tensor: The sorted tensor
    """
    if axis is not None:
        dimension = axis

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        kind = "stable" if is_stable else "quicksort"
        data = backend.execute_op("Sort", operand.data, axis=dimension, kind=kind)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, operand.dtype, operand.device),
        )

    inputs = [operand]
    attributes = {"dimension": dimension, "is_stable": is_stable}
    return _emit_shape_node("Sort", inputs, attributes, operand.shape, operand.dtype)


@dispatch_eager("Resize")
def image_resize(image: Tensor, shape: tuple[int, int], method: str = "bilinear") -> Tensor:
    """Resizes an image to the given target shape using interpolation.

    Args:
        image (Tensor): The input image tensor
        shape (tuple[int, int]): The target height and width
        method (str): The interpolation method (e.g. 'bilinear', 'nearest')

    Returns:
    Tensor: The resized image tensor
    """
    from ml_switcheroo_compiler.ops.shape.reshape import Resize

    op = Resize()
    out_shape = op.infer_shape(image, shape, method)

    return _emit_shape_node(
        "Resize",
        [image],
        {"shape": shape, "method": method},
        out_shape,
        image.dtype,
    )
