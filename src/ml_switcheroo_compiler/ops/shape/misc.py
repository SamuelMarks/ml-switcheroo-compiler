"""Shape operations for Tensor objects."""

from __future__ import annotations

from collections.abc import Sequence

# pylint: disable=duplicate-code
from typing import TYPE_CHECKING

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.reshape import Resize
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if TYPE_CHECKING:
    pass


def tile(input: Tensor, reps: Sequence[int]) -> Tensor:
    """Constructs a new tensor by repeating the input tensor the specified number of times.

    Args:
        input (Tensor): The input tensor
        reps (Sequence[int]): The number of repetitions along each dimension

    Returns:
    Tensor: The tiled tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Tile", (input.data if type(input).__name__ == "Tensor" else input), reps)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Tile",
        inputs,
        {"reps": reps},
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
        backend = get_active_backend()
        data = backend.execute_op("Repeat", (input.data if type(input).__name__ == "Tensor" else input), repeats, axis=dim)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs = [input]
    # shape calculation placeholder
    out_shape = inputs[0].shape
    return _emit_shape_node(
        "Repeat",
        inputs,
        {"repeats": repeats, "axis": dim},
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
        backend = get_active_backend()
        data = backend.execute_op("Triu", (input.data if type(input).__name__ == "Tensor" else input), k=diagonal)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
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
        backend = get_active_backend()
        data = backend.execute_op("Tril", (input.data if type(input).__name__ == "Tensor" else input), k=diagonal)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
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


def _compute_meshgrid_shape(inputs: list[Tensor], indexing: str) -> tuple[int, ...]:
    """Computes the shape for a meshgrid."""
    if not inputs:
        return ()
    out_shape = tuple(t.shape[0] if t.shape else 1 for t in inputs)
    if indexing == "xy" and len(inputs) >= 2:
        return (out_shape[1], out_shape[0]) + out_shape[2:]
    return out_shape


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
        backend = get_active_backend()
        datas = backend.execute_op("Meshgrid", *[(t.data if type(t).__name__ == "Tensor" else t) for t in tensors], indexing=indexing)
        return tuple(Tensor(d, TensorConfig(d.shape, tensors[0].dtype, tensors[0].device)) for d in datas)
    inputs = list(tensors)
    out_shape = _compute_meshgrid_shape(inputs, indexing)
    dtype = inputs[0].dtype if inputs else DType.Float32

    return tuple(_emit_shape_node("Meshgrid", inputs, {"indexing": indexing}, out_shape, dtype) for _ in inputs)


def _normalize_pad_width(pad_width: object, ndim: int) -> tuple:
    if isinstance(pad_width, int):
        return ((pad_width, pad_width),) * ndim
    if isinstance(pad_width, tuple) and len(pad_width) == 2 and isinstance(pad_width[0], int):
        return (pad_width,) * ndim
    return tuple(pad_width) if isinstance(pad_width, (list, tuple)) else ()


def _compute_pad_dim(dim: int, pw: object) -> int:
    if isinstance(pw, int):
        return dim + pw * 2
    if isinstance(pw, tuple) and len(pw) == 2:
        return dim + pw[0] + pw[1]
    return dim


@register_op("Pad")
class Pad(OpDef):
    """Pad op."""

    op_name = "Pad"

    def infer_shape(self, array: object, pad_width: object, mode: str = "constant", **kwargs: object) -> tuple[int, ...]:
        """Infer shape for pad."""
        in_shape = getattr(array, "shape", ())
        if not in_shape:
            return ()

        normalized_pw = _normalize_pad_width(pad_width, len(in_shape))

        out_shape = []
        for i, dim in enumerate(in_shape):
            if i < len(normalized_pw):
                out_shape.append(_compute_pad_dim(dim, normalized_pw[i]))
            else:
                out_shape.append(dim)
        return tuple(out_shape)


@dispatch_eager("Pad")
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
    op = Pad()
    out_shape = op.infer_shape(array, pad_width, mode, **kwargs)

    attributes = {"pad_width": pad_width, "mode": mode}
    attributes.update(kwargs)

    return _emit_shape_node("Pad", [array], attributes, out_shape, getattr(array, "dtype", None))


@dispatch_eager("TopK")
def top_k(operand: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Returns the top k values and their indices along the last dimension.

    Args:
        operand (Tensor): The input tensor
        k (int): Number of top elements to look for

    Returns:
    tuple[Tensor, Tensor]: Top k values and their indices

    """
    out_shape = list(operand.shape) if operand.shape else []
    if out_shape:
        out_shape[-1] = k
    out_shape = tuple(out_shape)

    inputs = [operand]
    # We cheat a bit by returning two tensors pointing to the same node for now,
    # as handling multi-output nodes properly requires more IR scaffolding
    val_node = _emit_shape_node("TopK", inputs, {"k": k, "return_indices": False}, out_shape, operand.dtype)
    idx_node = _emit_shape_node("TopK", inputs, {"k": k, "return_indices": True}, out_shape, DType.Int32)
    return val_node, idx_node


def argsort(
    operand: Tensor,
    dimension: int = -1,
    is_stable: bool = True,
    axis: int | None = None,
    dim: int | None = None,
) -> Tensor:
    """Returns the indices that would sort an array along a given dimension.

    Args:
        operand (Tensor): The input tensor
        dimension (int): The dimension to sort along
        is_stable (bool): Whether to use a stable sorting algorithm
        axis (int): Alias for dimension.
        dim (int): Alias for dimension.

    Returns:
    Tensor: The indices that sort the tensor
    """
    if axis is not None:
        dimension = axis
    if dim is not None:
        dimension = dim

    if config.eager_mode:
        backend = get_active_backend()
        kind = "stable" if is_stable else "quicksort"
        data = backend.execute_op("ArgSort", (operand.data if type(operand).__name__ == "Tensor" else operand), axis=dimension, kind=kind)

        return Tensor(data, TensorConfig(operand.shape, DType.Int32, operand.device))

    inputs = [operand]
    attributes = {"dimension": dimension, "is_stable": is_stable}

    return _emit_shape_node("ArgSort", inputs, attributes, operand.shape, DType.Int32)


def sort(
    operand: Tensor,
    dimension: int = -1,
    is_stable: bool = True,
    axis: int | None = None,
    dim: int | None = None,
) -> Tensor:
    """Sorts the elements of an array along a given dimension.

    Args:
        operand (Tensor): The input tensor
        dimension (int): The dimension to sort along
        is_stable (bool): Whether to use a stable sorting algorithm
        axis (int): Alias for dimension.
        dim (int): Alias for dimension.

    Returns:
    Tensor: The sorted tensor
    """
    if axis is not None:
        dimension = axis
    if dim is not None:
        dimension = dim

    if config.eager_mode:
        backend = get_active_backend()
        kind = "stable" if is_stable else "quicksort"
        data = backend.execute_op("Sort", (operand.data if type(operand).__name__ == "Tensor" else operand), axis=dimension, kind=kind)
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
    op = Resize()
    out_shape = op.infer_shape(image, shape, method)

    return _emit_shape_node(
        "Resize",
        [image],
        {"shape": shape, "method": method},
        out_shape,
        image.dtype,
    )


@register_op("DynamicShape")
class DynamicShape(OpDef):
    """DynamicShape op."""

    def infer_shape(self, x: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return (len(getattr(x, "shape", ())),)


@register_op("Rank")
class Rank(OpDef):
    """Rank op."""

    op_name = "Rank"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Size")
class Size(OpDef):
    """Size op."""

    op_name = "Size"

    def infer_shape(self, a: object, axis: object = None, **kwargs: object) -> object:
        """Infer shape."""
        return ()


def pad_constant(array: object, pad_width: object, value: float = 0.0, **kwargs: object) -> object:
    """Pad an array with a constant value.

    Args:
        array: The array to pad.
        pad_width: The padding widths.
        value: The constant value to pad with.
        kwargs: Additional arguments.

    Returns:
        The padded array.
    """
    return pad(array, pad_width, mode="constant", constant_values=value, **kwargs)


def pad_reflect(array: object, pad_width: object, **kwargs: object) -> object:
    """Pad an array with reflection.

    Args:
        array: The array to pad.
        pad_width: The padding widths.
        kwargs: Additional arguments.

    Returns:
        The padded array.
    """
    return pad(array, pad_width, mode="reflect", **kwargs)


def pad_replicate(array: object, pad_width: object, **kwargs: object) -> object:
    """Pad an array with edge replication.

    Args:
        array: The array to pad.
        pad_width: The padding widths.
        kwargs: Additional arguments.

    Returns:
        The padded array.
    """
    return pad(array, pad_width, mode="edge", **kwargs)


def pad_circular(array: object, pad_width: object, **kwargs: object) -> object:
    """Pad an array with circular wrapping.

    Args:
        array: The array to pad.
        pad_width: The padding widths.
        kwargs: Additional arguments.

    Returns:
        The padded array.
    """
    return pad(array, pad_width, mode="wrap", **kwargs)


@register_op("Flatnonzero")
class Flatnonzero(OpDef):
    """Return indices that are non-zero in the flattened version of a."""

    op_name = "Flatnonzero"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("IndexInDim")
class IndexInDim(OpDef):
    """Return elements of an array at specific indices along a given dimension."""

    op_name = "IndexInDim"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Lexsort")
class Lexsort(OpDef):
    """Perform an indirect stable sort using a sequence of keys."""

    op_name = "Lexsort"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Nonzero")
class Nonzero(OpDef):
    """Return the indices of the elements that are non-zero."""

    op_name = "Nonzero"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Percentile")
class Percentile(OpDef):
    """Compute the q-th percentile of the data along the specified axis."""

    op_name = "Percentile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Ppermute")
class Ppermute(OpDef):
    """Parallel permute operator."""

    op_name = "Ppermute"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("PsumScatter")
class PsumScatter(OpDef):
    """Parallel sum scatter operator."""

    op_name = "PsumScatter"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Quantile")
class Quantile(OpDef):
    """Compute the q-th quantile of the data along the specified axis."""

    op_name = "Quantile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("RavelMultiIndex")
class RavelMultiIndex(OpDef):
    """Converts a tuple of index arrays into an array of flat indices."""

    op_name = "RavelMultiIndex"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Repeat")
class Repeat(OpDef):
    """Repeat elements of an array."""

    op_name = "Repeat"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Searchsorted")
class Searchsorted(OpDef):
    """Find indices where elements should be inserted to maintain order."""

    op_name = "Searchsorted"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[1].shape if len(args) > 1 and hasattr(args[1], "shape") else ()


@register_op("SortComplex")
class SortComplex(OpDef):
    """Sort a complex array using the real part first, then the imaginary part."""

    op_name = "SortComplex"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Tile")
class Tile(OpDef):
    """Construct an array by repeating A the number of times given by reps."""

    op_name = "Tile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Unique")
class Unique(OpDef):
    """Find the unique elements of an array."""

    op_name = "Unique"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("UpdateSlice")
class UpdateSlice(OpDef):
    """Update a slice of an array."""

    op_name = "UpdateSlice"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()
