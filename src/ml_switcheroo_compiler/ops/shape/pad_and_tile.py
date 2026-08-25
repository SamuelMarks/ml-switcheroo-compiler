"""Module pad_and_tile.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape operations for Tensor objects."""
from collections.abc import Sequence

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.reshape import Resize
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def tile(input: Tensor, reps: Sequence[int]) -> object:
    """Construct a new tensor by repeating the input tensor the specified number of times.

    Args:
        input (Tensor): The input parameter.
        reps (Sequence): The reps parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op("Tile", (input.data if type(input).__name__ == "Tensor" else input), reps)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs: object = [input]
    # shape calculation placeholder
    out_shape: object = inputs[0].shape
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
) -> object:
    """Repeat elements of the input tensor along a specified dimension.

    Args:
        input (Tensor): The input parameter.
        repeats (object): The repeats parameter.
        dim (object): The dim parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op("Repeat", (input.data if type(input).__name__ == "Tensor" else input), repeats, axis=dim)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs: object = [input]
    # shape calculation placeholder
    out_shape: object = inputs[0].shape
    return _emit_shape_node(
        "Repeat",
        inputs,
        {"repeats": repeats, "axis": dim},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def triu(input: Tensor, diagonal: int = 0) -> object:
    """Return the upper triangular part of a matrix or batch of matrices.

    Args:
        input (Tensor): The input parameter.
        diagonal (int): The diagonal parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op("Triu", (input.data if type(input).__name__ == "Tensor" else input), k=diagonal)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs: object = [input]
    # shape calculation placeholder
    out_shape: object = inputs[0].shape
    return _emit_shape_node(
        "Triu",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def tril(input: Tensor, diagonal: int = 0) -> object:
    """Return the lower triangular part of a matrix or batch of matrices.

    Args:
        input (Tensor): The input parameter.
        diagonal (int): The diagonal parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op("Tril", (input.data if type(input).__name__ == "Tensor" else input), k=diagonal)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    inputs: object = [input]
    # shape calculation placeholder
    out_shape: object = inputs[0].shape
    return _emit_shape_node(
        "Tril",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def _compute_meshgrid_shape(inputs: list[Tensor], indexing: str) -> tuple[int, ...]:
    """Compute the shape for a meshgrid.

    Args:
        inputs (object): The inputs parameter.
        indexing (str): The indexing parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if not inputs:
        return ()
    out_shape: object = tuple(t.shape[0] if t.shape else 1 for t in inputs)
    if indexing == "xy" and len(inputs) >= 2:
        return (out_shape[1], out_shape[0]) + out_shape[2:]
    return out_shape


def meshgrid(*tensors: Tensor, indexing: str = "ij") -> Sequence[Tensor]:
    """Create coordinate grids from coordinate vectors.

    Args:
        indexing (str): Indexing.

    Args:
        *tensors (Tensor): Positional args.

    Returns:
        Sequence: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        datas: object = backend.execute_op("Meshgrid", *[(t.data if type(t).__name__ == "Tensor" else t) for t in tensors], indexing=indexing)
        return tuple(Tensor(d, TensorConfig(d.shape, tensors[0].dtype, tensors[0].device)) for d in datas)
    inputs: object = list(tensors)
    out_shape: object = _compute_meshgrid_shape(inputs, indexing)
    dtype: object = inputs[0].dtype if inputs else DType.Float32
    return tuple(_emit_shape_node("Meshgrid", inputs, {"indexing": indexing}, out_shape, dtype) for _ in inputs)


def _normalize_pad_width(pad_width: object, ndim: int) -> tuple[object, ...]:
    """Normalize pad width representation.

    Args:
        pad_width (object): Pad width descriptor.
        ndim (int): Number of dimensions.

    Returns:
        tuple: Normalized pad width.
    """
    if isinstance(pad_width, int):
        return ((pad_width, pad_width),) * ndim
    if isinstance(pad_width, tuple) and len(pad_width) == 2 and isinstance(pad_width[0], int):
        return (pad_width,) * ndim
    return tuple(pad_width) if isinstance(pad_width, (list, tuple)) else ()


def _compute_pad_dim(dim: int, pw: object) -> int:
    """Evaluate _compute_pad_dim operation.

    Args:
        dim (int): The dim parameter.
        pw (object): The pw parameter.

    Returns:
        int: Result.
    """
    if isinstance(pw, int):
        return dim + pw * 2
    if isinstance(pw, tuple) and len(pw) == 2:
        return dim + pw[0] + pw[1]
    return dim


@register_op("Pad")
class Pad(OpDef):
    """Pad op."""

    op_name: object = "Pad"

    def infer_shape(self, array: object, pad_width: object, mode: str = "constant", **kwargs: object) -> object:
        """Infer shape.

        Args:
            array (object): The array parameter.
            pad_width (object): The pad_width parameter.
            mode (str): The mode parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result.
        """
        in_shape: object = getattr(array, "shape", ())
        if not in_shape:
            return ()
        normalized_pw: object = _normalize_pad_width(pad_width, len(in_shape))
        out_shape: object = []
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
    """Pad an array with specified widths and values.

    Args:
        array (object): The array parameter.
        pad_width (object): The pad_width parameter.
        mode (str): The mode parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    op: object = Pad()
    out_shape: object = op.infer_shape(array, pad_width, mode, **kwargs)
    attributes: object = {"pad_width": pad_width, "mode": mode}
    attributes.update(kwargs)
    return _emit_shape_node("Pad", [array], attributes, out_shape, getattr(array, "dtype", None))


@dispatch_eager("TopK")
def top_k(operand: Tensor, k: int) -> object:
    """Return the top k values and their indices along the last dimension.

    Args:
        operand (Tensor): The operand parameter.
        k (int): The k parameter.

    Returns:
        tuple: Result.
    """
    out_shape: object = list(operand.shape) if operand.shape else []
    if out_shape:
        out_shape[-1] = k
    out_shape: object = tuple(out_shape)
    inputs: object = [operand]
    # We cheat a bit by returning two tensors pointing to the same node for now,
    # as handling multi-output nodes properly requires more IR scaffolding
    val_node: object = _emit_shape_node("TopK", inputs, {"k": k, "return_indices": False}, out_shape, operand.dtype)
    idx_node: object = _emit_shape_node("TopK", inputs, {"k": k, "return_indices": True}, out_shape, DType.Int32)
    return val_node, idx_node


def argsort(
    operand: Tensor,
    dimension: int = -1,
    is_stable: bool = True,
    axis: int | None = None,
    dim: int | None = None,
) -> object:
    """Return the indices that would sort an array along a given dimension.

    Args:
        operand (Tensor): The operand parameter.
        dimension (int): The dimension parameter.
        is_stable (bool): The is_stable parameter.
        axis (object): The axis parameter.
        dim (object): The dim parameter.

    Returns:
        Tensor: Result.
    """
    if axis is not None:
        dimension: object = axis
    if dim is not None:
        dimension: object = dim
    if config.eager_mode:
        backend: object = get_active_backend()
        kind: object = "stable" if is_stable else "quicksort"
        data: object = backend.execute_op("ArgSort", (operand.data if type(operand).__name__ == "Tensor" else operand), axis=dimension, kind=kind)
        return Tensor(data, TensorConfig(operand.shape, DType.Int32, operand.device))
    inputs: object = [operand]
    attributes: object = {"dimension": dimension, "is_stable": is_stable}
    return _emit_shape_node("ArgSort", inputs, attributes, operand.shape, DType.Int32)


def sort(
    operand: Tensor,
    dimension: int = -1,
    is_stable: bool = True,
    axis: int | None = None,
    dim: int | None = None,
) -> object:
    """Sorts the elements of an array along a given dimension.

    Args:
        operand (Tensor): The operand parameter.
        dimension (int): The dimension parameter.
        is_stable (bool): The is_stable parameter.
        axis (object): The axis parameter.
        dim (object): The dim parameter.

    Returns:
        Tensor: Result.
    """
    if axis is not None:
        dimension: object = axis
    if dim is not None:
        dimension: object = dim
    if config.eager_mode:
        backend: object = get_active_backend()
        kind: object = "stable" if is_stable else "quicksort"
        data: object = backend.execute_op("Sort", (operand.data if type(operand).__name__ == "Tensor" else operand), axis=dimension, kind=kind)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, operand.dtype, operand.device),
        )
    inputs: object = [operand]
    attributes: object = {"dimension": dimension, "is_stable": is_stable}
    return _emit_shape_node("Sort", inputs, attributes, operand.shape, operand.dtype)


@dispatch_eager("Resize")
def image_resize(image: Tensor, shape: tuple[int, int], method: str = "bilinear") -> object:
    """Resizes an image to the given target shape using interpolation.

    Args:
        image (Tensor): The image parameter.
        shape (tuple): The shape parameter.
        method (str): The method parameter.

    Returns:
        Tensor: Result.
    """
    op: object = Resize()
    out_shape: object = op.infer_shape(image, shape, method)
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

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            x (object): The x parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result.
        """
        return (len(getattr(x, "shape", ())),)


@register_op("Rank")
class Rank(OpDef):
    """Rank op."""

    op_name: object = "Rank"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        args[0] if len(args) > 0 else None
        return ()


@register_op("Size")
class Size(OpDef):
    """Size op."""

    op_name: object = "Size"

    def infer_shape(self, a: object, axis: object = None, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a (object): The a parameter.
            axis (object): The axis parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
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

    op_name: object = "Flatnonzero"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        args[0] if len(args) > 0 else None
        return (None,)


@register_op("Lexsort")
class Lexsort(OpDef):
    """Perform an indirect stable sort using a sequence of keys."""

    op_name: object = "Lexsort"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        keys: object = args[0] if len(args) > 0 else None
        kwargs.get("axis", -1)
        if isinstance(keys, (list, tuple)):
            return getattr(keys[0], "shape", ()) if keys else ()
        in_shape: object = getattr(keys, "shape", ())
        if len(in_shape) > 0:
            return in_shape[1:] if len(in_shape) > 1 else ()
        return ()


@register_op("Nonzero")
class Nonzero(OpDef):
    """Return the indices of the elements that are non-zero."""

    op_name: object = "Nonzero"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        a: object = args[0] if len(args) > 0 else None
        in_shape: object = getattr(a, "shape", ())
        return tuple((None,) for _ in in_shape)


def _infer_shape_percentile_quantile(a: object, q: object, axis: object = None, keepdims: bool = False) -> tuple[int, ...]:
    """Infer shape for percentile and quantile ops.

    Args:
        a (object): The a parameter.
        q (object): The q parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.

    Returns:
        tuple: Result.
    """
    in_shape: object = getattr(a, "shape", ())
    q_shape: object = getattr(q, "shape", ())
    if isinstance(q, (int, float)):
        q_shape: object = ()
    elif isinstance(q, (list, tuple)):
        q_shape: object = (len(q),)
    if axis is None:
        if keepdims:
            return q_shape + (1,) * len(in_shape)
        return q_shape
    axis_tup: object = (axis,) if isinstance(axis, int) else tuple(axis)
    out_shape: object = list(in_shape)
    for ax in sorted(axis_tup, reverse=True):
        if keepdims:
            out_shape[ax] = 1
        else:
            out_shape.pop(ax)
    return q_shape + tuple(out_shape)


@register_op("Percentile")
class Percentile(OpDef):
    """Compute the q-th percentile of the data along the specified axis."""

    op_name: object = "Percentile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        a: object = args[0] if len(args) > 0 else None
        q: object = args[1] if len(args) > 1 else None
        axis: object = kwargs.get("axis", None)
        keepdims: object = kwargs.get("keepdims", False)
        return _infer_shape_percentile_quantile(a, q, axis, keepdims)


@register_op("Quantile")
class Quantile(OpDef):
    """Compute the q-th quantile of the data along the specified axis."""

    op_name: object = "Quantile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        a: object = args[0] if len(args) > 0 else None
        q: object = args[1] if len(args) > 1 else None
        axis: object = kwargs.get("axis", None)
        keepdims: object = kwargs.get("keepdims", False)
        return _infer_shape_percentile_quantile(a, q, axis, keepdims)


@register_op("RavelMultiIndex")
class RavelMultiIndex(OpDef):
    """Convert a tuple of index arrays into an array of flat indices."""

    op_name: object = "RavelMultiIndex"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        multi_index: object = args[0] if len(args) > 0 else None
        args[1] if len(args) > 1 else None
        if isinstance(multi_index, (list, tuple)) and multi_index:
            return getattr(multi_index[0], "shape", ())
        return getattr(multi_index, "shape", ())


def _repeat_infer_no_axis(in_shape: tuple[object, ...], repeats: object) -> tuple[object, ...]:
    """Infer shape for repeat without an axis.

    Args:
        in_shape (tuple): The input shape.
        repeats (object): The repetitions.

    Returns:
        tuple: The inferred shape.
    """
    size: int | None = 1
    for s in in_shape:
        if s is None:
            size: object = None
            break
        size *= s
    if isinstance(repeats, int) and size is not None:
        return (size * repeats,)
    if isinstance(repeats, (list, tuple)):
        return (sum(repeats),)
    return (None,)


@register_op("Repeat")
class Repeat(OpDef):
    """Repeat elements of an array."""

    op_name: object = "Repeat"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        a: object = args[0] if len(args) > 0 else None
        repeats: object = args[1] if len(args) > 1 else None
        axis: object = kwargs.get("axis", None)
        in_shape: object = getattr(a, "shape", ())
        if axis is None:
            return _repeat_infer_no_axis(in_shape, repeats)
        out_shape: object = list(in_shape)
        if isinstance(repeats, int):
            out_shape[axis] = out_shape[axis] * repeats if out_shape[axis] is not None else None
        elif isinstance(repeats, (list, tuple)):
            out_shape[axis] = sum(repeats)
        else:
            out_shape[axis] = None
        return tuple(out_shape)


@register_op("Searchsorted")
class Searchsorted(OpDef):
    """Find indices where elements should be inserted to maintain order."""

    op_name: object = "Searchsorted"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        args[0] if len(args) > 0 else None
        v: object = args[1] if len(args) > 1 else None
        return getattr(v, "shape", ())


@register_op("SortComplex")
class SortComplex(OpDef):
    """Sort a complex array using the real part first, then the imaginary part."""

    op_name: object = "SortComplex"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        a: object = args[0] if len(args) > 0 else None
        return getattr(a, "shape", ())


def _compute_tile_shape(in_shape: tuple[object, ...], reps: tuple[object, ...]) -> tuple[object, ...]:
    """Evaluate _compute_tile_shape operation.

    Args:
        in_shape (tuple): The in_shape parameter.
        reps (tuple): The reps parameter.

    Returns:
        tuple: Result.
    """
    d: object = len(reps)
    c: object = len(in_shape)
    if c < d:
        in_shape: object = (1,) * (d - c) + in_shape
    elif c > d:
        reps: object = (1,) * (c - d) + reps
    return tuple(s * r if s is not None and r is not None else None for s, r in zip(in_shape, reps))


@register_op("Tile")
class Tile(OpDef):
    """Construct an array by repeating A the number of times given by reps."""

    op_name: object = "Tile"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        A = args[0] if len(args) > 0 else None
        reps: object = args[1] if len(args) > 1 else None
        if reps is None:
            return ()
        in_shape: object = getattr(A, "shape", ())
        if isinstance(reps, int):
            reps: object = (reps,)
        else:
            reps: object = tuple(reps)
        return _compute_tile_shape(in_shape, reps)


def _get_unique_inverse_shape(axis: int | None, in_shape: tuple[object, ...]) -> tuple[object, ...]:
    """Get the inverse shape for the unique operation.

    Args:
        axis (int | None): The axis along which unique operates.
        in_shape (tuple): The input shape.

    Returns:
        tuple: The inverse shape.
    """
    if axis is not None:
        return (in_shape[axis],)
    size: int | None = 1
    for s in in_shape:
        if s is None:
            size: object = None
            break
        size *= s
    return (size,)


@register_op("Unique")
class Unique(OpDef):
    """Find the unique elements of an array."""

    op_name: object = "Unique"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        ar: object = args[0] if len(args) > 0 else None
        return_index: object = kwargs.get("return_index", False)
        return_inverse: object = kwargs.get("return_inverse", False)
        return_counts: object = kwargs.get("return_counts", False)
        axis: object = kwargs.get("axis", None)
        in_shape: object = getattr(ar, "shape", ())
        if axis is None:
            ret_shape: object = (None,)
        else:
            ret_shape_list: object = list(in_shape)
            ret_shape_list[axis] = None
            ret_shape: object = tuple(ret_shape_list)
        ret: object = [ret_shape]
        if return_index:
            ret.append((None,) if axis is None else (in_shape[axis],))
        if return_inverse:
            ret.append(_get_unique_inverse_shape(axis, in_shape))
        if return_counts:
            ret.append((None,))
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)


def percentile(*args: object, **kwargs: object) -> object:
    """Evaluate percentile operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Percentile", *args, **kwargs)


def quantile(*args: object, **kwargs: object) -> object:
    """Evaluate quantile operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Quantile", *args, **kwargs)


def flatnonzero(*args: object, **kwargs: object) -> object:
    """Return indices that are non-zero in the flattened version of a.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Flatnonzero", *args, **kwargs)


def nonzero(*args: object, **kwargs: object) -> object:
    """Return the indices of the elements that are non-zero.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Nonzero", *args, **kwargs)


def ravel_multi_index(*args: object, **kwargs: object) -> object:
    """Convert a tuple of index arrays into an array of flat indices.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("RavelMultiIndex", *args, **kwargs)


def lexsort(*args: object, **kwargs: object) -> object:
    """Perform an indirect stable sort using a sequence of keys.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Lexsort", *args, **kwargs)


def searchsorted(*args: object, **kwargs: object) -> object:
    """Find indices where elements should be inserted to maintain order.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Searchsorted", *args, **kwargs)


def sort_complex(*args: object, **kwargs: object) -> object:
    """Sort a complex array using the real part first, then the imaginary part.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("SortComplex", *args, **kwargs)


def unique(*args: object, **kwargs: object) -> object:
    """Find the unique elements of an array.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Unique", *args, **kwargs)
