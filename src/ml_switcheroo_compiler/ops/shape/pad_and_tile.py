from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape operations for Tensor objects."""
from collections.abc import Sequence
from typing import Any

# pylint: disable=duplicate-code
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.reshape import Resize
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def tile(input: Tensor, reps: Sequence[int]) -> Any:
    """Construct a new tensor by repeating the input tensor the specified number of times.

    Args:
        input (Tensor): The input parameter.
        reps (Sequence): The reps parameter.

    Returns:
        Tensor: Result.
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
) -> Any:
    """Repeat elements of the input tensor along a specified dimension.

    Args:
        input (Tensor): The input parameter.
        repeats (object): The repeats parameter.
        dim (object): The dim parameter.

    Returns:
        Tensor: Result.
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


def triu(input: Tensor, diagonal: int = 0) -> Any:
    """Return the upper triangular part of a matrix or batch of matrices.

    Args:
        input (Tensor): The input parameter.
        diagonal (int): The diagonal parameter.

    Returns:
        Tensor: Result.
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


def tril(input: Tensor, diagonal: int = 0) -> Any:
    """Return the lower triangular part of a matrix or batch of matrices.

    Args:
        input (Tensor): The input parameter.
        diagonal (int): The diagonal parameter.

    Returns:
        Tensor: Result.
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
    """Compute the shape for a meshgrid.

    Args:
        inputs (object): The inputs parameter.
        indexing (str): The indexing parameter.

    Returns: Any: Result.
    """
    if not inputs:
        return ()
    out_shape = tuple(t.shape[0] if t.shape else 1 for t in inputs)
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
        backend = get_active_backend()
        datas = backend.execute_op("Meshgrid", *[(t.data if type(t).__name__ == "Tensor" else t) for t in tensors], indexing=indexing)
        return tuple(Tensor(d, TensorConfig(d.shape, tensors[0].dtype, tensors[0].device)) for d in datas)
    inputs = list(tensors)
    out_shape = _compute_meshgrid_shape(inputs, indexing)
    dtype = inputs[0].dtype if inputs else DType.Float32
    return tuple(_emit_shape_node("Meshgrid", inputs, {"indexing": indexing}, out_shape, dtype) for _ in inputs)


def _normalize_pad_width(pad_width: Any, ndim: int) -> tuple:
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


def _compute_pad_dim(dim: int, pw: Any) -> int:
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

    op_name = "Pad"

    def infer_shape(self, array: Any, pad_width: Any, mode: str = "constant", **kwargs: Any) -> tuple[int, ...]:
        """Infer shape.

        Args:
            array (object): The array parameter.
            pad_width (object): The pad_width parameter.
            mode (str): The mode parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result.
        """
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
    array: Any,
    pad_width: Any,
    mode: str = "constant",
    **kwargs: Any,
) -> Any:
    """Pad an array with specified widths and values.

    Args:
        array (object): The array parameter.
        pad_width (object): The pad_width parameter.
        mode (str): The mode parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    op = Pad()
    out_shape = op.infer_shape(array, pad_width, mode, **kwargs)
    attributes = {"pad_width": pad_width, "mode": mode}
    attributes.update(kwargs)
    return _emit_shape_node("Pad", [array], attributes, out_shape, getattr(array, "dtype", None))


@dispatch_eager("TopK")
def top_k(operand: Tensor, k: int) -> Any:
    """Return the top k values and their indices along the last dimension.

    Args:
        operand (Tensor): The operand parameter.
        k (int): The k parameter.

    Returns:
        tuple: Result.
    """
    out_shape = list(operand.shape) if operand.shape else []
    if out_shape:
        out_shape[-1] = k
    out_shape = tuple(out_shape)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
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
) -> Any:
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
        dimension = axis
    if dim is not None:
        dimension = dim
    if config.eager_mode:
        backend = get_active_backend()
        kind = "stable" if is_stable else "quicksort"
        data = backend.execute_op("ArgSort", (operand.data if type(operand).__name__ == "Tensor" else operand), axis=dimension, kind=kind)
        return Tensor(data, TensorConfig(operand.shape, DType.Int32, operand.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    inputs = [operand]
    attributes = {"dimension": dimension, "is_stable": is_stable}
    return _emit_shape_node("ArgSort", inputs, attributes, operand.shape, DType.Int32)


def sort(
    operand: Tensor,
    dimension: int = -1,
    is_stable: bool = True,
    axis: int | None = None,
    dim: int | None = None,
) -> Any:
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
def image_resize(image: Tensor, shape: tuple[int, int], method: str = "bilinear") -> Any:
    """Resizes an image to the given target shape using interpolation.

    Args:
        image (Tensor): The image parameter.
        shape (tuple): The shape parameter.
        method (str): The method parameter.

    Returns:
        Tensor: Result.
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

    def infer_shape(self, x: Any, **kwargs: Any) -> tuple[int, ...]:
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

    op_name = "Rank"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        args[0] if len(args) > 0 else None
        return ()


@register_op("Size")
class Size(OpDef):
    """Size op."""

    op_name = "Size"

    def infer_shape(self, a: Any, axis: Any = None, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            a (object): The a parameter.
            axis (object): The axis parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


def pad_constant(array: Any, pad_width: Any, value: float = 0.0, **kwargs: Any) -> Any:
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


def pad_reflect(array: Any, pad_width: Any, **kwargs: Any) -> Any:
    """Pad an array with reflection.

    Args:
        array: The array to pad.
        pad_width: The padding widths.
        kwargs: Additional arguments.

    Returns:
        The padded array.
    """
    return pad(array, pad_width, mode="reflect", **kwargs)


def pad_replicate(array: Any, pad_width: Any, **kwargs: Any) -> Any:
    """Pad an array with edge replication.

    Args:
        array: The array to pad.
        pad_width: The padding widths.
        kwargs: Additional arguments.

    Returns:
        The padded array.
    """
    return pad(array, pad_width, mode="edge", **kwargs)


def pad_circular(array: Any, pad_width: Any, **kwargs: Any) -> Any:
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

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        args[0] if len(args) > 0 else None
        return (None,)


@register_op("Lexsort")
class Lexsort(OpDef):
    """Perform an indirect stable sort using a sequence of keys."""

    op_name = "Lexsort"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        keys = args[0] if len(args) > 0 else None
        kwargs.get("axis", -1)
        if isinstance(keys, (list, tuple)):
            return getattr(keys[0], "shape", ()) if keys else ()
        in_shape = getattr(keys, "shape", ())
        if len(in_shape) > 0:
            return in_shape[1:] if len(in_shape) > 1 else ()
        return ()


@register_op("Nonzero")
class Nonzero(OpDef):
    """Return the indices of the elements that are non-zero."""

    op_name = "Nonzero"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        a = args[0] if len(args) > 0 else None
        in_shape = getattr(a, "shape", ())
        return tuple((None,) for _ in in_shape)


def _infer_shape_percentile_quantile(a: Any, q: Any, axis: Any = None, keepdims: bool = False) -> tuple[int, ...]:
    """Infer shape for percentile and quantile ops.

    Args:
        a (object): The a parameter.
        q (object): The q parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.

    Returns:
        tuple: Result.
    """
    in_shape = getattr(a, "shape", ())
    q_shape = getattr(q, "shape", ())
    if isinstance(q, (int, float)):
        q_shape = ()
    elif isinstance(q, (list, tuple)):
        q_shape = (len(q),)
    if axis is None:
        if keepdims:
            return q_shape + (1,) * len(in_shape)
        return q_shape
    axis_tup = (axis,) if isinstance(axis, int) else tuple(axis)  # type: ignore[arg-type]
    out_shape = list(in_shape)
    for ax in sorted(axis_tup, reverse=True):
        if keepdims:
            out_shape[ax] = 1
        else:
            out_shape.pop(ax)
    return q_shape + tuple(out_shape)


@register_op("Percentile")
class Percentile(OpDef):
    """Compute the q-th percentile of the data along the specified axis."""

    op_name = "Percentile"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        a = args[0] if len(args) > 0 else None
        q = args[1] if len(args) > 1 else None
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        return _infer_shape_percentile_quantile(a, q, axis, keepdims)


@register_op("Quantile")
class Quantile(OpDef):
    """Compute the q-th quantile of the data along the specified axis."""

    op_name = "Quantile"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        a = args[0] if len(args) > 0 else None
        q = args[1] if len(args) > 1 else None
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        return _infer_shape_percentile_quantile(a, q, axis, keepdims)


@register_op("RavelMultiIndex")
class RavelMultiIndex(OpDef):
    """Convert a tuple of index arrays into an array of flat indices."""

    op_name = "RavelMultiIndex"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        multi_index = args[0] if len(args) > 0 else None
        args[1] if len(args) > 1 else None
        if isinstance(multi_index, (list, tuple)) and multi_index:
            return getattr(multi_index[0], "shape", ())
        return getattr(multi_index, "shape", ())


def _repeat_infer_no_axis(in_shape: tuple, repeats: Any) -> tuple:
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
            size = None
            break
        size *= s  # type: ignore[operator]
    if isinstance(repeats, int) and size is not None:
        return (size * repeats,)
    if isinstance(repeats, (list, tuple)):
        return (sum(repeats),)
    return (None,)


@register_op("Repeat")
class Repeat(OpDef):
    """Repeat elements of an array."""

    op_name = "Repeat"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        a = args[0] if len(args) > 0 else None
        repeats = args[1] if len(args) > 1 else None
        axis = kwargs.get("axis", None)
        in_shape = getattr(a, "shape", ())
        if axis is None:
            return _repeat_infer_no_axis(in_shape, repeats)
        out_shape = list(in_shape)
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

    op_name = "Searchsorted"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        args[0] if len(args) > 0 else None
        v = args[1] if len(args) > 1 else None
        return getattr(v, "shape", ())


@register_op("SortComplex")
class SortComplex(OpDef):
    """Sort a complex array using the real part first, then the imaginary part."""

    op_name = "SortComplex"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        a = args[0] if len(args) > 0 else None
        return getattr(a, "shape", ())


def _compute_tile_shape(in_shape: tuple, reps: tuple) -> tuple:
    """Evaluate _compute_tile_shape operation.

    Args:
        in_shape (tuple): The in_shape parameter.
        reps (tuple): The reps parameter.

    Returns:
        tuple: Result.
    """
    d = len(reps)
    c = len(in_shape)
    if c < d:
        in_shape = (1,) * (d - c) + in_shape
    elif c > d:
        reps = (1,) * (c - d) + reps
    return tuple(s * r if s is not None and r is not None else None for s, r in zip(in_shape, reps))


@register_op("Tile")
class Tile(OpDef):
    """Construct an array by repeating A the number of times given by reps."""

    op_name = "Tile"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        A = args[0] if len(args) > 0 else None
        reps = args[1] if len(args) > 1 else None
        if reps is None:
            return ()
        in_shape = getattr(A, "shape", ())
        if isinstance(reps, int):
            reps = (reps,)
        else:
            reps = tuple(reps)  # type: ignore[arg-type]
        return _compute_tile_shape(in_shape, reps)


def _get_unique_inverse_shape(axis: int | None, in_shape: tuple) -> tuple:
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
            size = None
            break
        size *= s  # type: ignore[operator]
    return (size,)


@register_op("Unique")
class Unique(OpDef):
    """Find the unique elements of an array."""

    op_name = "Unique"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        ar = args[0] if len(args) > 0 else None
        return_index = kwargs.get("return_index", False)
        return_inverse = kwargs.get("return_inverse", False)
        return_counts = kwargs.get("return_counts", False)
        axis = kwargs.get("axis", None)
        in_shape = getattr(ar, "shape", ())
        if axis is None:
            ret_shape = (None,)
        else:
            ret_shape_list = list(in_shape)
            ret_shape_list[axis] = None
            ret_shape = tuple(ret_shape_list)
        ret = [ret_shape]
        if return_index:
            ret.append((None,) if axis is None else (in_shape[axis],))
        if return_inverse:
            ret.append(_get_unique_inverse_shape(axis, in_shape))
        if return_counts:
            ret.append((None,))
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)


def percentile(*args: Any, **kwargs: Any) -> Any:
    """Evaluate percentile operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Percentile", *args, **kwargs)


def quantile(*args: Any, **kwargs: Any) -> Any:
    """Evaluate quantile operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Quantile", *args, **kwargs)


def flatnonzero(*args: Any, **kwargs: Any) -> Any:
    """Return indices that are non-zero in the flattened version of a.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Flatnonzero", *args, **kwargs)


def nonzero(*args: Any, **kwargs: Any) -> Any:
    """Return the indices of the elements that are non-zero.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Nonzero", *args, **kwargs)


def ravel_multi_index(*args: Any, **kwargs: Any) -> Any:
    """Convert a tuple of index arrays into an array of flat indices.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("RavelMultiIndex", *args, **kwargs)


def lexsort(*args: Any, **kwargs: Any) -> Any:
    """Perform an indirect stable sort using a sequence of keys.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Lexsort", *args, **kwargs)


def searchsorted(*args: Any, **kwargs: Any) -> Any:
    """Find indices where elements should be inserted to maintain order.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Searchsorted", *args, **kwargs)


def sort_complex(*args: Any, **kwargs: Any) -> Any:
    """Sort a complex array using the real part first, then the imaginary part.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("SortComplex", *args, **kwargs)


def unique(*args: Any, **kwargs: Any) -> Any:
    """Find the unique elements of an array.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Unique", *args, **kwargs)
