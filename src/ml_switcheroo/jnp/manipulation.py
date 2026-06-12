"""Docstring."""

from typing import Any, Optional
import ml_switcheroo.ops as ops
from ml_switcheroo.jnp.creation import asarray
from ml_switcheroo.jnp.array import _to_tensor, _wrap


def transpose(x: object, axes: Optional[list[int]] = None) -> object:
    """Reverse or permute the axes of an array.

    Args:
        x (Any): Argument x.
        axes (Any): Argument axes.

    Returns:
        Any: The result of the operation.
    """
    t = _to_tensor(x)
    if axes is not None:
        return _wrap(ops.permute(t, dims=axes))
    axes = list(range(len(t.shape))[::-1])
    return _wrap(ops.permute(t, dims=axes))


def reshape(x: object, newshape: tuple[int, ...]) -> object:
    """Gives a new shape to an array without changing its data.

    Args:
        x (Any): Argument x.
        newshape (Any): Argument newshape.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.reshape(_to_tensor(x), shape=newshape))


def broadcast_to(x: object, shape: tuple[int, ...]) -> object:
    """Broadcast an array to a new shape.

    Args:
        x (Any): Argument x.
        shape (Any): Argument shape.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.broadcast_to(_to_tensor(x), size=shape))


def concatenate(arrays: list[Any], axis: int = 0) -> object:
    """Join a sequence of arrays along an existing axis.

    Args:
        arrays (Any): Argument arrays.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in arrays]
    return _wrap(ops.concatenate(tensors, dim=axis))


def where(condition: object, x: object, y: object) -> object:
    """Return elements chosen from x or y depending on condition.

    Args:
        condition (Any): Argument condition.
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.where(_to_tensor(condition), _to_tensor(x), _to_tensor(y)))


def expand_dims(a: object, axis: int) -> object:
    """Expand the shape of an array.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.unsqueeze(_to_tensor(a), dim=axis))


def ravel(a: object, order: str = "C") -> object:
    """Return a contiguous flattened array.

    Args:
        a (Any): Argument a.
        order (Any): Argument order.

    Returns:
        Any: The result of the operation.
    """
    # Eager fallback or reshape if order='C'
    if order != "C":
        raise NotImplementedError("ravel only supports order='C'")
    return reshape(a, (-1,))


def squeeze(a: object, axis: object = None) -> object:
    """Remove axes of length one from a.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.squeeze(_to_tensor(a), dim=axis))


def swapaxes(a: object, axis1: int, axis2: int) -> object:
    """Interchange two axes of an array.

    Args:
        a (Any): Argument a.
        axis1 (Any): Argument axis1.
        axis2 (Any): Argument axis2.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.swapaxes(_to_tensor(a), axis1=axis1, axis2=axis2))


def moveaxis(a: object, source: object, destination: object) -> object:
    """Move axes of an array to new positions.

    Args:
        a (Any): Argument a.
        source (Any): Argument source.
        destination (Any): Argument destination.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.moveaxis(_to_tensor(a), source=source, destination=destination))


def stack(arrays: object, axis: int = 0) -> object:
    """Join a sequence of arrays along a new axis.

    Args:
        arrays (Any): Argument arrays.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(arr) for arr in arrays]
    return _wrap(ops.stack(tensors, dim=axis))


def vstack(tup: object) -> object:
    """Stack arrays in sequence vertically (row wise).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.vstack([_to_tensor(arr) for arr in tup]))


def hstack(tup: object) -> object:
    """Stack arrays in sequence horizontally (column wise).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.hstack([_to_tensor(arr) for arr in tup]))


def dstack(tup: object) -> object:
    """Stack arrays in sequence depth wise (along third axis).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.dstack([_to_tensor(arr) for arr in tup]))


def split(ary: object, indices_or_sections: object, axis: int = 0) -> object:
    """Split an array into multiple sub-arrays as views into ary.

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return tuple(
        _wrap(t) for t in ops.split(_to_tensor(ary), indices_or_sections, axis)
    )


def array_split(ary: object, indices_or_sections: object, axis: int = 0) -> object:
    """Split an array into multiple sub-arrays.

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return tuple(
        _wrap(t) for t in ops.array_split(_to_tensor(ary), indices_or_sections, axis)
    )


def vsplit(ary: object, indices_or_sections: object) -> object:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.vsplit(_to_tensor(ary), indices_or_sections))


def hsplit(ary: object, indices_or_sections: object) -> object:
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.hsplit(_to_tensor(ary), indices_or_sections))


def dsplit(ary: object, indices_or_sections: object) -> object:
    """Split array into multiple sub-arrays along the 3rd axis (depth).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.dsplit(_to_tensor(ary), indices_or_sections))


def tile(A: object, reps: object) -> object:
    """Construct an array by repeating A the number of times given by reps.

    Args:
        A (Any): Argument A.
        reps (Any): Argument reps.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tile(_to_tensor(A), reps=reps))


def repeat(a: object, repeats: object, axis: object = None) -> object:
    """Repeat elements of an array.

    Args:
        a (Any): Argument a.
        repeats (Any): Argument repeats.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.repeat(_to_tensor(a), repeats=repeats, axis=axis))


def pad(
    array: object, pad_width: object, mode: str = "constant", **kwargs: object
) -> object:
    """Pad an array.

    Args:
        array (Any): Argument array.
        pad_width (Any): Argument pad_width.
        mode (Any): Argument mode.
        **kwargs: Additional kwargs.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.pad(_to_tensor(array), pad_width, mode=mode, **kwargs))


def take(a: object, indices: object, axis: int = None, mode: str = None) -> object:
    """Take elements from an array along an axis.

    Args:
        a (Any): Argument a.
        indices (Any): Argument indices.
        axis (Any): Argument axis.
        mode (Any): Argument mode.
        **kwargs: Additional kwargs.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.take(_to_tensor(a), _to_tensor(indices)))


def take_along_axis(arr: object, indices: object, axis: int) -> object:
    """Take values from the input array by matching 1d index and data slices.

    Args:
        arr (Any): Argument arr.
        indices (Any): Argument indices.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.take_along_axis(_to_tensor(arr), _to_tensor(indices), axis=axis))


def shape(a: object) -> object:
    """Get the shape of the array.

    Returns:
        Any: The shape property of the underlying tensor.
    """
    return asarray(a).shape
