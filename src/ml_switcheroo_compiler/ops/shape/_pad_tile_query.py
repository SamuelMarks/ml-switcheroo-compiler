"""Non-zero index queries, multi-index raveling, order statistics, and unique element operations."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import (
    coerce_bool,
    coerce_int,
    coerce_opt_int,
    coerce_shape,
)
from ml_switcheroo_compiler.core.shape import OpValue, extract_shape, extract_symbolic_shape
from ml_switcheroo_compiler.ir.shape_system import SymNode
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _extract_shape_or_seq


@register_op("Flatnonzero")
class Flatnonzero(OpDef):
    """Schema definition returning non-zero indices of a flattened array."""

    op_name = "Flatnonzero"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer symbolic 1D shape `(SymVar('nonzero_count'),)` for flat non-zero indices.

        Args:
            *args (OpValue): Input tensor operand positional arguments.
            **kwargs (OpValue): Unused keyword operation attributes.

        Returns:
            OpValue: 1D symbolic shape tuple representing non-zero element count.
        """
        del args, kwargs
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        return (SymVar("nonzero_count"),)


def flatnonzero(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return flat indices of elements that are non-zero in the flattened input array.

    Args:
        *args (OpValue): Input array operand.
        **kwargs (OpValue): Additional keyword options forwarded to `Flatnonzero`.

    Returns:
        OpValue: 1D tensor of non-zero flat indices.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Flatnonzero", *args, **kwargs)


@register_op("Nonzero")
class Nonzero(OpDef):
    """Schema definition returning coordinate index arrays of non-zero elements."""

    op_name = "Nonzero"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer tuple of 1D symbolic shapes `(SymVar('nonzero_count'),)` per input axis.

        Args:
            *args (OpValue): Input array operand positional arguments.
            **kwargs (OpValue): Optional keyword arguments including `a`.

        Returns:
            OpValue: Tuple of 1D symbolic coordinate array shapes, one per axis.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        a = args[0] if len(args) > 0 else kwargs.get("a")
        in_shape = _extract_shape_or_seq(a)
        return tuple((SymVar("nonzero_count"),) for _ in in_shape)


def nonzero(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return coordinate index tensors of elements that are non-zero.

    Args:
        *args (OpValue): Input array operand.
        **kwargs (OpValue): Additional keyword options forwarded to `Nonzero`.

    Returns:
        OpValue: Tuple of 1D index tensors for each dimension of the input.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Nonzero", *args, **kwargs)


@register_op("RavelMultiIndex")
class RavelMultiIndex(OpDef):
    """Schema definition converting multi-dimensional coordinate arrays into flat indices."""

    op_name = "RavelMultiIndex"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer flat index array shape from a tuple of coordinate index arrays.

        Args:
            *args (OpValue): Tuple of coordinate arrays followed by grid `dims`.
            **kwargs (OpValue): Optional keyword arguments including `multi_index`.

        Returns:
            tuple[int, ...]: Output shape matching the coordinate index arrays.
        """
        multi_index = args[0] if len(args) > 0 else kwargs.get("multi_index")
        if isinstance(multi_index, (list, tuple)) and multi_index:
            first = multi_index[0]
            if isinstance(first, (list, tuple)) and first:
                return _extract_shape_or_seq(first[0])
            return _extract_shape_or_seq(first)
        return extract_shape(multi_index)


def ravel_multi_index(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Convert a tuple of coordinate index arrays into an array of flat linear indices.

    Args:
        *args (OpValue): Multi-index coordinate tuple and target grid dimensions.
        **kwargs (OpValue): Optional `mode` and `order` keyword parameters.

    Returns:
        OpValue: Tensor of flattened linear indices.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("RavelMultiIndex", *args, **kwargs)


def _infer_shape_percentile_quantile(
    a: OpValue,
    q: OpValue,
    axis: OpValue = None,
    keepdims: bool = False,
) -> tuple[int, ...]:
    """Infer output shape shared by `Percentile` and `Quantile` reductions.

    Args:
        a (OpValue): Input data tensor or array-like operand.
        q (OpValue): Scalar or 1D sequence of target percentiles/quantiles.
        axis (OpValue): Axis or tuple of axes along which statistics are computed.
        keepdims (bool): Whether reduced axes are retained with size `1`.

    Returns:
        tuple[int, ...]: Resulting shape combining leading `q` dimensions with reduced shape.
    """
    in_shape = extract_shape(a)
    q_shape = extract_shape(q)
    if isinstance(q, (int, float)):
        q_shape = ()
    elif isinstance(q, (list, tuple)):
        q_shape = (len(q),)
    if axis is None:
        if keepdims:
            _typed_ret_3: tuple[int, ...] = q_shape + (1,) * len(in_shape)
            return _typed_ret_3
        _typed_ret_2: tuple[int, ...] = q_shape
        return _typed_ret_2
    axis_tup = (axis,) if isinstance(axis, int) else coerce_shape(axis)
    out_shape = list(in_shape)
    for ax in sorted(axis_tup, reverse=True):
        if keepdims:
            out_shape[ax] = 1
        else:
            out_shape.pop(ax)
    _typed_ret_1: tuple[int, ...] = q_shape + tuple(out_shape)
    return _typed_ret_1


@register_op("Percentile")
class Percentile(OpDef):
    """Schema definition computing the q-th percentile along a specified axis."""

    op_name = "Percentile"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer output shape for `Percentile`.

        Args:
            *args (OpValue): Input tensor `a` and percentile value(s) `q`.
            **kwargs (OpValue): Optional `axis` and `keepdims` reduction attributes.

        Returns:
            OpValue: Inferred percentile output shape tuple.
        """
        a = args[0] if len(args) > 0 else None
        q = args[1] if len(args) > 1 else None
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        return _infer_shape_percentile_quantile(a, q, axis, coerce_bool(keepdims))


def percentile(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the q-th percentile of data along the specified axis.

    Args:
        *args (OpValue): Input data tensor and percentile value(s) in `[0, 100]`.
        **kwargs (OpValue): Reduction options including `axis`, `method`, and `keepdims`.

    Returns:
        OpValue: Computed percentile tensor or scalar.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Percentile", *args, **kwargs)


@register_op("Quantile")
class Quantile(OpDef):
    """Schema definition computing the q-th quantile along a specified axis."""

    op_name = "Quantile"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer output shape for `Quantile`.

        Args:
            *args (OpValue): Input tensor `a` and quantile probability value(s) `q`.
            **kwargs (OpValue): Optional `axis` and `keepdims` reduction attributes.

        Returns:
            OpValue: Inferred quantile output shape tuple.
        """
        a = args[0] if len(args) > 0 else None
        q = args[1] if len(args) > 1 else None
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        return _infer_shape_percentile_quantile(a, q, axis, coerce_bool(keepdims))


def quantile(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Compute the q-th quantile of data along the specified axis.

    Args:
        *args (OpValue): Input data tensor and quantile value(s) in `[0.0, 1.0]`.
        **kwargs (OpValue): Reduction options including `axis`, `method`, and `keepdims`.

    Returns:
        OpValue: Computed quantile tensor or scalar.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Quantile", *args, **kwargs)


def _get_unique_inverse_shape(axis: int | None, in_shape: Sequence[int | None]) -> tuple[int | None, ...]:
    """Compute the `return_inverse` index tensor shape for `Unique`.

    Args:
        axis (int | None): Optional axis along which unique slices are deduplicated.
        in_shape (Sequence[int | None]): Input tensor dimensions.

    Returns:
        tuple[int | None, ...]: 1D inverse index shape matching flattened size or `in_shape[axis]`.
    """
    if axis is not None:
        return (in_shape[axis],)
    size: int | None = 1
    for s in in_shape:
        if s is None:
            size = None
            break
        size = int(coerce_int(size)) * s
    return (size,)


@register_op("Unique")
class Unique(OpDef):
    """Schema definition finding unique elements and optional index/count metadata."""

    op_name = "Unique"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer output shape(s) for `Unique` including optional index, inverse, and count outputs.

        Args:
            *args (OpValue): Input array operand `ar`.
            **kwargs (OpValue): Flags `return_index`, `return_inverse`, `return_counts`, and `axis`.

        Returns:
            OpValue: Single symbolic shape tuple or tuple of output shape tuples.
        """
        ar = args[0] if len(args) > 0 else None
        return_index = kwargs.get("return_index", False)
        return_inverse = kwargs.get("return_inverse", False)
        return_counts = kwargs.get("return_counts", False)
        axis = kwargs.get("axis", None)
        in_shape = tuple(d if isinstance(d, int) else None for d in extract_symbolic_shape(ar))
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        if axis is None:
            ret_shape: tuple[int | SymNode | None, ...] = (SymVar("unique_count"),)
        else:
            ret_shape_list: list[int | SymNode | None] = list(in_shape)
            ret_shape_list[coerce_int(axis)] = SymVar("unique_count")
            ret_shape = tuple(ret_shape_list)
        ret: list[tuple[int | SymNode | None, ...]] = [ret_shape]
        if return_index:
            ret.append((SymVar("unique_count"),) if axis is None else (in_shape[coerce_int(axis)],))
        if return_inverse:
            ret.append(_get_unique_inverse_shape(coerce_opt_int(axis), in_shape))
        if return_counts:
            ret.append((SymVar("unique_count"),))
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)


def unique(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Find the sorted unique elements of an input array.

    Args:
        *args (OpValue): Input tensor operand.
        **kwargs (OpValue): Optional flags `return_index`, `return_inverse`, `return_counts`, and `axis`.

    Returns:
        OpValue: Unique values tensor, or tuple with requested auxiliary index/count tensors.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Unique", *args, **kwargs)
