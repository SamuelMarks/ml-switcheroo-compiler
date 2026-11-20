"""Sorting, top-k selection, binary search, and lexical indirect sort operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue, broadcast_shapes, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager, register_op
from ml_switcheroo_compiler.ops.shape.utils import _extract_shape_or_seq


@dispatch_eager("TopK")
def top_k(operand: Tensor, k: int) -> OpValue:
    """Return the top `k` largest values and their indices along the last dimension.

    Args:
        operand (Tensor): Input tensor searched along its trailing axis.
        k (int): Number of top elements to select.

    Returns:
        OpValue: Pair `(values, indices)` with trailing dimension size `k`.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    out_shape_list = list(operand.shape) if operand.shape else []
    if out_shape_list:
        out_shape_list[-1] = k
    out_shape = tuple(out_shape_list)
    inputs = [operand]
    val_node = _pt_mod._emit_shape_node("TopK", inputs, {"k": k, "return_indices": False}, out_shape, operand.dtype)
    idx_node = _pt_mod._emit_shape_node("TopK", inputs, {"k": k, "return_indices": True}, out_shape, DType.Int32)
    return val_node, idx_node


def argsort(
    operand: Tensor,
    axis: int = -1,
    is_stable: bool = True,
) -> OpValue:
    """Return integer indices that sort `operand` along `axis`.

    Args:
        operand (Tensor): Input tensor whose sort order is computed.
        axis (int): Dimension along which to sort.
        is_stable (bool): Whether equal elements preserve their relative order.

    Returns:
        OpValue: `Int32` index tensor matching `operand.shape`.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    if config.eager_mode:
        backend = _pt_mod.get_active_backend()
        kind = "stable" if is_stable else "quicksort"
        data = backend.execute_op(
            "ArgSort", (operand.data if type(operand).__name__ == "Tensor" else operand), axis=axis, kind=kind
        )
        return Tensor(data, TensorConfig(operand.shape, DType.Int32, operand.device))
    inputs = [operand]
    attributes = {"dimension": axis, "is_stable": is_stable}
    return _pt_mod._emit_shape_node("ArgSort", inputs, attributes, operand.shape, DType.Int32)


def sort(
    operand: Tensor,
    axis: int = -1,
    is_stable: bool = True,
) -> OpValue:
    """Return a sorted copy of `operand` along `axis`.

    Args:
        operand (Tensor): Input tensor to sort.
        axis (int): Dimension along which elements are ordered.
        is_stable (bool): Whether equal elements preserve their relative order.

    Returns:
        OpValue: Sorted tensor with the same shape and dtype as `operand`.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    if config.eager_mode:
        backend = _pt_mod.get_active_backend()
        kind = "stable" if is_stable else "quicksort"
        data = backend.execute_op(
            "Sort", (operand.data if type(operand).__name__ == "Tensor" else operand), axis=axis, kind=kind
        )
        return Tensor(
            backend.array(data),
            TensorConfig(extract_shape(backend.array(data)), operand.dtype, operand.device),
        )
    inputs = [operand]
    attributes = {"dimension": axis, "is_stable": is_stable}
    return _pt_mod._emit_shape_node("Sort", inputs, attributes, operand.shape, operand.dtype)


@register_op("Lexsort")
class Lexsort(OpDef):
    """Schema definition performing indirect stable sort using a sequence of sort keys."""

    op_name = "Lexsort"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer index array shape produced by lexical indirect sorting.

        Args:
            *args (OpValue): Sequence or stacked tensor of sort keys.
            **kwargs (OpValue): Optional keyword arguments including `keys`.

        Returns:
            OpValue: Shape of a single key slice along the primary key axis.
        """
        keys = args[0] if len(args) > 0 else kwargs.get("keys")
        if isinstance(keys, (list, tuple)):
            return _extract_shape_or_seq(keys[0]) if keys else ()
        in_shape = _extract_shape_or_seq(keys)
        if len(in_shape) > 0:
            return in_shape[1:] if len(in_shape) > 1 else ()
        return ()


def lexsort(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Perform an indirect stable sort using a sequence of keys.

    Args:
        *args (OpValue): Sequence or tensor of sort keys.
        **kwargs (OpValue): Additional keyword options such as `axis`.

    Returns:
        OpValue: Integer index tensor giving lexicographical sort order.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Lexsort", *args, **kwargs)


@register_op("Searchsorted")
class Searchsorted(OpDef):
    """Schema definition finding insertion indices into a sorted sequence."""

    op_name = "Searchsorted"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer insertion index shape by broadcasting leading dimensions with `values`.

        Args:
            *args (OpValue): Sorted sequence followed by candidate `values`.
            **kwargs (OpValue): Unused keyword attributes such as `side` or `sorter`.

        Returns:
            OpValue: Broadcasted shape of insertion indices matching `values`.
        """
        del kwargs
        seq = args[0] if len(args) > 0 else None
        v = args[1] if len(args) > 1 else None
        v_shape: tuple[int, ...] = extract_shape(v)
        seq_shape: tuple[int, ...] = extract_shape(seq)
        if not v_shape and not seq_shape:
            return ()
        if not seq_shape or len(seq_shape) <= 1:
            return tuple(v_shape)
        leading = broadcast_shapes(seq_shape[:-1], v_shape[:-1]) if len(v_shape) > 1 else seq_shape[:-1]
        v_dim = (v_shape[-1],) if len(v_shape) >= 1 else ()
        return tuple(leading) + tuple(v_dim)


def searchsorted(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Find indices where elements should be inserted into a sorted array to preserve order.

    Args:
        *args (OpValue): Sorted 1D/ND sequence and target values to locate.
        **kwargs (OpValue): Optional search options such as `side` and `sorter`.

    Returns:
        OpValue: Insertion index tensor matching candidate values shape.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Searchsorted", *args, **kwargs)


@register_op("SortComplex")
class SortComplex(OpDef):
    """Schema definition sorting complex arrays by real part then imaginary part."""

    op_name = "SortComplex"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer output shape preserving the input array dimensions.

        Args:
            *args (OpValue): Complex or real input array operand.
            **kwargs (OpValue): Unused keyword operation attributes.

        Returns:
            OpValue: Output shape identical to the input operand shape.
        """
        del kwargs
        if not args or args[0] is None:
            return ()
        a = args[0]
        return extract_shape(a)


def sort_complex(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Sort a complex array using the real part first, then the imaginary part.

    Args:
        *args (OpValue): Complex or real input array operand.
        **kwargs (OpValue): Additional keyword attributes forwarded to `SortComplex`.

    Returns:
        OpValue: Sorted complex tensor output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("SortComplex", *args, **kwargs)
