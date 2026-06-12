"""Reduction Operations."""

from typing import Optional

import uuid
from typing import Union
from collections.abc import Sequence
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def _emit_reduction_node(
    input: Tensor,
    op_type: str,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out_dtype: DType = None,
    extra_attrs: dict = None,
) -> Tensor:
    """Emit a reduction node to the IR graph."""
    if not _tracer.is_tracing:
        raise RuntimeError(f"Cannot emit {op_type} node outside of a tracing context.")

    out_id = str(uuid.uuid4())
    attrs = {"keepdims": int(keepdims)}
    if axis is not None:
        attrs["axes"] = [axis] if isinstance(axis, int) else list(axis)
    if extra_attrs:
        attrs.update(extra_attrs)

    out_shape = list(input.shape)
    if axis is not None:
        axes = [axis] if isinstance(axis, int) else list(axis)
        axes = [a if a >= 0 else len(out_shape) + a for a in axes]
        if keepdims:
            for a in axes:
                out_shape[a] = 1
        else:
            out_shape = [s for i, s in enumerate(out_shape) if i not in axes]
    else:
        out_shape = [1] * len(out_shape) if keepdims else []

    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[input.data.id],
        attributes=attrs,
        shape_metadata=tuple(out_shape),
    )
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=tuple(out_shape), dtype=out_dtype.value)
    return Tensor(
        data=proxy, shape=tuple(out_shape), dtype=out_dtype, device=input.device
    )


def sum(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes sum of input."""
    if config.eager_mode:
        data = np.sum(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_reduction_node(input, "ReduceSum", axis, keepdims, input.dtype)


def prod(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes prod of input."""
    if config.eager_mode:
        data = np.prod(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_reduction_node(input, "ReduceProd", axis, keepdims, input.dtype)


def mean(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes mean of input."""
    if config.eager_mode:
        data = np.mean(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_reduction_node(input, "ReduceMean", axis, keepdims, input.dtype)


def variance(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: int = 0,
    keepdims: bool = False,
) -> Tensor:
    """Computes variance of input."""
    if config.eager_mode:
        data = np.var(input.data, axis=axis, ddof=correction, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_reduction_node(
            input,
            "ReduceVariance",
            axis,
            keepdims,
            input.dtype,
            {correction: correction},
        )


def std(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: int = 0,
    keepdims: bool = False,
) -> Tensor:
    """Computes std of input."""
    if config.eager_mode:
        data = np.std(input.data, axis=axis, ddof=correction, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_reduction_node(
            input, "ReduceStd", axis, keepdims, input.dtype, {correction: correction}
        )


def max(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes max of input."""
    if config.eager_mode:
        data = np.max(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_reduction_node(input, "ReduceMax", axis, keepdims, input.dtype)


def min(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes min of input."""
    if config.eager_mode:
        data = np.min(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_reduction_node(input, "ReduceMin", axis, keepdims, input.dtype)


def argmax(input: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Computes argmax of input."""
    if config.eager_mode:
        data = np.argmax(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, DType.Int64, input.device)
    else:
        return _emit_reduction_node(input, "Argmax", axis, keepdims, DType.Int64)


def argmin(input: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Computes argmin of input."""
    if config.eager_mode:
        data = np.argmin(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, DType.Int64, input.device)
    else:
        return _emit_reduction_node(input, "Argmin", axis, keepdims, DType.Int64)


def all(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes all of input."""
    if config.eager_mode:
        data = np.all(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, DType.Bool, input.device)
    else:
        return _emit_reduction_node(input, "All", axis, keepdims, DType.Bool)


def any(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes any of input."""
    if config.eager_mode:
        data = np.any(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, DType.Bool, input.device)
    else:
        return _emit_reduction_node(input, "Any", axis, keepdims, DType.Bool)


def logsumexp(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes logsumexp of input."""
    if config.eager_mode:
        from ml_switcheroo.core.errors import UnimplementedMathError

        raise UnimplementedMathError("No direct NumPy equivalent for logsumexp.")
    else:
        return _emit_reduction_node(input, "Logsumexp", axis, keepdims, input.dtype)


def count_nonzero(
    input: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes count_nonzero of input."""
    if config.eager_mode:
        data = np.count_nonzero(input.data, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, DType.Int64, input.device)
    else:
        return _emit_reduction_node(input, "CountNonzero", axis, keepdims, DType.Int64)


def norm(
    input: Tensor,
    ord: Optional[Union[int, float, str]] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Computes norm of input."""
    if config.eager_mode:
        data = np.linalg.norm(input.data, ord=ord, axis=axis, keepdims=keepdims)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_reduction_node(
            input, "Norm", axis, keepdims, input.dtype, {ord: ord}
        )


def cumsum(
    x: object, axis: Optional[int] = None, dtype: Optional[object] = None
) -> object:
    """Docstring."""
    import numpy as np
    from ml_switcheroo.core.tensor import Tensor

    arr = np.cumsum((x.data if hasattr(x, "device") else x), axis=axis, dtype=dtype)
    return Tensor(arr, arr.shape, x.dtype, getattr(x, "device", None))
