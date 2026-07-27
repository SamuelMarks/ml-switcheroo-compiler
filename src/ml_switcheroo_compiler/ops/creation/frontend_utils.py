"""Constants & Creation Operations."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state

if TYPE_CHECKING:
    pass


def _emit_creation_node(
    op_type: str,
    shape: Sequence[int],
    dtype: DType,
    attributes: dict | None = None,
) -> Tensor:
    """Emit a creation node to the IR graph.

    Args:
        op_type (str): The op_type parameter for the operation.
        shape (Sequence[int]): The target shape.
        dtype (DType): The target data type.
        attributes (dict | None): The attributes parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[],
        attributes={**(attributes or {}), "dtype": dtype.value},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)

    proxy = ProxyTensor(
        id=out_id,
        shape=shape,
        dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
    )
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


def _emit_constant_node(
    value: object,
    dtype: DType,
) -> Tensor:
    """Emit a Constant node to the IR graph.

    Args:
        value (object): The value parameter for the operation.
        dtype (DType): The target data type.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if not global_tracing_state.is_tracing:
        msg = "Cannot emit Constant node outside of a tracing context."
        raise RuntimeError(msg)

    out_id = str(uuid.uuid4())
    val_arr = get_active_backend().array(value, dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)))
    shape = tuple(val_arr.shape)

    node = LogicalNode(
        id=out_id,
        op_type="Constant",
        inputs=[],
        attributes={"value": val_arr.tolist() if val_arr.ndim > 0 else val_arr.item()},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)

    proxy = ProxyTensor(
        id=out_id,
        shape=shape,
        dtype=dtype.value if hasattr(dtype, "value") else getattr(dtype, "name", str(dtype)),
    )
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


@register_op("FromDlpack")
class FromDlpack(OpDef):
    """Create a switcheroo array from a DLPack capsule."""

    op_name = "FromDlpack"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Frompyfunc")
class Frompyfunc(OpDef):
    """Takes an arbitrary Python function and returns a NumPy ufunc."""

    op_name = "Frompyfunc"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Geomspace")
class Geomspace(OpDef):
    """Return numbers spaced evenly on a log scale (a geometric progression)."""

    op_name = "Geomspace"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        if "num" in kwargs:
            return (kwargs["num"],)
        if len(args) > 2:
            return (args[2],)
        return (50,)  # Default


@register_op("Geometric")
class Geometric(OpDef):
    """Draw samples from the geometric distribution."""

    op_name = "Geometric"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        if "size" in kwargs and kwargs["size"] is not None:
            return tuple(kwargs["size"]) if isinstance(kwargs["size"], (list, tuple)) else (kwargs["size"],)
        return ()
