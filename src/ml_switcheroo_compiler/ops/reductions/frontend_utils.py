"""Frontend reductions ops."""

from __future__ import annotations


from typing import TYPE_CHECKING

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import WindowConfig


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ml_switcheroo_compiler.core.dtype import DType


def _emit_reduction_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shape: tuple,
    out_dtype: DType,
) -> Tensor:
    """Execute _emit_reduction_node.

    Args:
        op_type (Any): Argument op_type.
        inputs (Any): Argument inputs.
        attrs (Any): Argument attrs.
        out_shape (Any): Argument out_shape.
        out_dtype (Any): Argument out_dtype.

    Returns:
    Any: The result.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)

    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    return Tensor(proxy, TensorConfig(out_shape, out_dtype, inputs[0].device))


def _reduce_window_eager(
    operand: Tensor, init_value: Tensor | float, computation: str, window_config: WindowConfig
) -> Tensor:
    """Function docstring.

    Args:
        operand: Arg.
        init_value: Arg.
        computation: Arg.
        window_config: Arg.
    """
    backend = get_active_backend()
    init_val_data = init_value.data if isinstance(init_value, Tensor) else init_value
    data = backend.execute_op(
        "ReduceWindow",
        operand.data,
        init_val_data,
        computation,
        config=window_config,
    )
    return Tensor(
        backend.array(data), TensorConfig(backend.array(data).shape, operand.dtype, operand.device)
    )


def _build_reduce_window_attributes(
    init_value: Tensor | float, computation: str, window_config: WindowConfig
) -> dict:
    """Function docstring.

    Args:
        init_value: Arg.
        computation: Arg.
        window_config: Arg.
    """
    attributes = {
        "computation": computation,
        "window_dimensions": window_config.window_dimensions,
        "window_strides": window_config.window_strides,
        "padding": window_config.padding,
        "base_dilation": window_config.base_dilation,
        "window_dilation": window_config.window_dilation,
    }
    if not isinstance(init_value, Tensor):
        attributes["init_value"] = init_value
    return attributes


def _reduce_window_trace(
    operand: Tensor, init_value: Tensor | float, computation: str, window_config: WindowConfig
) -> Tensor:
    """Function docstring.

    Args:
        operand: Arg.
        init_value: Arg.
        computation: Arg.
        window_config: Arg.
    """
    inputs = [operand]
    if isinstance(init_value, Tensor):
        inputs.append(init_value)

    attributes = _build_reduce_window_attributes(init_value, computation, window_config)

    from ml_switcheroo_compiler.ops.reductions.aggregations import ReduceWindow

    rw_op = ReduceWindow()
    out_shape = rw_op.infer_shape(operand, init_value, computation, window_config)

    return _emit_reduction_node("ReduceWindow", inputs, attributes, out_shape, operand.dtype)


def reduce_window(
    operand: Tensor,
    init_value: Tensor | float,
    computation: str,
    window_config: WindowConfig,
) -> Tensor:
    """Applies a reduction function over a sliding window of the input.

    Args:
        operand (Tensor): The input tensor
        init_value (Tensor | float | int): The initial value for the reduction
        computation (str): The reduction to apply (e.g. 'max', 'sum')
        window_config (WindowConfig): Configuration parameters for the window

    Returns:
    Tensor: The reduced tensor
    """
    if config.eager_mode:
        return _reduce_window_eager(operand, init_value, computation, window_config)
    return _reduce_window_trace(operand, init_value, computation, window_config)
