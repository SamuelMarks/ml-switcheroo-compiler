# ruff: noqa
"""Frontend reductions ops."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import WindowConfig


class ReduceWindow:
    """ReduceWindow class."""

    def infer_shape(self, *args, **kwargs) -> tuple:
        """infer_shape function.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result.
        """
        return ()


from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, global_tracing_state


def _emit_reduction_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shape: tuple,
    out_dtype: DType,
) -> Tensor:
    """Evaluate _emit_reduction_node operation.

    Args:
        op_type (str): The op_type parameter.
        inputs (Sequence): The inputs parameter.
        attrs (dict): The attrs parameter.
        out_shape (tuple): The out_shape parameter.
        out_dtype (DType): The out_dtype parameter.

    Returns:
        Tensor: Result.
    """
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=out_shape,
    )
    global_tracing_state.add_node(node)

    dtype_val = getattr(out_dtype, "value", out_dtype)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype_val)
    return Tensor(proxy, TensorConfig(out_shape, out_dtype, inputs[0].device))


def _reduce_window_eager(operand: Tensor, init_value: Tensor | float, computation: str, window_config: WindowConfig) -> Tensor:
    """Evaluate _reduce_window_eager operation.

    Args:
        operand (Tensor): The operand parameter.
        init_value (object): The init_value parameter.
        computation (str): The computation parameter.
        window_config (WindowConfig): The window_config parameter.

    Returns:
        Tensor: Result.
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
    return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, operand.dtype, operand.device))


def _build_reduce_window_attributes(init_value: Tensor | float, computation: str, window_config: WindowConfig) -> dict:
    """Evaluate _build_reduce_window_attributes operation.

    Args:
        init_value (object): The init_value parameter.
        computation (str): The computation parameter.
        window_config (WindowConfig): The window_config parameter.

    Returns:
        dict: Result.
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


def _reduce_window_trace(operand: Tensor, init_value: Tensor | float, computation: str, window_config: WindowConfig) -> Tensor:
    """Evaluate _reduce_window_trace operation.

    Args:
        operand (Tensor): The operand parameter.
        init_value (object): The init_value parameter.
        computation (str): The computation parameter.
        window_config (WindowConfig): The window_config parameter.

    Returns:
        Tensor: Result.
    """
    inputs = [operand]
    if isinstance(init_value, Tensor):
        inputs.append(init_value)

    attributes = _build_reduce_window_attributes(init_value, computation, window_config)

    rw_op = ReduceWindow()
    out_shape = rw_op.infer_shape(operand, init_value, computation, window_config)

    return _emit_reduction_node("ReduceWindow", inputs, attributes, out_shape, operand.dtype)


def reduce_window(
    operand: Tensor,
    init_value: Tensor | float,
    computation: str,
    window_config: WindowConfig,
) -> Tensor:
    """Apply a reduction function over a sliding window of the input.

    Args:
        operand (Tensor): The operand parameter.
        init_value (object): The init_value parameter.
        computation (str): The computation parameter.
        window_config (WindowConfig): The window_config parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        return _reduce_window_eager(operand, init_value, computation, window_config)
    return _reduce_window_trace(operand, init_value, computation, window_config)
