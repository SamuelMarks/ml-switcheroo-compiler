"""Frontend reductions ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor

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
    return Tensor(data=proxy, shape=out_shape, dtype=out_dtype, device=inputs[0].device)


@dataclass
class ReduceWindowConfig:
    """Reduce window config."""

    window_dimensions: Sequence[int]
    window_strides: Sequence[int] | None = None
    padding: Sequence[tuple[int, int]] | None = None
    base_dilation: Sequence[int] | None = None
    window_dilation: Sequence[int] | None = None


def reduce_window(
    operand: Tensor,
    init_value: Tensor | float,
    computation: str,
    window_config: ReduceWindowConfig,
) -> Tensor:
    """Applies a reduction function over a sliding window of the input.

    Args:
        operand (Tensor): The input tensor
        init_value (Tensor | float | int): The initial value for the reduction
        computation (str): The reduction to apply (e.g. 'max', 'sum')
        window_config (ReduceWindowConfig): Configuration parameters for the window

    Returns:
    Tensor: The reduced tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for reduce_window"
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        raise UnimplementedMathError(msg)

    inputs = [operand]
    # In a full implementation init_value might be a tensor input,
    # but for simplicty we'll put it in attributes if it's a scalar.
    if isinstance(init_value, Tensor):
        inputs.append(init_value)
        init_val_attr = None
    else:
        init_val_attr = init_value

    attributes = {
        "computation": computation,
        "window_dimensions": window_config.window_dimensions,
        "window_strides": window_config.window_strides,
        "padding": window_config.padding,
        "base_dilation": window_config.base_dilation,
        "window_dilation": window_config.window_dilation,
    }
    if init_val_attr is not None:
        attributes["init_value"] = init_val_attr

    # We do a rough shape inference here to satisfy the IR node shape metadata requirement
    # Real shape inference is handled by the OpDef
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindow

    rw_op = ReduceWindow()

    out_shape = rw_op.infer_shape(operand, init_value, computation, window_config)

    return _emit_reduction_node("ReduceWindow", inputs, attributes, out_shape, operand.dtype)


def psum(x: Tensor, axis_name: str) -> Tensor:
    """Computes an all-reduce sum over the specified mapped axis.

    Args:
        x (Tensor): The input tensor
        axis_name (str): The axis to map over

    Returns:
    Tensor: The reduced tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        msg = "No direct numpy for psum"
        raise UnimplementedMathError(msg)

    return _emit_reduction_node("Psum", [x], {"axis_name": axis_name}, x.shape, x.dtype)


def pmean(x: Tensor, axis_name: str) -> Tensor:
    """Computes an all-reduce mean over the specified mapped axis.

    Args:
        x (Tensor): The input tensor
        axis_name (str): The axis to map over

    Returns:
    Tensor: The reduced tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        msg = "No direct numpy for pmean"
        raise UnimplementedMathError(msg)

    return _emit_reduction_node("Pmean", [x], {"axis_name": axis_name}, x.shape, x.dtype)


def segment_sum(
    data: Tensor,
    segment_ids: Tensor,
    num_segments: int | None = None,
) -> Tensor:
    """Computes the sum of tensor elements grouped by segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
    Tensor: The segmented sum tensor

    Raises:
    UnimplementedMathError: If called in eager mode
    """
    if config.eager_mode:
        msg = "No direct numpy for segment_sum"
        from ml_switcheroo_compiler.core.errors import UnimplementedMathError

        raise UnimplementedMathError(msg)

    inputs = [data, segment_ids]
    attributes = {}
    if num_segments is not None:
        attributes["num_segments"] = num_segments

    return _emit_reduction_node(
        "SegmentSum",
        inputs,
        attributes,
        (),  # Placeholder shape
        data.dtype,
    )
