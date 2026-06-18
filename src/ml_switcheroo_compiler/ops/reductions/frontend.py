"""Frontend reductions ops."""

from __future__ import annotations
from ml_switcheroo_compiler.ops.configs import WindowConfig

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import dispatch_eager

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


def _reduce_window_eager(
    operand: Tensor, init_value: Tensor | float, computation: str, window_config: WindowConfig
) -> Tensor:
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    init_val_data = init_value.data if isinstance(init_value, Tensor) else init_value
    data = backend.execute_op(
        "ReduceWindow",
        operand.data,
        init_val_data,
        computation,
        config=window_config,
    )
    return Tensor(backend.array(data), backend.array(data).shape, operand.dtype, operand.device)


def _build_reduce_window_attributes(
    init_value: Tensor | float, computation: str, window_config: WindowConfig
) -> dict:
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
    inputs = [operand]
    if isinstance(init_value, Tensor):
        inputs.append(init_value)

    attributes = _build_reduce_window_attributes(init_value, computation, window_config)

    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindow

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


@dispatch_eager("Psum")
def psum(x: Tensor, axis_name: str) -> Tensor:
    """Computes an all-reduce sum over the specified mapped axis.

    Args:
        x (Tensor): The input tensor
        axis_name (str): The axis to map over

    Returns:
    Tensor: The reduced tensor

    """
    return _emit_reduction_node("Psum", [x], {"axis_name": axis_name}, x.shape, x.dtype)


@dispatch_eager("Pmean")
def pmean(x: Tensor, axis_name: str) -> Tensor:
    """Computes an all-reduce mean over the specified mapped axis.

    Args:
        x (Tensor): The input tensor
        axis_name (str): The axis to map over

    Returns:
    Tensor: The reduced tensor

    """
    return _emit_reduction_node("Pmean", [x], {"axis_name": axis_name}, x.shape, x.dtype)


@dispatch_eager("SegmentSum")
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

    """
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


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
) -> Tensor:
    """Connectionist Temporal Classification Loss.

    Args:
        log_probs (Tensor): Log probabilities.
        targets (Tensor): Targets.
        input_lengths (Tensor): Input lengths.
        target_lengths (Tensor): Target lengths.

    Returns:
        Tensor: The loss.
    """
    inputs = [log_probs, targets, input_lengths, target_lengths]
    return _emit_reduction_node("CTCLoss", inputs, {}, (), log_probs.dtype)


def fractional_max_pool2d(
    operand: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    """Fractional max pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape = list(operand.shape)
    if len(out_shape) >= 2:
        out_shape[-2] = output_size[0]
        out_shape[-1] = output_size[1]
    return _emit_reduction_node(
        "FractionalMaxPool2D",
        [operand],
        {"output_size": output_size},
        tuple(out_shape),
        operand.dtype,
    )


def adaptive_avg_pool2d(
    operand: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    """Adaptive average pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape = list(operand.shape)
    if len(out_shape) >= 2:
        out_shape[-2] = output_size[0]
        out_shape[-1] = output_size[1]
    return _emit_reduction_node(
        "AdaptiveAvgPool2D",
        [operand],
        {"output_size": output_size},
        tuple(out_shape),
        operand.dtype,
    )


def adaptive_max_pool2d(
    operand: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    """Adaptive max pooling 2D.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.

    Returns:
        Tensor: The pooled tensor.
    """
    out_shape = list(operand.shape)
    if len(out_shape) >= 2:
        out_shape[-2] = output_size[0]
        out_shape[-1] = output_size[1]
    return _emit_reduction_node(
        "AdaptiveMaxPool2D",
        [operand],
        {"output_size": output_size},
        tuple(out_shape),
        operand.dtype,
    )


def unfold(
    operand: Tensor,
    kernel_size: tuple[int, int],
) -> Tensor:
    """Unfold (Im2Col) operator.

    Args:
        operand (Tensor): The input tensor.
        kernel_size (tuple[int, int]): The kernel size.

    Returns:
        Tensor: The unfolded tensor.
    """
    return _emit_reduction_node(
        "Unfold", [operand], {"kernel_size": kernel_size}, (), operand.dtype
    )


def fold(
    operand: Tensor,
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
) -> Tensor:
    """Fold (Col2Im) operator.

    Args:
        operand (Tensor): The input tensor.
        output_size (tuple[int, int]): The output size.
        kernel_size (tuple[int, int]): The kernel size.

    Returns:
        Tensor: The folded tensor.
    """
    return _emit_reduction_node(
        "Fold",
        [operand],
        {"output_size": output_size, "kernel_size": kernel_size},
        (),
        operand.dtype,
    )
