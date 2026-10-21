"""Implementation of pooling operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.nn.pooling`)."""

from __future__ import annotations

import math
import typing

import pydantic

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import divide
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.creation import ones_like
from ml_switcheroo_compiler.ops.reductions import reduce_window

if typing.TYPE_CHECKING:
    pass


def _prepare_pool_config(
    rank: int,
    spatial_rank: int,
    window_shape: TensorOrScalar,
    strides: TensorOrScalar,
    padding: str | tuple[int, ...] | tuple[tuple[int, int], ...],
) -> WindowConfig:
    """Prepare and validates the window configuration for pooling operations.

    Args:
        rank (int): The total rank of the input tensor.
        spatial_rank (int): The number of spatial dimensions.
        window_shape (TensorOrScalar): The shape of the pooling window.
        strides (TensorOrScalar): The strides for the pooling window.
        padding (str | tuple[int, ...] | tuple[tuple[int, int], ...]): The padding configuration to apply.

    Returns:
        WindowConfig: A configuration containing the processed dimensions, strides, and padding.
    """
    pad_dims = max(0, rank - spatial_rank - 1)

    full_window_shape = (1,) * pad_dims + tuple(getattr(getattr(window_shape, "__iter__")(), "__iter__")()) + (1,) if rank > spatial_rank else tuple(getattr(getattr(window_shape, "__iter__")(), "__iter__")())
    full_strides = (1,) * pad_dims + tuple(getattr(getattr(strides, "__iter__")(), "__iter__")()) + (1,) if rank > spatial_rank else tuple(getattr(getattr(strides, "__iter__")(), "__iter__")())

    full_padding: str | tuple[tuple[int, int], ...]
    if isinstance(padding, str):
        full_padding = padding
    else:
        norm_pad: tuple[tuple[int, int], ...] = tuple((p[0], p[1]) if isinstance(p, tuple) else (int(p), int(p)) for p in padding)
        full_padding = ((0, 0),) * pad_dims + norm_pad + ((0, 0),) if rank > spatial_rank else norm_pad

    return WindowConfig(
        window_dimensions=full_window_shape,
        window_strides=full_strides,
        padding=full_padding,
    )


def _compute_pool_out_shape(in_shape: tuple[int, ...], config: WindowConfig) -> list[int]:
    """Compute the expected output shape for a pooling operation.

    Args:
        in_shape (tuple[int, ...]): The shape of the input tensor.
        config (WindowConfig): The configuration for the pooling window.

    Returns:
        list[int]: A list of integers representing the expected output shape.
    """
    out_shape = []
    if isinstance(config.padding, str):
        for d, w, s in zip(in_shape, config.window_dimensions, getattr(config.window_strides, "__iter__")()):
            if config.padding == "SAME":
                out_shape.append((d + s - 1) // s)
            else:
                out_shape.append((d - w) // s + 1)
    else:
        for d, w, s, p in zip(in_shape, config.window_dimensions, getattr(config.window_strides, "__iter__")(), getattr(config.padding, "__iter__")()):
            p_sum = p[0] + p[1] if isinstance(p, tuple) else int(p)
            out_shape.append((d + p_sum - w) // s + 1)
    return out_shape


def _max_pool_with_indices(
    operand: Tensor,
    window_shape: tuple[int, ...],
    strides: tuple[int, ...],
    padding: str | tuple[tuple[int, int], ...],
    config: WindowConfig,
) -> tuple[Tensor, Tensor]:
    """Compute the max pooling operation and returns the resulting tensor along with its indices.

    Args:
        operand (Tensor): The input tensor to be pooled.
        window_shape (tuple[int, ...]): The shape of the pooling window.
        strides (tuple[int, ...]): The strides of the pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding mode or explicit padding values.
        config (WindowConfig): The configuration containing processed window parameters.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the pooled output tensor and a tensor of the corresponding indices.
    """
    out_shape = _compute_pool_out_shape(operand.shape, config)

    from ml_switcheroo_compiler.core.config import config as _config

    if _config.eager_mode:
        backend = get_active_backend()
        raw_res = backend.execute_op(
            "MaxPoolWithIndices",
            operand.data,
            window_shape=window_shape,
            strides=strides,
            padding=padding,
        )
        data = getattr(raw_res, "__getitem__")(0)
        indices = getattr(raw_res, "__getitem__")(1)
        return (
            Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device)),
            Tensor(backend.array(indices), TensorConfig(tuple(out_shape), DType.Int64, operand.device)),
        )

    from ml_switcheroo_compiler.ops.reductions.frontend_utils import _emit_reduction_node

    pooled = _emit_reduction_node(
        "MaxPoolWithIndices",
        [operand],
        {"window_shape": window_shape, "strides": strides, "padding": padding},
        tuple(out_shape),
        operand.dtype,
    )
    indices_node = _emit_reduction_node(
        "MaxPoolWithIndices_Indices",
        [operand],
        {"window_shape": window_shape, "strides": strides, "padding": padding},
        tuple(out_shape),
        DType.Int64,
    )
    return pooled, indices_node


def max_pool(
    operand: Tensor,
    window_shape: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Apply a max pooling operation over the input tensor.

    Args:
        operand (Tensor): The input tensor to pool.
        window_shape (tuple[int, ...]): The dimensions of the pooling window.
        strides (tuple[int, ...] | None): The strides of the pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding mode ('VALID' or 'SAME') or explicit padding configuration.
        return_indices (bool): Whether to return the indices of the maximum values alongside the pooled tensor.

    Returns:
        Tensor | tuple[Tensor, Tensor]: The pooled tensor, or a tuple of the pooled tensor and the indices if return_indices is True.
    """
    if strides is None:
        strides = (1,) * len(window_shape)

    config = _prepare_pool_config(len(operand.shape), len(window_shape), window_shape, strides, padding)

    if return_indices:
        return _max_pool_with_indices(operand, window_shape, strides, padding, config)

    init_val = -math.inf
    return reduce_window(operand, init_val, "max", config)


def avg_pool(
    operand: Tensor,
    window_shape: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
) -> TensorOrScalar:
    """Apply an average pooling operation over the input tensor.

    Args:
        operand (Tensor): The input tensor to pool.
        window_shape (tuple[int, ...]): The dimensions of the pooling window.
        strides (tuple[int, ...] | None): The strides of the pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding mode ('VALID' or 'SAME') or explicit padding configuration.

    Returns:
        TensorOrScalar: The resulting tensor after applying average pooling.
    """
    if strides is None:
        strides = (1,) * len(window_shape)

    config = _prepare_pool_config(len(operand.shape), len(window_shape), window_shape, strides, padding)
    init_val = 0.0

    sum_pooled = reduce_window(operand, init_val, "sum", config)

    ones = ones_like(operand)
    counts = reduce_window(ones, 0.0, "sum", config)

    return divide(sum_pooled, counts)


def pool1d(
    operand: Tensor,
    window_shape: int,
    strides: int | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    pool_mode: str = "max",
) -> TensorOrScalar:
    """Apply a 1D pooling operation over the input tensor.

    Args:
        operand (Tensor): The operand parameter.
        window_shape (int): The window_shape parameter.
        strides (int | None): The strides parameter.
        padding (str | tuple[tuple[int, int], ...]): The padding parameter.
        pool_mode (str): The pool_mode parameter.

    Returns:
        TensorOrScalar: Result.

    Raises:
        ValueError: An exception.
    """
    shape = (window_shape,)
    stride = (strides,) if strides is not None else None
    if pool_mode == "max":
        return max_pool(operand, shape, stride, padding)
    if pool_mode == "avg":
        return avg_pool(operand, shape, stride, padding)
    raise ValueError(f"Unknown pool_mode: {pool_mode}")


def pool2d(
    operand: Tensor,
    window_shape: tuple[int, int],
    strides: tuple[int, int] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    pool_mode: str = "max",
) -> TensorOrScalar:
    """Apply a 2D pooling operation over the input tensor.

    Args:
        operand (Tensor): The operand parameter.
        window_shape (tuple[int, int]): The window_shape parameter.
        strides (tuple[int, int] | None): The strides parameter.
        padding (str | tuple[tuple[int, int], ...]): The padding parameter.
        pool_mode (str): The pool_mode parameter.

    Returns:
        TensorOrScalar: Result.

    Raises:
        ValueError: An exception.
    """
    if pool_mode == "max":
        return max_pool(operand, window_shape, strides, padding)
    if pool_mode == "avg":
        return avg_pool(operand, window_shape, strides, padding)
    raise ValueError(f"Unknown pool_mode: {pool_mode}")


def pool3d(
    operand: Tensor,
    window_shape: tuple[int, int, int],
    strides: tuple[int, int, int] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    pool_mode: str = "max",
) -> TensorOrScalar:
    """Apply a 3D pooling operation over the input tensor.

    Args:
        operand (Tensor): The operand parameter.
        window_shape (tuple[int, int, int]): The window_shape parameter.
        strides (tuple[int, int, int] | None): The strides parameter.
        padding (str | tuple[tuple[int, int], ...]): The padding parameter.
        pool_mode (str): The pool_mode parameter.

    Returns:
        TensorOrScalar: Result.

    Raises:
        ValueError: An exception.
    """
    if pool_mode == "max":
        return max_pool(operand, window_shape, strides, padding)
    if pool_mode == "avg":
        return avg_pool(operand, window_shape, strides, padding)
    raise ValueError(f"Unknown pool_mode: {pool_mode}")


def average_pool(
    inputs: Tensor,
    pool_size: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    data_format: str | None = None,
) -> TensorOrScalar:
    """Compute an average pool over the given input tensor.

    Args:
        inputs (Tensor): The input tensor to process.
        pool_size (tuple[int, ...]): The size of the pooling window.
        strides (tuple[int, ...] | None): The stride for the pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding configuration.
        data_format (str | None): An optional string specifying the data format (currently ignored).

    Returns:
        TensorOrScalar: The resulting pooled tensor.
    """
    del data_format
    return avg_pool(inputs, pool_size, strides, padding)


class SpatialConfig(pydantic.BaseModel):
    """Configuration for spatial dimensions in a pooling operation.

    Args:
        ksize (int | tuple[int, ...] | list[int] | None): The kernel size for pooling.
        strides (int | tuple[int, ...] | list[int] | None): The stride dimensions for pooling.
        padding (str | int | tuple[int, ...] | list[int] | None): The padding configuration.
        dilation_rate (int | tuple[int, ...] | list[int] | None): The dilation rate for pooling. Defaults to None.
        data_format (str | None): The data format string, e.g., 'NHWC'. Defaults to None.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ksize: int | tuple[int, ...] | list[int] | None = None
    strides: int | tuple[int, ...] | list[int] | None = None
    padding: str | int | tuple[int, ...] | list[int] | None = None
    dilation_rate: int | tuple[int, ...] | list[int] | None = None
    data_format: str | None = None


class PoolingBehaviorConfig(pydantic.BaseModel):
    """Configuration detailing specific behavior for pooling algorithms.

    Args:
        pooling_type (str): The type of pooling ('MAX' or 'AVG'). Defaults to 'MAX'.
        pooling_ratio (list[float] | tuple[float, ...] | None): The pooling ratio used for fractional pooling. Defaults to None.
        output_dtype (DType | str | None): The desired output data type. Defaults to None.
        include_batch_in_index (bool): Whether to include the batch dimension in indices. Defaults to False.
        pseudo_random (bool): Whether to use pseudo-randomness. Defaults to False.
        overlapping (bool): Whether to allow overlapping regions. Defaults to False.
        deterministic (bool): Whether the pooling should be deterministic. Defaults to False.
        seed (int): The primary seed for random operations. Defaults to 0.
        seed2 (int): The secondary seed for random operations. Defaults to 0.
        name (str | None): An optional name for the operation. Defaults to None.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    pooling_type: str = "MAX"
    pooling_ratio: list[float] | tuple[float, ...] | None = None
    output_dtype: DType | str | None = None
    include_batch_in_index: bool = False
    pseudo_random: bool = False
    overlapping: bool = False
    deterministic: bool = False
    seed: int = 0
    seed2: int = 0
    name: str | None = None


class PoolingConfig(pydantic.BaseModel):
    """Comprehensive configuration wrapping spatial and behavioral settings for a pooling operation.

    Args:
        window (SpatialConfig): The spatial dimension configuration.
        behavior (PoolingBehaviorConfig): The specific pooling algorithm configuration.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    window: SpatialConfig
    behavior: PoolingBehaviorConfig


def _extract_window_args(
    config: PoolingConfig,
) -> tuple[tuple[int, ...], tuple[int, ...] | None, str | tuple[tuple[int, int], ...]]:
    """Extract typed window_shape, strides, and padding from a PoolingConfig.

    Args:
        config (PoolingConfig): Pooling configuration model.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...] | None, str | tuple[tuple[int, int], ...]]: Extracted parameters.
    """
    from ml_switcheroo_compiler.backends.registry import as_shape, as_str

    return as_shape(config.window.ksize), as_shape(config.window.strides), as_str(config.window.padding)


def _as_tensor(val: TensorOrScalar) -> Tensor:
    """Coerce a TensorOrScalar value into a Tensor instance.

    Args:
        val (TensorOrScalar): Input tensor or array-like value.

    Returns:
        Tensor: Wrapped or existing Tensor.
    """
    from ml_switcheroo_compiler.core.tensor import as_tensor_obj

    return as_tensor_obj(val)


def avg_pool1d(value: TensorOrScalar, config: PoolingConfig) -> TensorOrScalar:
    """Execute a 1D average pooling operation based on the given configuration.

    Args:
        value (TensorOrScalar): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration containing window shapes and behavior.

    Returns:
        TensorOrScalar: The resulting 1D average pooled data.
    """
    ws, st, pad = _extract_window_args(config)
    return avg_pool(_as_tensor(value), window_shape=ws, strides=st, padding=pad)


def avg_pool2d(value: TensorOrScalar, config: PoolingConfig) -> TensorOrScalar:
    """Execute a 2D average pooling operation based on the given configuration.

    Args:
        value (TensorOrScalar): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        TensorOrScalar: The resulting 2D average pooled data.
    """
    ws, st, pad = _extract_window_args(config)
    return avg_pool(_as_tensor(value), window_shape=ws, strides=st, padding=pad)


def avg_pool3d(value: TensorOrScalar, config: PoolingConfig) -> TensorOrScalar:
    """Execute a 3D average pooling operation based on the given configuration.

    Args:
        value (TensorOrScalar): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        TensorOrScalar: The resulting 3D average pooled data.
    """
    ws, st, pad = _extract_window_args(config)
    return avg_pool(_as_tensor(value), window_shape=ws, strides=st, padding=pad)


def max_pool1d(inputs: TensorOrScalar, config: PoolingConfig) -> TensorOrScalar:
    """Execute a 1D max pooling operation based on the given configuration.

    Args:
        inputs (TensorOrScalar): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        TensorOrScalar: The resulting 1D max pooled data.
    """
    ws, st, pad = _extract_window_args(config)
    return max_pool(_as_tensor(inputs), window_shape=ws, strides=st, padding=pad)


def max_pool2d(inputs: TensorOrScalar, config: PoolingConfig) -> TensorOrScalar:
    """Execute a 2D max pooling operation based on the given configuration.

    Args:
        inputs (TensorOrScalar): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        TensorOrScalar: The resulting 2D max pooled data.
    """
    ws, st, pad = _extract_window_args(config)
    return max_pool(_as_tensor(inputs), window_shape=ws, strides=st, padding=pad)


def max_pool3d(inputs: TensorOrScalar, config: PoolingConfig) -> TensorOrScalar:
    """Execute a 3D max pooling operation based on the given configuration.

    Args:
        inputs (TensorOrScalar): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        TensorOrScalar: The resulting 3D max pooled data.
    """
    ws, st, pad = _extract_window_args(config)
    return max_pool(_as_tensor(inputs), window_shape=ws, strides=st, padding=pad)


def max_pool_with_argmax(
    input: TensorOrScalar,
    config: PoolingConfig,
) -> TensorOrScalar:
    """Execute a max pooling operation and additionally compute the argmax indices based on the configuration.

    Args:
        input (TensorOrScalar): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        TensorOrScalar: A tuple containing the pooled output and a tensor representing the argmax indices.
    """
    ws, st, pad = _extract_window_args(config)
    return max_pool(_as_tensor(input), window_shape=ws, strides=st, padding=pad, return_indices=True)


def fractional_avg_pool(
    value: Tensor,
    config: PoolingConfig,
) -> TensorOrScalar:
    """Perform fractional average pooling on the input tensor based on the given configuration.

    Args:
        value (Tensor): The input tensor to undergo fractional average pooling.
        config (PoolingConfig): The comprehensive pooling configuration specifying the pooling ratio.

    Returns:
        TensorOrScalar: A tuple containing the output tensor and two additional tensors indicating states.
    """
    from ml_switcheroo_compiler.core.config import config as _config

    if _config.eager_mode:
        data = get_active_backend().execute_op("FractionalAvgPool", value.data, pooling_ratio=config.behavior.pooling_ratio)
        return (
            Tensor(data, TensorConfig(tuple(getattr(data, "shape")), value.dtype, value.device)),
            Tensor([0], TensorConfig((1,), value.dtype, value.device)),
            Tensor([0], TensorConfig((1,), value.dtype, value.device)),
        )

    out_shape = getattr(FractionalAvgPool(), "infer_shape")(value, config.behavior.pooling_ratio)
    from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

    res = _emit_linalg_node(
        "FractionalAvgPool",
        [value],
        {"pooling_ratio": config.behavior.pooling_ratio},
        [tuple(out_shape)],
        [value.dtype],
    )
    return res, res, res


def fractional_max_pool(
    value: Tensor,
    config: PoolingConfig,
) -> TensorOrScalar:
    """Perform fractional max pooling on the input tensor based on the given configuration.

    Args:
        value (Tensor): The input tensor to undergo fractional max pooling.
        config (PoolingConfig): The comprehensive pooling configuration specifying the pooling ratio.

    Returns:
        TensorOrScalar: A tuple containing the output tensor and two additional tensors indicating states.
    """
    from ml_switcheroo_compiler.core.config import config as _config

    if _config.eager_mode:
        data = get_active_backend().execute_op("FractionalMaxPool", value.data, pooling_ratio=config.behavior.pooling_ratio)
        return (
            Tensor(data, TensorConfig(tuple(getattr(data, "shape")), value.dtype, value.device)),
            Tensor([0], TensorConfig((1,), value.dtype, value.device)),
            Tensor([0], TensorConfig((1,), value.dtype, value.device)),
        )

    out_shape = getattr(FractionalMaxPool(), "infer_shape")(value, config.behavior.pooling_ratio)
    from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

    res = _emit_linalg_node(
        "FractionalMaxPool",
        [value],
        {"pooling_ratio": config.behavior.pooling_ratio},
        [tuple(out_shape)],
        [value.dtype],
    )
    return res, res, res


def pool(
    input: TensorOrScalar,
    config: PoolingConfig,
) -> TensorOrScalar:
    """Execute a generic pooling operation, delegating to either average or max pooling depending on the configuration.

    Args:
        input (TensorOrScalar): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration specifying spatial dimensions and pooling type.

    Returns:
        TensorOrScalar: The resulting pooled data.
    """
    if config.behavior.pooling_type == "AVG":
        return avg_pool1d(input, config)
    return max_pool1d(input, config)


@register_op("FractionalAvgPool")
class FractionalAvgPool(OpDef):
    """Operator definition for fractional average pooling."""

    op_name = "FractionalAvgPool"

    def infer_shape(self, value: EagerTensorProtocol, pooling_ratio: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infers the output shape of a fractional average pooling operation.

        Args:
            value (EagerTensorProtocol): The input tensor.
            pooling_ratio (EagerTensorProtocol): The specified pooling ratio.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: The inferred output shape.
        """
        del pooling_ratio, kwargs
        return value.shape


@register_op("FractionalMaxPool")
class FractionalMaxPool(OpDef):
    """Operator definition for fractional max pooling."""

    op_name = "FractionalMaxPool"

    def infer_shape(self, value: EagerTensorProtocol, pooling_ratio: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infers the output shape of a fractional max pooling operation.

        Args:
            value (EagerTensorProtocol): The input tensor.
            pooling_ratio (EagerTensorProtocol): The specified pooling ratio.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: The inferred output shape.
        """
        del pooling_ratio, kwargs
        return value.shape
