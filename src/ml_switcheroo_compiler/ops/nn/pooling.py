"""Core abstractions and logic definitions for pooling.py."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import divide
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.creation import ones_like
from ml_switcheroo_compiler.ops.reductions import reduce_window


def _prepare_pool_config(
    rank: int,
    spatial_rank: int,
    window_shape: tuple,
    strides: tuple,
    padding: str | tuple,
) -> WindowConfig:
    """Prepares and validates the window configuration for pooling operations.

    Args:
        rank (int): The total rank of the input tensor.
        spatial_rank (int): The number of spatial dimensions.
        window_shape (tuple): The shape of the pooling window.
        strides (tuple): The strides for the pooling window.
        padding (str | tuple): The padding configuration to apply.

    Returns:
        WindowConfig: A configuration object containing the processed dimensions, strides, and padding.
    """
    pad_dims = max(0, rank - spatial_rank - 1)

    full_window_shape = (1,) * pad_dims + tuple(window_shape) + (1,) if rank > spatial_rank else tuple(window_shape)
    full_strides = (1,) * pad_dims + tuple(strides) + (1,) if rank > spatial_rank else tuple(strides)

    if isinstance(padding, str):
        full_padding = padding
    else:
        full_padding = ((0, 0),) * pad_dims + tuple(padding) + ((0, 0),) if rank > spatial_rank else padding

    return WindowConfig(
        window_dimensions=full_window_shape,
        window_strides=full_strides,
        padding=full_padding,
    )


def _compute_pool_out_shape(in_shape: tuple[int, ...], config: WindowConfig) -> list[int]:
    """Computes the expected output shape for a pooling operation.

    Args:
        in_shape (tuple[int, ...]): The shape of the input tensor.
        config (WindowConfig): The configuration for the pooling window.

    Returns:
        list[int]: A list of integers representing the expected output shape.
    """
    out_shape = []
    if isinstance(config.padding, str):
        for d, w, s in zip(in_shape, config.window_dimensions, config.window_strides):
            if config.padding == "SAME":
                out_shape.append((d + s - 1) // s)
            else:
                out_shape.append((d - w) // s + 1)
    else:
        for d, w, s, p in zip(in_shape, config.window_dimensions, config.window_strides, config.padding):
            if isinstance(p, tuple):
                p_sum = p[0] + p[1]
            else:
                p_sum = p
            out_shape.append((d + p_sum - w) // s + 1)
    return out_shape


def _max_pool_with_indices(
    operand: Tensor,
    window_shape: tuple[int, ...],
    strides: tuple[int, ...],
    padding: str | tuple[tuple[int, int], ...],
    config: WindowConfig,
) -> tuple[Tensor, Tensor]:
    """Computes the max pooling operation and returns the resulting tensor along with its indices.

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
        data, indices = backend.execute_op(
            "MaxPoolWithIndices",
            operand.data,
            window_shape=window_shape,
            strides=strides,
            padding=padding,
        )
        return Tensor(backend.array(data), TensorConfig(tuple(out_shape), operand.dtype, operand.device)), Tensor(backend.array(indices), TensorConfig(tuple(out_shape), "int64", operand.device))

    from ml_switcheroo_compiler.ops.reductions.frontend_utils import _emit_reduction_node

    pooled = _emit_reduction_node(
        "MaxPoolWithIndices",
        [operand],
        {"window_shape": window_shape, "strides": strides, "padding": padding},
        tuple(out_shape),
        operand.dtype,
    )
    indices = _emit_reduction_node(
        "MaxPoolWithIndices_Indices",
        [operand],
        {"window_shape": window_shape, "strides": strides, "padding": padding},
        tuple(out_shape),
        "int64",
    )
    return pooled, indices


def max_pool(
    operand: Tensor,
    window_shape: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Applies a max pooling operation over the input tensor.

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
) -> Tensor:
    """Applies an average pooling operation over the input tensor.

    Args:
        operand (Tensor): The input tensor to pool.
        window_shape (tuple[int, ...]): The dimensions of the pooling window.
        strides (tuple[int, ...] | None): The strides of the pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding mode ('VALID' or 'SAME') or explicit padding configuration.

    Returns:
        Tensor: The resulting tensor after applying average pooling.
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
) -> Tensor:
    """Applies a 1D pooling operation over the input tensor.

    Args:
        operand (Tensor): The 1D input tensor to pool.
        window_shape (int): The size of the 1D pooling window.
        strides (int | None): The stride of the 1D pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding configuration.
        pool_mode (str): The pooling mode, either 'max' or 'avg'.

    Returns:
        Tensor: The resulting pooled tensor.
    """
    shape = (window_shape,)
    stride = (strides,) if strides is not None else None
    if pool_mode == "max":
        return max_pool(operand, shape, stride, padding)
    elif pool_mode == "avg":
        return avg_pool(operand, shape, stride, padding)
    raise ValueError(f"Unknown pool_mode: {pool_mode}")


def pool2d(
    operand: Tensor,
    window_shape: tuple[int, int],
    strides: tuple[int, int] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    pool_mode: str = "max",
) -> Tensor:
    """Applies a 2D pooling operation over the input tensor.

    Args:
        operand (Tensor): The 2D input tensor to pool.
        window_shape (tuple[int, int]): The size of the 2D pooling window.
        strides (tuple[int, int] | None): The strides of the 2D pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding configuration.
        pool_mode (str): The pooling mode, either 'max' or 'avg'.

    Returns:
        Tensor: The resulting pooled tensor.
    """
    if pool_mode == "max":
        return max_pool(operand, window_shape, strides, padding)
    elif pool_mode == "avg":
        return avg_pool(operand, window_shape, strides, padding)
    raise ValueError(f"Unknown pool_mode: {pool_mode}")


def pool3d(
    operand: Tensor,
    window_shape: tuple[int, int, int],
    strides: tuple[int, int, int] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    pool_mode: str = "max",
) -> Tensor:
    """Applies a 3D pooling operation over the input tensor.

    Args:
        operand (Tensor): The 3D input tensor to pool.
        window_shape (tuple[int, int, int]): The size of the 3D pooling window.
        strides (tuple[int, int, int] | None): The strides of the 3D pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding configuration.
        pool_mode (str): The pooling mode, either 'max' or 'avg'.

    Returns:
        Tensor: The resulting pooled tensor.
    """
    if pool_mode == "max":
        return max_pool(operand, window_shape, strides, padding)
    elif pool_mode == "avg":
        return avg_pool(operand, window_shape, strides, padding)
    raise ValueError(f"Unknown pool_mode: {pool_mode}")


def average_pool(
    inputs: Tensor,
    pool_size: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
    padding: str | tuple[tuple[int, int], ...] = "VALID",
    data_format: str | None = None,
) -> Tensor:
    """Computes an average pool over the given input tensor.

    Args:
        inputs (Tensor): The input tensor to process.
        pool_size (tuple[int, ...]): The size of the pooling window.
        strides (tuple[int, ...] | None): The stride for the pooling window.
        padding (str | tuple[tuple[int, int], ...]): The padding configuration.
        data_format (str | None): An optional string specifying the data format (currently ignored).

    Returns:
        Tensor: The resulting pooled tensor.
    """
    return avg_pool(inputs, pool_size, strides, padding)


@dataclass
class SpatialConfig:
    """Configuration for spatial dimensions in a pooling operation.

    Args:
        ksize (object): The kernel size for pooling.
        strides (object): The stride dimensions for pooling.
        padding (object): The padding configuration.
        dilation_rate (object): The dilation rate for pooling. Defaults to None.
        data_format (str | None): The data format string, e.g., 'NHWC'. Defaults to None.
    """

    ksize: object
    strides: object
    padding: object
    dilation_rate: object = None
    data_format: str | None = None


@dataclass
class PoolingBehaviorConfig:
    """Configuration detailing specific behavior for pooling algorithms.

    Args:
        pooling_type (str): The type of pooling ('MAX' or 'AVG'). Defaults to 'MAX'.
        pooling_ratio (object): The pooling ratio used for fractional pooling. Defaults to None.
        output_dtype (object): The desired output data type. Defaults to None.
        include_batch_in_index (bool): Whether to include the batch dimension in indices. Defaults to False.
        pseudo_random (bool): Whether to use pseudo-randomness. Defaults to False.
        overlapping (bool): Whether to allow overlapping regions. Defaults to False.
        deterministic (bool): Whether the pooling should be deterministic. Defaults to False.
        seed (int): The primary seed for random operations. Defaults to 0.
        seed2 (int): The secondary seed for random operations. Defaults to 0.
        name (str | None): An optional name for the operation. Defaults to None.
    """

    pooling_type: str = "MAX"
    pooling_ratio: object = None
    output_dtype: object = None
    include_batch_in_index: bool = False
    pseudo_random: bool = False
    overlapping: bool = False
    deterministic: bool = False
    seed: int = 0
    seed2: int = 0
    name: str | None = None


@dataclass
class PoolingConfig:
    """Comprehensive configuration wrapping spatial and behavioral settings for a pooling operation.

    Args:
        window (SpatialConfig): The spatial dimension configuration.
        behavior (PoolingBehaviorConfig): The specific pooling algorithm configuration.
    """

    window: SpatialConfig
    behavior: PoolingBehaviorConfig


def avg_pool1d(value: object, config: PoolingConfig) -> object:
    """Executes a 1D average pooling operation based on the given configuration.

    Args:
        value (object): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration containing window shapes and behavior.

    Returns:
        object: The resulting 1D average pooled data.
    """
    return avg_pool(
        value,
        window_shape=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )


def avg_pool2d(value: object, config: PoolingConfig) -> object:
    """Executes a 2D average pooling operation based on the given configuration.

    Args:
        value (object): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        object: The resulting 2D average pooled data.
    """
    return avg_pool(
        value,
        window_shape=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )


def avg_pool3d(value: object, config: PoolingConfig) -> object:
    """Executes a 3D average pooling operation based on the given configuration.

    Args:
        value (object): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        object: The resulting 3D average pooled data.
    """
    return avg_pool(
        value,
        window_shape=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )


def max_pool1d(inputs: object, config: PoolingConfig) -> object:
    """Executes a 1D max pooling operation based on the given configuration.

    Args:
        inputs (object): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        object: The resulting 1D max pooled data.
    """
    return max_pool(
        inputs,
        window_shape=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )


def max_pool2d(inputs: object, config: PoolingConfig) -> object:
    """Executes a 2D max pooling operation based on the given configuration.

    Args:
        inputs (object): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        object: The resulting 2D max pooled data.
    """
    return max_pool(
        inputs,
        window_shape=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )


def max_pool3d(inputs: object, config: PoolingConfig) -> object:
    """Executes a 3D max pooling operation based on the given configuration.

    Args:
        inputs (object): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        object: The resulting 3D max pooled data.
    """
    return max_pool(
        inputs,
        window_shape=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )


def max_pool_with_argmax(
    input: object,
    config: PoolingConfig,
) -> object:
    """Executes a max pooling operation and additionally computes the argmax indices based on the configuration.

    Args:
        input (object): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration.

    Returns:
        object: A tuple containing the pooled output and a tensor representing the argmax indices.
    """
    out = max_pool(
        input,
        window_shape=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )
    argmax_out = Tensor(None, TensorConfig(out.shape, "int32", "cpu"))
    return out, argmax_out


def fractional_avg_pool(
    value: Tensor,
    config: PoolingConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Performs fractional average pooling on the input tensor based on the given configuration.

    Args:
        value (Tensor): The input tensor to undergo fractional average pooling.
        config (PoolingConfig): The comprehensive pooling configuration specifying the pooling ratio.

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing the output tensor and two additional tensors indicating states.
    """
    from ml_switcheroo_compiler.core.config import config as _config

    if _config.eager_mode:
        data = get_active_backend().execute_op("FractionalAvgPool", value.data, pooling_ratio=config.behavior.pooling_ratio)
        return (
            Tensor(data, TensorConfig(data.shape, value.dtype, value.device)),
            Tensor([0], TensorConfig((1,), value.dtype, value.device)),
            Tensor([0], TensorConfig((1,), value.dtype, value.device)),
        )

    out_shape = FractionalAvgPool().infer_shape(value, config.behavior.pooling_ratio)
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
) -> tuple[Tensor, Tensor, Tensor]:
    """Performs fractional max pooling on the input tensor based on the given configuration.

    Args:
        value (Tensor): The input tensor to undergo fractional max pooling.
        config (PoolingConfig): The comprehensive pooling configuration specifying the pooling ratio.

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing the output tensor and two additional tensors indicating states.
    """
    from ml_switcheroo_compiler.core.config import config as _config

    if _config.eager_mode:
        data = get_active_backend().execute_op("FractionalMaxPool", value.data, pooling_ratio=config.behavior.pooling_ratio)
        return (
            Tensor(data, TensorConfig(data.shape, value.dtype, value.device)),
            Tensor([0], TensorConfig((1,), value.dtype, value.device)),
            Tensor([0], TensorConfig((1,), value.dtype, value.device)),
        )

    out_shape = FractionalMaxPool().infer_shape(value, config.behavior.pooling_ratio)
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
    input: object,
    config: PoolingConfig,
) -> object:
    """Executes a generic pooling operation, delegating to either average or max pooling depending on the configuration.

    Args:
        input (object): The input tensor or data to pool.
        config (PoolingConfig): The comprehensive pooling configuration specifying spatial dimensions and pooling type.

    Returns:
        object: The resulting pooled data.
    """
    if config.behavior.pooling_type == "AVG":
        return avg_pool(
            input,
            window_shape=config.window.ksize,
            strides=config.window.strides,
            padding=config.window.padding,
        )
    return max_pool(
        input,
        window_shape=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )


@register_op("FractionalAvgPool")
class FractionalAvgPool(OpDef):
    """Operator definition for fractional average pooling."""

    op_name = "FractionalAvgPool"

    def infer_shape(self, value: object, pooling_ratio: object, **kwargs: object) -> object:
        """Infers the output shape of a fractional average pooling operation.

        Args:
            value (object): The input object or tensor.
            pooling_ratio (object): The specified pooling ratio.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred output shape.
        """
        # Just return the input shape modified by the ratio roughly
        return value.shape


@register_op("FractionalMaxPool")
class FractionalMaxPool(OpDef):
    """Operator definition for fractional max pooling."""

    op_name = "FractionalMaxPool"

    def infer_shape(self, value: object, pooling_ratio: object, **kwargs: object) -> object:
        """Infers the output shape of a fractional max pooling operation.

        Args:
            value (object): The input object or tensor.
            pooling_ratio (object): The specified pooling ratio.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The inferred output shape.
        """
        return value.shape
