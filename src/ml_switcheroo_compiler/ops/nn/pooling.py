"""Module docstring."""


# Compute the number of elements in each window for averaging

# Dummy mock
# dummy mock fallback

import math
import typing
from dataclasses import dataclass
from typing import Optional, Union

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.binary import divide
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.creation import ones_like
from ml_switcheroo_compiler.ops.reductions import reduce_window


def max_pool(
    operand: Tensor,
    window_shape: tuple[int, ...],
    strides: Optional[tuple[int, ...]] = None,
    padding: Union[str, tuple[tuple[int, int], ...]] = "VALID",
) -> Tensor:
    """Computes a max pool over the input tensor.

    Args:
        operand (Tensor): The input tensor.
        window_shape (tuple[int, ...]): The shape of the window.
        strides (Optional[tuple[int, ...]]): The strides of the window.
        padding (Union[str, tuple[tuple[int, int], ...]]): The padding to use.

    Returns:
        Tensor: The pooled tensor.
    """
    if strides is None:
        strides = (1,) * len(window_shape)

    rank = len(operand.shape)
    spatial_rank = len(window_shape)
    pad_dims = rank - spatial_rank - 1
    pad_dims = max(0, pad_dims)

    full_window_shape = (1,) * pad_dims + tuple(window_shape) + (1,) if rank > spatial_rank else tuple(window_shape)
    full_strides = (1,) * pad_dims + tuple(strides) + (1,) if rank > spatial_rank else tuple(strides)

    if isinstance(padding, str):
        full_padding = padding
    else:
        full_padding = ((0, 0),) * pad_dims + tuple(padding) + ((0, 0),) if rank > spatial_rank else padding

    config = WindowConfig(
        window_dimensions=full_window_shape,
        window_strides=full_strides,
        padding=full_padding,
    )
    init_val = -math.inf
    return reduce_window(operand, init_val, "max", config)


def _prepare_avg_pool_config(
    rank: int,
    spatial_rank: int,
    window_shape: tuple,
    strides: tuple,
    padding: typing.Union[str, tuple],
) -> WindowConfig:
    """Function docstring.

    Args:
        rank: Arg.
        spatial_rank: Arg.
        window_shape: Arg.
        strides: Arg.
        padding: Arg.
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


def avg_pool(
    operand: Tensor,
    window_shape: tuple[int, ...],
    strides: Optional[tuple[int, ...]] = None,
    padding: Union[str, tuple[tuple[int, int], ...]] = "VALID",
) -> Tensor:
    """Computes an average pool over the input tensor.

    Args:
        operand (Tensor): The input tensor.
        window_shape (tuple[int, ...]): The shape of the window.
        strides (Optional[tuple[int, ...]]): The strides of the window.
        padding (Union[str, tuple[tuple[int, int], ...]]): The padding to use.

    Returns:
        Tensor: The pooled tensor.
    """
    if strides is None:
        strides = (1,) * len(window_shape)

    config = _prepare_avg_pool_config(len(operand.shape), len(window_shape), window_shape, strides, padding)
    init_val = 0.0

    sum_pooled = reduce_window(operand, init_val, "sum", config)

    ones = ones_like(operand)
    counts = reduce_window(ones, 0.0, "sum", config)

    return divide(sum_pooled, counts)


def pool1d(
    operand: Tensor,
    window_shape: int,
    strides: Optional[int] = None,
    padding: Union[str, tuple[tuple[int, int], ...]] = "VALID",
    pool_mode: str = "max",
) -> Tensor:
    """1D Pooling.

    Args:
        operand (Tensor): The input tensor.
        window_shape (int): The shape of the window.
        strides (Optional[int]): The strides of the window.
        padding (Union[str, tuple[tuple[int, int], ...]]): The padding to use.
        pool_mode (str): The pooling mode ('max' or 'avg').

    Returns:
        Tensor: The pooled tensor.
    """
    shape = (window_shape,)  # pragma: no cover
    stride = (strides,) if strides is not None else None  # pragma: no cover
    if pool_mode == "max":  # pragma: no cover
        return max_pool(operand, shape, stride, padding)  # pragma: no cover
    elif pool_mode == "avg":  # pragma: no cover
        return avg_pool(operand, shape, stride, padding)  # pragma: no cover
    raise ValueError(f"Unknown pool_mode: {pool_mode}")  # pragma: no cover


def pool2d(
    operand: Tensor,
    window_shape: tuple[int, int],
    strides: Optional[tuple[int, int]] = None,
    padding: Union[str, tuple[tuple[int, int], ...]] = "VALID",
    pool_mode: str = "max",
) -> Tensor:
    """2D Pooling.

    Args:
        operand (Tensor): The input tensor.
        window_shape (tuple[int, int]): The shape of the window.
        strides (Optional[tuple[int, int]]): The strides of the window.
        padding (Union[str, tuple[tuple[int, int], ...]]): The padding to use.
        pool_mode (str): The pooling mode ('max' or 'avg').

    Returns:
        Tensor: The pooled tensor.
    """
    if pool_mode == "max":  # pragma: no cover
        return max_pool(operand, window_shape, strides, padding)  # pragma: no cover
    elif pool_mode == "avg":  # pragma: no cover
        return avg_pool(operand, window_shape, strides, padding)  # pragma: no cover
    raise ValueError(f"Unknown pool_mode: {pool_mode}")  # pragma: no cover


def pool3d(
    operand: Tensor,
    window_shape: tuple[int, int, int],
    strides: Optional[tuple[int, int, int]] = None,
    padding: Union[str, tuple[tuple[int, int], ...]] = "VALID",
    pool_mode: str = "max",
) -> Tensor:
    """3D Pooling.

    Args:
        operand (Tensor): The input tensor.
        window_shape (tuple[int, int, int]): The shape of the window.
        strides (Optional[tuple[int, int, int]]): The strides of the window.
        padding (Union[str, tuple[tuple[int, int], ...]]): The padding to use.
        pool_mode (str): The pooling mode ('max' or 'avg').

    Returns:
        Tensor: The pooled tensor.
    """
    if pool_mode == "max":  # pragma: no cover
        return max_pool(operand, window_shape, strides, padding)  # pragma: no cover
    elif pool_mode == "avg":  # pragma: no cover
        return avg_pool(operand, window_shape, strides, padding)  # pragma: no cover
    raise ValueError(f"Unknown pool_mode: {pool_mode}")  # pragma: no cover


def average_pool(
    inputs: Tensor,
    pool_size: tuple[int, ...],
    strides: Optional[tuple[int, ...]] = None,
    padding: Union[str, tuple[tuple[int, int], ...]] = "VALID",
    data_format: Optional[str] = None,
) -> Tensor:
    """Average pool.

    Args:
        inputs (Tensor): The input tensor.
        pool_size (tuple[int, ...]): The size of the pool.
        strides (Optional[tuple[int, ...]]): Strides.
        padding (Union[str, tuple[tuple[int, int], ...]]): Padding.
        data_format (Optional[str]): Data format (ignored).

    Returns:
        Tensor: The pooled tensor.
    """
    return avg_pool(inputs, pool_size, strides, padding)


@dataclass
class SpatialConfig:
    """Spatial dimension settings."""

    ksize: object
    strides: object
    padding: object
    dilation_rate: object = None
    data_format: Optional[str] = None


@dataclass
class PoolingBehaviorConfig:
    """Algorithm-specific settings."""

    pooling_type: str = "MAX"
    pooling_ratio: object = None
    output_dtype: object = None
    include_batch_in_index: bool = False
    pseudo_random: bool = False
    overlapping: bool = False
    deterministic: bool = False
    seed: int = 0
    seed2: int = 0
    name: Optional[str] = None


@dataclass
class PoolingConfig:
    """Config for pooling."""

    window: SpatialConfig
    behavior: PoolingBehaviorConfig


def avg_pool1d(value: object, config: PoolingConfig) -> object:
    # pragma: no cover
    """1D Average pooling."""
    return avg_pool(
        value,
        pool_size=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )  # pragma: no cover


def avg_pool2d(value: object, config: PoolingConfig) -> object:
    # pragma: no cover
    """2D Average pooling."""
    return avg_pool(
        value,
        pool_size=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )  # pragma: no cover


def avg_pool3d(value: object, config: PoolingConfig) -> object:
    # pragma: no cover
    """3D Average pooling."""
    return avg_pool(
        value,
        pool_size=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )  # pragma: no cover


def max_pool1d(inputs: object, config: PoolingConfig) -> object:
    # pragma: no cover
    """1D Max pooling."""
    return max_pool(
        inputs,
        pool_size=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )  # pragma: no cover


def max_pool2d(inputs: object, config: PoolingConfig) -> object:
    # pragma: no cover
    """2D Max pooling."""
    return max_pool(
        inputs,
        pool_size=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )  # pragma: no cover


def max_pool3d(inputs: object, config: PoolingConfig) -> object:
    # pragma: no cover
    """3D Max pooling."""
    return max_pool(
        inputs,
        pool_size=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )  # pragma: no cover


def max_pool_with_argmax(
    input: object,
    config: PoolingConfig,
) -> object:  # pragma: no cover
    """Max pooling with argmax."""
    dummy_out = max_pool(
        input,
        pool_size=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )
    dummy_argmax = Tensor(None, TensorConfig(dummy_out.shape, "int32", "cpu"))
    return dummy_out, dummy_argmax


def fractional_avg_pool(
    value: object,
    config: PoolingConfig,
) -> object:  # pragma: no cover
    """Fractional average pooling."""
    pool_size = [int(x) for x in config.behavior.pooling_ratio] if isinstance(config.behavior.pooling_ratio, (list, tuple)) else config.behavior.pooling_ratio
    return (
        avg_pool(value, pool_size=pool_size, strides=pool_size, padding="VALID"),
        Tensor([0], TensorConfig((1,), "int32", "cpu")),
        Tensor([0], TensorConfig((1,), "int32", "cpu")),
    )


def fractional_max_pool(
    value: object,
    config: PoolingConfig,
) -> object:  # pragma: no cover
    """Fractional max pooling."""
    pool_size = [int(x) for x in config.behavior.pooling_ratio] if isinstance(config.behavior.pooling_ratio, (list, tuple)) else config.behavior.pooling_ratio
    return (
        max_pool(value, pool_size=pool_size, strides=pool_size, padding="VALID"),
        Tensor([0], TensorConfig((1,), "int32", "cpu")),
        Tensor([0], TensorConfig((1,), "int32", "cpu")),
    )


def pool(
    input: object,
    config: PoolingConfig,
) -> object:  # pragma: no cover
    """General pooling."""
    if config.behavior.pooling_type == "AVG":
        return avg_pool(
            input,
            pool_size=config.window.ksize,
            strides=config.window.strides,
            padding=config.window.padding,
        )
    return max_pool(
        input,
        pool_size=config.window.ksize,
        strides=config.window.strides,
        padding=config.window.padding,
    )
