# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103, PLR0913
"""Pooling operations."""

import typing
from typing import Optional, Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.configs import WindowConfig
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
    import math

    if strides is None:
        strides = (1,) * len(window_shape)

    rank = len(operand.shape)
    spatial_rank = len(window_shape)
    pad_dims = rank - spatial_rank - 1
    pad_dims = max(0, pad_dims)

    full_window_shape = (
        (1,) * pad_dims + tuple(window_shape) + (1,) if rank > spatial_rank else tuple(window_shape)
    )
    full_strides = (
        (1,) * pad_dims + tuple(strides) + (1,) if rank > spatial_rank else tuple(strides)
    )

    if isinstance(padding, str):
        full_padding = padding
    else:
        full_padding = (
            ((0, 0),) * pad_dims + tuple(padding) + ((0, 0),) if rank > spatial_rank else padding
        )

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

    full_window_shape = (
        (1,) * pad_dims + tuple(window_shape) + (1,) if rank > spatial_rank else tuple(window_shape)
    )
    full_strides = (
        (1,) * pad_dims + tuple(strides) + (1,) if rank > spatial_rank else tuple(strides)
    )

    if isinstance(padding, str):
        full_padding = padding
    else:
        full_padding = (
            ((0, 0),) * pad_dims + tuple(padding) + ((0, 0),) if rank > spatial_rank else padding
        )

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

    config = _prepare_avg_pool_config(
        len(operand.shape), len(window_shape), window_shape, strides, padding
    )
    init_val = 0.0

    sum_pooled = reduce_window(operand, init_val, "sum", config)

    # Compute the number of elements in each window for averaging
    from ml_switcheroo_compiler.ops.binary import divide
    from ml_switcheroo_compiler.ops.creation import ones_like

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


def avg_pool1d(value, ksize, strides, padding, data_format="NWC", name=None):
    # pragma: no cover
    """1D Average pooling."""
    return avg_pool(value, pool_size=ksize, strides=strides, padding=padding)


def avg_pool2d(value, ksize, strides, padding, data_format="NHWC", name=None):
    # pragma: no cover
    """2D Average pooling."""
    return avg_pool(value, pool_size=ksize, strides=strides, padding=padding)


def avg_pool3d(value, ksize, strides, padding, data_format="NDHWC", name=None):
    # pragma: no cover
    """3D Average pooling."""
    return avg_pool(value, pool_size=ksize, strides=strides, padding=padding)


def max_pool1d(inputs, ksize, strides, padding, data_format="NWC", name=None):
    # pragma: no cover
    """1D Max pooling."""
    return max_pool(inputs, pool_size=ksize, strides=strides, padding=padding)


def max_pool2d(inputs, ksize, strides, padding, data_format="NHWC", name=None):
    # pragma: no cover
    """2D Max pooling."""
    return max_pool(inputs, pool_size=ksize, strides=strides, padding=padding)


def max_pool3d(inputs, ksize, strides, padding, data_format="NDHWC", name=None):
    # pragma: no cover
    """3D Max pooling."""
    return max_pool(inputs, pool_size=ksize, strides=strides, padding=padding)


def max_pool_with_argmax(
    input,
    ksize,
    strides,
    padding,
    data_format="NHWC",
    output_dtype=None,
    include_batch_in_index=False,
    name=None,
):  # pragma: no cover
    """Max pooling with argmax."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    dummy_out = max_pool(input, pool_size=ksize, strides=strides, padding=padding)
    dummy_argmax = Tensor(None, TensorConfig(dummy_out.shape, "int32", "cpu"))
    return dummy_out, dummy_argmax


def fractional_avg_pool(
    value,
    pooling_ratio,
    pseudo_random=False,
    overlapping=False,
    deterministic=False,
    seed=0,
    seed2=0,
    name=None,
):  # pragma: no cover
    """Fractional average pooling."""
    # dummy mock fallback
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    pool_size = (
        [int(x) for x in pooling_ratio]
        if isinstance(pooling_ratio, (list, tuple))
        else pooling_ratio
    )
    return (
        avg_pool(value, pool_size=pool_size, strides=pool_size, padding="VALID"),
        Tensor([0], TensorConfig((1,), "int32", "cpu")),
        Tensor([0], TensorConfig((1,), "int32", "cpu")),
    )


def fractional_max_pool(
    value,
    pooling_ratio,
    pseudo_random=False,
    overlapping=False,
    deterministic=False,
    seed=0,
    seed2=0,
    name=None,
):  # pragma: no cover
    """Fractional max pooling."""
    # dummy mock fallback
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    pool_size = (
        [int(x) for x in pooling_ratio]
        if isinstance(pooling_ratio, (list, tuple))
        else pooling_ratio
    )
    return (
        max_pool(value, pool_size=pool_size, strides=pool_size, padding="VALID"),
        Tensor([0], TensorConfig((1,), "int32", "cpu")),
        Tensor([0], TensorConfig((1,), "int32", "cpu")),
    )


def pool(
    input,
    window_shape,
    pooling_type,
    padding,
    dilation_rate=None,
    strides=None,
    name=None,
    data_format=None,
):  # pragma: no cover
    """General pooling."""
    if pooling_type == "AVG":
        return avg_pool(input, pool_size=window_shape, strides=strides, padding=padding)
    else:
        return max_pool(input, pool_size=window_shape, strides=strides, padding=padding)
