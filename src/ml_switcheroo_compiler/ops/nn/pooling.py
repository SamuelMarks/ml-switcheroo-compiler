"""Pooling operations."""

from typing import Optional, Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.reductions import reduce_window
from ml_switcheroo_compiler.ops.configs import WindowConfig


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
    init_val = 0.0

    sum_pooled = reduce_window(operand, init_val, "sum", config)

    # Compute the number of elements in each window for averaging
    from ml_switcheroo_compiler.ops.creation import ones_like
    from ml_switcheroo_compiler.ops.binary import divide

    ones = ones_like(operand)
    counts = reduce_window(ones, init_val, "sum", config)

    return divide(sum_pooled, counts)
