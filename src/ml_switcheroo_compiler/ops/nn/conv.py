"""Convolution operations."""

import math
import typing
from collections.abc import Sequence
from typing import Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg import conv_general_dilated


def conv1d(
    lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object
) -> Tensor:
    """1D Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, length, in_channels).
        rhs (Tensor): Right-hand side tensor (kernel_size, in_channels, out_channels).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):
            strides = (strides,)
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation,)
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation,)
        config_obj = ConvConfig(
            window_strides=strides,
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=((0, 2, 1), (2, 1, 0), (0, 2, 1)),
        )

    return conv_general_dilated(lhs, rhs, config_obj)


def conv2d(
    lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object
) -> Tensor:
    """2D Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (height, width, in_channels, out_channels).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):
            strides = (strides, strides)
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation, lhs_dilation)
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation, rhs_dilation)
        config_obj = ConvConfig(
            window_strides=strides,
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)),
        )

    return conv_general_dilated(lhs, rhs, config_obj)


def conv3d(
    lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object
) -> Tensor:
    """3D Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, depth, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (depth, height, width, in_channels, out_channels).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation, lhs_dilation, lhs_dilation)
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation, rhs_dilation, rhs_dilation)
        config_obj = ConvConfig(
            window_strides=strides,
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=((0, 4, 1, 2, 3), (4, 3, 0, 1, 2), (0, 4, 1, 2, 3)),
        )

    return conv_general_dilated(lhs, rhs, config_obj)


def _calc_same_pad(k: int, s: int) -> tuple[int, int]:
    """Calc SAME pad.

    Args:
        k (int): Kernel.
        s (int): Stride.

    Returns:
        tuple[int, int]: Pads.
    """
    pad_len = k + s - 2
    if s > k - 1:
        pad_a = k - 1
    else:
        pad_a = int(math.ceil(pad_len / 2.0))
    return pad_a, pad_len - pad_a


def _calc_valid_pad(k: int, s: int) -> tuple[int, int]:
    """Calc VALID pad.

    Args:
        k (int): Kernel.
        s (int): Stride.

    Returns:
        tuple[int, int]: Pads.
    """
    pad_len = k + s - 2 + max(k - s, 0)
    pad_a = k - 1
    return pad_a, pad_len - pad_a


def _calculate_conv_transpose_padding(
    padding: Union[str, Sequence[tuple[int, int]]],
    k_sdims: tuple[int, ...],
    strides_tuple: tuple[int, ...],
) -> Sequence[tuple[int, int]]:
    """Calculate padding for transposed convolution.

    Args:
        padding (Union[str, Sequence[tuple[int, int]]]): Padding mode or sequence.
        k_sdims (tuple[int, ...]): Kernel spatial dimensions.
        strides_tuple (tuple[int, ...]): Strides tuple.

    Returns:
        Sequence[tuple[int, int]]: The calculated padding sequence.
    """
    if not (isinstance(padding, str) and padding in {"SAME", "VALID"}):
        return padding

    pads = []
    for k, s in zip(k_sdims, strides_tuple):
        if padding == "SAME":
            pads.append(_calc_same_pad(k, s))
        else:  # VALID
            pads.append(_calc_valid_pad(k, s))
    return pads


def conv1d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> Tensor:
    """1D convolution transpose.

    Args:
        lhs (Tensor): Left-hand side tensor.
        rhs (Tensor): Right-hand side tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.

    Returns:
        Tensor: The result of the convolution.
    """
    return conv_transpose(lhs, rhs, strides, padding)


def conv2d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> Tensor:
    """2D convolution transpose.

    Args:
        lhs (Tensor): Left-hand side tensor.
        rhs (Tensor): Right-hand side tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.

    Returns:
        Tensor: The result of the convolution.
    """
    return conv_transpose(lhs, rhs, strides, padding)


def conv3d_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> Tensor:
    """3D convolution transpose.

    Args:
        lhs (Tensor): Left-hand side tensor.
        rhs (Tensor): Right-hand side tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.

    Returns:
        Tensor: The result of the convolution.
    """
    return conv_transpose(lhs, rhs, strides, padding)


def conv_transpose(
    lhs: Tensor,
    rhs: Tensor,
    strides: Union[Sequence[int], int] = 1,
    padding: Union[str, Sequence[tuple[int, int]]] = "VALID",
) -> Tensor:
    """Convolution transpose.

    Args:
        lhs (Tensor): Left-hand side tensor.
        rhs (Tensor): Right-hand side tensor.
        strides (Union[Sequence[int], int]): Strides.
        padding (Union[str, Sequence[tuple[int, int]]]): Padding.

    Returns:
        Tensor: The result of the convolution.
    """
    spatial_dims = len(lhs.shape) - 2
    if isinstance(strides, int):
        strides_tuple = (strides,) * spatial_dims
    else:
        strides_tuple = tuple(strides)

    # For default layout, k_sdims is at the end. Assuming OIW / OIHW / OIDHW
    k_sdims = rhs.shape[2:]

    pads = _calculate_conv_transpose_padding(padding, k_sdims, strides_tuple)

    from ml_switcheroo_compiler.ops.configs import ConvConfig

    config_obj = ConvConfig(
        window_strides=(1,) * spatial_dims,
        padding=pads,
        lhs_dilation=strides_tuple,
    )

    return conv_general_dilated(lhs, rhs, config_obj)


def depthwise_conv1d(
    lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object
) -> Tensor:
    """1D Depthwise Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, length, in_channels).
        rhs (Tensor): Right-hand side tensor (kernel_size, in_channels, channel_multiplier).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):
            strides = (strides,)
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation,)
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation,)

        in_channels = lhs.shape[-1]

        config_obj = ConvConfig(
            window_strides=strides,
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=((0, 2, 1), (2, 1, 0), (0, 2, 1)),
            feature_group_count=in_channels,
        )

    # Reshape rhs from (kernel_size, in_channels, channel_multiplier)
    # to (kernel_size, 1, in_channels * channel_multiplier)
    # Since jax and our ConvGeneralDilated expects rhs to be (kernel_spatial, in_channels // groups, out_channels)
    from ml_switcheroo_compiler.ops.shape import reshape

    spatial_dims = len(rhs.shape) - 2
    in_channels = rhs.shape[-2]
    channel_multiplier = rhs.shape[-1]

    new_rhs_shape = rhs.shape[:spatial_dims] + (1, in_channels * channel_multiplier)

    # Wait, the data needs to be permuted so that it matches in_channels * channel_multiplier
    # For depthwise, out_channels = in_channels * channel_multiplier.
    # We want groups=in_channels. Each group gets 1 in_channel, and produces channel_multiplier out_channels.
    # The layout for ConvGeneralDilated rhs is usually (spatial..., in_channels_per_group, out_channels)
    # If the input rhs is (spatial..., in_channels, channel_multiplier),
    # we need to reshape it. But wait, out_channels must be grouped.
    # Keras depthwise kernel: (kernel_size, in_channels, depth_multiplier).
    # If we reshape to (kernel_size, 1, in_channels * depth_multiplier), the memory layout matches if we transpose?
    # Keras does depthwise by interleaving?
    # Let's just reshape and see if it passes tests.
    rhs_reshaped = reshape(rhs, new_rhs_shape)

    return conv_general_dilated(lhs, rhs_reshaped, config_obj)


def depthwise_conv2d(
    lhs: Tensor, rhs: Tensor, config_obj: typing.Optional[object] = None, **kwargs: object
) -> Tensor:
    """2D Depthwise Convolution.

    Args:
        lhs (Tensor): Left-hand side tensor (batch, height, width, in_channels).
        rhs (Tensor): Right-hand side tensor (height, width, in_channels, channel_multiplier).
        config_obj (ConvConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: The result of the convolution.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import ConvConfig

        strides = kwargs.get("strides", 1)
        if isinstance(strides, int):
            strides = (strides, strides)
        lhs_dilation = kwargs.get("lhs_dilation", None)
        if isinstance(lhs_dilation, int):
            lhs_dilation = (lhs_dilation, lhs_dilation)
        rhs_dilation = kwargs.get("rhs_dilation", None)
        if isinstance(rhs_dilation, int):
            rhs_dilation = (rhs_dilation, rhs_dilation)

        in_channels = lhs.shape[-1]

        config_obj = ConvConfig(
            window_strides=strides,
            padding=kwargs.get("padding", "VALID"),
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)),
            feature_group_count=in_channels,
        )

    from ml_switcheroo_compiler.ops.shape import reshape

    spatial_dims = len(rhs.shape) - 2
    in_channels = rhs.shape[-2]
    channel_multiplier = rhs.shape[-1]

    # Keras kernel: (H, W, in_channels, depth_multiplier)
    # Output channels for group convolution: ordered by group.
    # Group convolution output channel order is: group0_out0, group0_out1, ..., group1_out0, group1_out1...
    # Keras kernel memory layout is (H, W, in, depth).
    # If we reshape to (H, W, 1, in * depth) directly, does the channel order match?
    # No, reshaping (H, W, in, depth) to (H, W, 1, in * depth) gives order:
    # in0_depth0, in0_depth1, ..., in1_depth0... which IS exactly grouped order!
    # So we can just reshape!

    new_rhs_shape = rhs.shape[:spatial_dims] + (1, in_channels * channel_multiplier)
    rhs_reshaped = reshape(rhs, new_rhs_shape)

    return conv_general_dilated(lhs, rhs_reshaped, config_obj)
