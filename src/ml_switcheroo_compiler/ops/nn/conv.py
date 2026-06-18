"""Convolution operations."""

import math
from collections.abc import Sequence
import typing
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
    if isinstance(strides, int):
        strides_tuple = (strides,) * (len(lhs.shape) - 2)
    else:
        strides_tuple = tuple(strides)

    spatial_dims = len(lhs.shape) - 2
    # For default layout, k_sdims is at the end. Assuming OIW / OIHW / OIDHW
    k_sdims = rhs.shape[2:]

    if isinstance(padding, str) and padding in {"SAME", "VALID"}:
        pads = []
        for k, s in zip(k_sdims, strides_tuple):
            if padding == "SAME":
                pad_len = k + s - 2
                if s > k - 1:
                    pad_a = k - 1
                else:
                    pad_a = int(math.ceil(pad_len / 2.0))
            else:  # VALID
                pad_len = k + s - 2 + max(k - s, 0)
                pad_a = k - 1
            pad_b = pad_len - pad_a
            pads.append((pad_a, pad_b))
    else:
        pads = padding

    from ml_switcheroo_compiler.ops.configs import ConvConfig

    config_obj = ConvConfig(
        window_strides=(1,) * spatial_dims,
        padding=pads,
        lhs_dilation=strides_tuple,
    )

    return conv_general_dilated(lhs, rhs, config_obj)
