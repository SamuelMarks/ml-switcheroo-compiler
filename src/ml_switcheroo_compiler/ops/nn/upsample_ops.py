"""Upsampling operations."""

from collections.abc import Sequence
from typing import Optional, Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import image


def _upsample_resolve_size(size: Optional[Union[int, Sequence[int]]], scale_factor: Optional[Union[float, Sequence[float]]]) -> tuple[int, int]:
    """Function docstring."""
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be defined.")
    if size is not None and scale_factor is not None:
        raise ValueError("Only one of size or scale_factor should be defined.")

    if scale_factor is not None:
        raise NotImplementedError("upsample with scale_factor is not fully supported without explicit spatial sizes yet.")

    if isinstance(size, int):
        return (size, size)
    if size is not None:
        t_size = tuple(size)
        if len(t_size) != 2:
            raise NotImplementedError("Only 2D upsampling is currently supported.")
        return t_size
    return (0, 0)


def _upsample_dispatch(input: Tensor, mode: str, target_size: tuple[int, int], align_corners: bool) -> Tensor:
    """Function docstring."""
    if mode == "nearest":
        return image.resize_nearest(input, size=target_size, align_corners=align_corners)
    if mode in ("linear", "bilinear", "trilinear"):
        return image.resize_bilinear(input, size=target_size, align_corners=align_corners)
    if mode == "bicubic":
        return image.resize_bicubic(input, size=target_size, align_corners=align_corners)
    return image.resize_nearest(input, size=target_size, align_corners=align_corners)


def upsample(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """Upsamples a given multi-channel data.

    Args:
        input: The input tensor.
        size: The output spatial size.
        scale_factor: The multiplier for the spatial size.
        mode: The upsampling algorithm: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'.
        align_corners: If True, the corner pixels of the input and output tensors are aligned.

    Returns:
        The upsampled tensor.
    """
    align_corners = bool(align_corners)
    target_size = _upsample_resolve_size(size, scale_factor)
    return _upsample_dispatch(input, mode, target_size, align_corners)
