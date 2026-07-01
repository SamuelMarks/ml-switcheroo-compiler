"""Upsampling operations."""

from typing import Optional, Union
from collections.abc import Sequence

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import image


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
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be defined.")
    if size is not None and scale_factor is not None:
        raise ValueError("Only one of size or scale_factor should be defined.")

    if align_corners is None:
        align_corners = False

    target_size = size
    if scale_factor is not None:
        # Calculate target size based on input shape
        # Assuming input shape is (N, C, H, W) or similar, but the resize API takes (H, W).
        # We need the spatial dims. Usually, resize expects `size` to be a tuple of ints.
        # This will depend on the backend, but we'll try to infer it.
        raise NotImplementedError(
            "upsample with scale_factor is not fully supported without explicit spatial sizes yet."
        )
        # But for now, we just pass what we can or wait for the user to provide size.

    # Ensure target_size is a tuple
    if isinstance(target_size, int):
        # We assume 2D spatial for now if int is passed.
        target_size = (target_size, target_size)
    elif target_size is not None:
        target_size = tuple(target_size)
        if len(target_size) != 2:
            raise NotImplementedError("Only 2D upsampling is currently supported.")
    else:
        # Fallback dummy for scale_factor not implemented
        target_size = (0, 0)

    # We map common upsample modes to our image resize ops.
    if mode == "nearest":
        return image.resize_nearest(input, size=target_size, align_corners=align_corners)
    elif mode in ("linear", "bilinear", "trilinear"):
        return image.resize_bilinear(input, size=target_size, align_corners=align_corners)
    elif mode == "bicubic":
        return image.resize_bicubic(input, size=target_size, align_corners=align_corners)
    else:
        # Fallback or pass through to nearest if mode is unrecognized in our ops
        return image.resize_nearest(input, size=target_size, align_corners=align_corners)
