"""Upsampling operations."""

from collections.abc import Sequence
from typing import Optional, Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import image


def _resolve_scale_factor(
    input: Tensor,
    scale_factor: Union[float, Sequence[float]],
    spatial_dims: int,
) -> tuple[int, ...]:
    """Resolves the scale factor for upsampling.

    Args:
        input (Tensor): The input tensor.
        scale_factor (Union[float, Sequence[float]]): The scale factor.
        spatial_dims (int): The number of spatial dimensions.

    Returns:
        tuple[int, ...]: The resolved target size.
    """
    if isinstance(scale_factor, (float, int)):
        sfs = [float(scale_factor)] * spatial_dims
    else:
        sfs = [float(sf) for sf in scale_factor]

    if len(sfs) != spatial_dims:
        raise ValueError(f"scale_factor length ({len(sfs)}) must match spatial dimensions ({spatial_dims}).")

    try:
        # Assuming shape is (N, C, D1, D2, ...) meaning spatial dimensions start at index 2
        return tuple(int(input.shape[2 + i] * sfs[i]) for i in range(spatial_dims))
    except (TypeError, IndexError):
        # Fallback for dynamic shapes
        return None


def _upsample_resolve_size(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]],
    scale_factor: Optional[Union[float, Sequence[float]]],
) -> tuple[int, ...]:
    """Evaluate and process the upsample resolve size operation.

    Args:
        input (Tensor): Required parameter for input.
        size (Optional): Required parameter for size.
        scale_factor (Optional): Required parameter for scale_factor.

    Returns:
        tuple: The evaluated or processed output.
    """
    has_size = size is not None
    has_sf = scale_factor is not None

    if not has_size:
        if not has_sf:
            raise ValueError("Either size or scale_factor must be defined.")
    elif has_sf:
        raise ValueError("Only one of size or scale_factor should be defined.")

    spatial_dims = len(input.shape) - 2 if len(input.shape) >= 3 else 1

    if has_sf:
        return _resolve_scale_factor(input, scale_factor, spatial_dims)

    if isinstance(size, int):
        return tuple([size] * spatial_dims)

    return tuple(size)  # type: ignore


def _upsample_dispatch(  # noqa: PLR0911
    input: Tensor,
    mode: str,
    size: Optional[Union[int, Sequence[int]]],
    scale_factor: Optional[Union[float, Sequence[float]]],
    align_corners: bool,
) -> Tensor:
    """Evaluate and process the upsample dispatch operation.

    Args:
        input (Tensor): Required parameter for input.
        mode (str): Required parameter for mode.
        size (Optional): Required parameter for size.
        scale_factor (Optional): Required parameter for scale_factor.
        align_corners (bool): Required parameter for align_corners.

    Returns:
        Tensor: The evaluated or processed output.
    """
    target_size = _upsample_resolve_size(input, size, scale_factor)
    spatial_dims = len(input.shape) - 2 if len(input.shape) >= 3 else 1
    sf = None if target_size else scale_factor

    # Hand off to image ops for 2D spatial size (N, C, H, W)
    if spatial_dims == 2:
        dispatch_map = {
            "linear": image.resize_bilinear,
            "bilinear": image.resize_bilinear,
            "trilinear": image.resize_bilinear,
            "bicubic": image.resize_bicubic,
        }
        fn = dispatch_map.get(mode, image.resize_nearest)
        return fn(input, size=target_size, align_corners=align_corners)

    # 1D or 3D fallback
    dispatch_map_fallback = {
        "linear": upsample_bilinear,
        "bilinear": upsample_bilinear,
        "trilinear": upsample_bilinear,
        "bicubic": upsample_bicubic,
    }
    fn_fb = dispatch_map_fallback.get(mode, upsample_nearest)
    return fn_fb(input, size=target_size, scale_factor=sf)


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
    return _upsample_dispatch(input, mode, size, scale_factor, align_corners)


def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    """Rearranges elements in a tensor of shape (*, C, H, W) to a tensor of shape (*, C/r^2, H*r, W*r).

    Args:
        input: The input tensor.
        upscale_factor: Factor to increase spatial resolution by.

    Returns:
        The shuffled tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("PixelShuffle", input.data, upscale_factor=upscale_factor)
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, input.dtype, input.device),
        )
    return _emit_shape_node(
        "PixelShuffle",
        [input],
        {"upscale_factor": upscale_factor},
        (),
        input.dtype,
    )


def upsample_nearest(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
) -> Tensor:
    """Upsamples the input using nearest-neighbor interpolation.

    Args:
        input: The input tensor.
        size: The output spatial size.
        scale_factor: The multiplier for the spatial size.

    Returns:
        The upsampled tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    target_size = _upsample_resolve_size(input, size, scale_factor)
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("UpsampleNearest", input.data, size=target_size, scale_factor=scale_factor)
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, input.dtype, input.device),
        )
    return _emit_shape_node(
        "UpsampleNearest",
        [input],
        {"size": target_size, "scale_factor": scale_factor},
        (),
        input.dtype,
    )


def upsample_bilinear(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
) -> Tensor:
    """Upsamples the input using bilinear interpolation.

    Args:
        input: The input tensor.
        size: The output spatial size.
        scale_factor: The multiplier for the spatial size.

    Returns:
        The upsampled tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    target_size = _upsample_resolve_size(input, size, scale_factor)
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("UpsampleBilinear", input.data, size=target_size, scale_factor=scale_factor)
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, input.dtype, input.device),
        )
    return _emit_shape_node(
        "UpsampleBilinear",
        [input],
        {"size": target_size, "scale_factor": scale_factor},
        (),
        input.dtype,
    )


def upsample_bicubic(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
) -> Tensor:
    """Upsamples the input using bicubic interpolation.

    Args:
        input: The input tensor.
        size: The output spatial size.
        scale_factor: The multiplier for the spatial size.

    Returns:
        The upsampled tensor.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    target_size = _upsample_resolve_size(input, size, scale_factor)
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("UpsampleBicubic", input.data, size=target_size, scale_factor=scale_factor)
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, input.dtype, input.device),
        )
    return _emit_shape_node(
        "UpsampleBicubic",
        [input],
        {"size": target_size, "scale_factor": scale_factor},
        (),
        input.dtype,
    )
