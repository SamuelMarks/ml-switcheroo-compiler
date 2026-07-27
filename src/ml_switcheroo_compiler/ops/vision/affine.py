"""Vision operations."""

from __future__ import annotations

from dataclasses import dataclass

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.config import config as global_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, get_op, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@dataclass
class AffineConfig:
    """Affine Config."""

    fill_mode: str = "reflect"
    interpolation: str = "bilinear"
    seed: int = None
    fill_value: float = 0.0
    data_format: str = None


def affine_transform(images: Tensor, transforms: Tensor, interpolation: str = "nearest") -> Tensor:
    """Applies the given 2D affine transforms to the given images.

    Args:
        images (Tensor): Input images.
        transforms (Tensor): Transform matrices.
        interpolation (str): Interpolation method.

    Returns:
        Tensor: Transformed images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AffineTransform", images.data, transforms.data, interpolation=interpolation)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node("AffineTransform", [images, transforms], {"interpolation": interpolation}, (), DType.Int32)


def affine_generator(batch_size: int, angles: Tensor, shears: Tensor, zooms: Tensor) -> Tensor:
    """Constructs 2D/3D affine matrices from angles, shears, and zoom factors.

    Args:
        batch_size (int): The batch size.
        angles (Tensor): Rotation angles.
        shears (Tensor): Shear values.
        zooms (Tensor): Zoom factors.

    Returns:
        Tensor: Generated affine matrices.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "AffineGenerator",
            batch_size=batch_size,
            angles=angles.data,
            shears=shears.data,
            zooms=zooms.data,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, angles.dtype, angles.device),
        )
    return _emit_shape_node("AffineGenerator", [angles, shears, zooms], {"batch_size": batch_size}, (), angles.dtype)


def random_flip(images: Tensor, mode: str = "horizontal_and_vertical", seed: int = None, **kwargs: object) -> Tensor:
    """Randomly flip images horizontally and/or vertically.

    Args:
        images (Tensor): Input images.
        mode (str): String indicating which flip to apply.
        seed (int): Random seed.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Flipped images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RandomFlip", images.data, mode=mode, seed=seed)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    kwargs = {"mode": mode, "seed": seed}
    return get_op("RandomFlip")()(images, **kwargs)


def random_rotation(
    images: Tensor,
    factor: float,
    config: AffineConfig | None = None,
    **kwargs: object,
) -> Tensor:
    """Randomly rotate images.

    Args:
        images (Tensor): Input images.
        factor (float): A float represented as fraction of 2 Pi.
        config: Config.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Rotated images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RandomRotation",
            images.data,
            factor=factor,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomRotation",
        [images],
        {
            "factor": factor,
            **kwargs,
        },
        (),
        images.dtype,
    )


def random_crop(images: Tensor, size: tuple, seed: int = None, **kwargs: object) -> Tensor:
    """Randomly crop images.

    Args:
        images (Tensor): Input images.
        size (tuple): Target size.
        seed (int): Random seed.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Cropped images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RandomCrop", images.data, size=size, seed=seed)
        new_shape = list(backend.array(data).shape)
        return Tensor(backend.array(data), TensorConfig(tuple(new_shape), images.dtype, images.device))

    kwargs = {"size": size, "seed": seed}
    return get_op("RandomCrop")()(images, **kwargs)


def random_zoom(
    images: Tensor,
    height_factor: tuple[float, float] | float,
    width_factor: tuple[float, float] | float | None = None,
    **kwargs: object,
) -> Tensor:
    """Randomly zoom images.

    Args:
        images (Tensor): Input images.
        height_factor (Union[tuple[float, float], float]): Factor for zooming height.
        width_factor (Union[tuple[float, float], float, None]): Factor for zooming width.
        config: Config.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Zoomed images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RandomZoom",
            images.data,
            height_factor=height_factor,
            width_factor=width_factor,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomZoom",
        [images],
        {
            "height_factor": height_factor,
            "width_factor": width_factor,
            **kwargs,
        },
        (),
        images.dtype,
    )


def random_translation(
    images: Tensor,
    height_factor: tuple[float, float] | float,
    width_factor: tuple[float, float] | float,
    config: AffineConfig | None = None,
    **kwargs: object,
) -> Tensor:
    """Randomly translate images.

    Args:
        images (Tensor): Input images.
        height_factor (Union[tuple[float, float], float]): Factor for translating height.
        width_factor (Union[tuple[float, float], float]): Factor for translating width.
        config: Config.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Translated images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RandomTranslation",
            images.data,
            height_factor=height_factor,
            width_factor=width_factor,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomTranslation",
        [images],
        {
            "height_factor": height_factor,
            "width_factor": width_factor,
            **kwargs,
        },
        (),
        images.dtype,
    )


def random_shear(
    images: Tensor,
    y_factor: float | tuple[float, float],
    x_factor: float | tuple[float, float] | None = None,
    **kwargs: object,
) -> Tensor:
    """Randomly shear images.

    Args:
        images (Tensor): Input images.
        y_factor (Union[float, tuple[float, float]]): Factor for y shear.
        x_factor (Union[float, tuple[float, float], None]): Factor for x shear.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Sheared images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RandomShear",
            images.data,
            y_factor=y_factor,
            x_factor=x_factor,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomShear",
        [images],
        {
            "y_factor": y_factor,
            "x_factor": x_factor,
            **kwargs,
        },
        (),
        images.dtype,
    )


def random_perspective(
    images: Tensor,
    factor: float | tuple[float, float],
    **kwargs: object,
) -> Tensor:
    """Randomly apply perspective transform to images.

    Args:
        images (Tensor): Input images.
        factor (Union[float, tuple[float, float]]): Factor for perspective transform.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Transformed images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RandomPerspective",
            images.data,
            factor=factor,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomPerspective",
        [images],
        {
            "factor": factor,
            **kwargs,
        },
        (),
        images.dtype,
    )


def random_elastic_transform(
    images: Tensor,
    alpha: float | tuple[float, float],
    sigma: float | tuple[float, float],
    **kwargs: object,
) -> Tensor:
    """Randomly apply elastic transform to images.

    Args:
        images (Tensor): Input images.
        alpha (Union[float, tuple[float, float]]): Alpha factor for elastic transform.
        sigma (Union[float, tuple[float, float]]): Sigma factor for elastic transform.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Transformed images.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RandomElasticTransform",
            images.data,
            alpha=alpha,
            sigma=sigma,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomElasticTransform",
        [images],
        {
            "alpha": alpha,
            "sigma": sigma,
            **kwargs,
        },
        (),
        images.dtype,
    )


def affine_grid(theta: Tensor, size: tuple[int, ...], align_corners: bool = False) -> Tensor:
    """Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta.

    Args:
        theta (Tensor): input batch of affine matrices with shape (N, 2, 3) for 2D or (N, 3, 4) for 3D
        size (tuple[int, ...]): the target output image size
        align_corners (bool): if True, consider -1 and 1 to refer to the centers of the corner pixels

    Returns:
    Tensor: output Tensor of size (N, H, W, 2)
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AffineGrid", theta.data, size=size, align_corners=align_corners)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, theta.dtype, theta.device))
    return _emit_shape_node("AffineGrid", [theta], {"size": size, "align_corners": align_corners}, (), theta.dtype)


def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
    """Given an input and a flow-field grid, computes the output using input values and pixel locations from grid.

    Args:
        input (Tensor): input of shape (N, C, H_in, W_in)
        grid (Tensor): flow-field of shape (N, H_out, W_out, 2)
        mode (str): interpolation mode to calculate output values 'bilinear' | 'nearest' | 'bicubic'
        padding_mode (str): padding mode for outside grid values 'zeros' | 'border' | 'reflection'
        align_corners (bool): Geometrically, we consider the pixels of the input as squares rather than points.

    Returns:
    Tensor: output Tensor
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "GridSample",
            input.data,
            grid.data,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    return _emit_shape_node(
        "GridSample",
        [input, grid],
        {"mode": mode, "padding_mode": padding_mode, "align_corners": align_corners},
        (),
        input.dtype,
    )


@register_op("AffineGenerator")
class AffineGenerator(OpDef):
    """AffineGenerator operation."""

    op_name = "AffineGenerator"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(inputs, "shape", ())


@register_op("AffineGrid")
class AffineGrid(OpDef):
    """AffineGrid operation."""

    op_name = "AffineGrid"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(inputs, "shape", ())


@register_op("AffineTransform")
class AffineTransform(OpDef):
    """AffineTransform operation."""

    op_name = "AffineTransform"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(inputs, "shape", ())


@register_op("PerspectiveTransform")
class PerspectiveTransform(OpDef):
    """PerspectiveTransform operation."""

    op_name = "PerspectiveTransform"

    def infer_shape(self, inputs: object, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(inputs, "shape", ())
