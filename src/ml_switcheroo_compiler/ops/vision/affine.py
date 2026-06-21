"""Vision operations."""

from __future__ import annotations

from __future__ import annotations
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def affine_transform(images: Tensor, transforms: Tensor, interpolation: str = "nearest") -> Tensor:
    """Applies the given 2D affine transforms to the given images.

    Args:
        images (Tensor): Input images.
        transforms (Tensor): Transform matrices.
        interpolation (str): Interpolation method.

    Returns:
        Tensor: Transformed images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "AffineTransform", images.data, transforms.data, interpolation=interpolation
        )
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
    return _emit_shape_node(
        "AffineTransform", [images, transforms], {"interpolation": interpolation}, (), DType.Int32
    )


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
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

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
    return _emit_shape_node(
        "AffineGenerator", [angles, shears, zooms], {"batch_size": batch_size}, (), angles.dtype
    )


def random_flip(
    images: Tensor, mode: str = "horizontal_and_vertical", seed: int = None, **kwargs: object
) -> Tensor:
    """Randomly flip images horizontally and/or vertically.

    Args:
        images (Tensor): Input images.
        mode (str): String indicating which flip to apply.
        seed (int): Random seed.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Flipped images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("RandomFlip", images.data, mode=mode, seed=seed)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op

    kwargs = {"mode": mode, "seed": seed}
    return get_op("RandomFlip")()(images, **kwargs)


def random_rotation(  # pylint: disable=too-many-arguments
    images: Tensor,
    factor: float,
    fill_mode: str = "reflect",
    interpolation: str = "bilinear",
    seed: int = None,
    fill_value: float = 0.0,
    data_format: str = None,
    **kwargs: object,
) -> Tensor:
    """Randomly rotate images.

    Args:
        images (Tensor): Input images.
        factor (float): A float represented as fraction of 2 Pi.
        fill_mode (str): Points outside boundaries are filled according to mode.
        interpolation (str): Interpolation method.
        seed (int): Random seed.
        fill_value (float): Fill value.
        data_format (str): Data format.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Rotated images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

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
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("RandomCrop", images.data, size=size, seed=seed)
        new_shape = list(backend.array(data).shape)
        return Tensor(
            backend.array(data), TensorConfig(tuple(new_shape), images.dtype, images.device)
        )
    from ml_switcheroo_compiler.ops.base import get_op

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
        fill_mode (str): Points outside boundaries are filled according to mode.
        interpolation (str): Interpolation method.
        seed (int): Random seed.
        fill_value (float): Fill value.
        data_format (str): Data format.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Zoomed images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

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


def random_translation(  # pylint: disable=too-many-arguments
    images: Tensor,
    height_factor: tuple[float, float] | float,
    width_factor: tuple[float, float] | float,
    fill_mode: str = "reflect",
    interpolation: str = "bilinear",
    seed: int = None,
    fill_value: float = 0.0,
    data_format: str = None,
    **kwargs: object,
) -> Tensor:
    """Randomly translate images.

    Args:
        images (Tensor): Input images.
        height_factor (Union[tuple[float, float], float]): Factor for translating height.
        width_factor (Union[tuple[float, float], float]): Factor for translating width.
        fill_mode (str): Points outside boundaries are filled according to mode.
        interpolation (str): Interpolation method.
        seed (int): Random seed.
        fill_value (float): Fill value.
        data_format (str): Data format.
        **kwargs: Additional kwargs.

    Returns:
        Tensor: Translated images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

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
