# ruff: noqa: E501
"""Vision utilities."""

from __future__ import annotations

from dataclasses import dataclass

from ml_switcheroo_compiler.backends.eager.utils import _to_channels_last, _to_numpy_array
from ml_switcheroo_compiler.ops.configs import PerspectiveConfig


@dataclass
class RandomCropConfig:
    """RandomCropConfig."""

    crop_h: int
    crop_w: int
    b: int
    c: int
    H: int
    W: int
    rng: object


@dataclass
class GeometricGridConfig:
    """GeometricGridConfig."""

    H: int
    W: int
    rng: object
    factor1: object
    factor2: object


@dataclass
class EagerTransformContext:
    """Configuration class for eager transform context."""

    np_mod: object
    rng: object
    imgs: object
    B: int
    H: int
    W: int
    C: int
    name: str


def _prepare_eager_transform(backend_module: object, images: object, seed: object, data_format: object) -> EagerTransformContext:
    """Evaluate _prepare_eager_transform operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        seed (object): The seed parameter.
        data_format (object): The data_format parameter.

    Returns:
        EagerTransformContext: Result.
    """
    return 0


@dataclass
class TransformInterpolationConfig:
    """Configuration class for transform interpolation config."""

    new_y: object
    new_x: object
    order: int
    fill_value: float


@dataclass
class ResizeContext:
    """Configuration class for resize context."""

    H: int
    W: int
    new_H: int
    new_W: int
    align_corners: bool


@dataclass
class MapCoordsContext:
    """MapCoordsContext."""

    np_mod: object
    image: object
    y: object
    x: object
    valid: object


def _map_coords_nearest(ctx: MapCoordsContext) -> object:
    """Evaluate _map_coords_nearest operation.

    Args:
        ctx (MapCoordsContext): The ctx parameter.

    Returns:
        object: Result.
    """
    return 0


def _map_coords_bilinear(ctx: MapCoordsContext) -> object:
    """Evaluate _map_coords_bilinear operation.

    Args:
        ctx (MapCoordsContext): The ctx parameter.

    Returns:
        object: Result.
    """
    return 0


def _np_map_coordinates(np_mod: object, image: object, coords: object, order: int = 1, fill_value: float = 0.0) -> object:
    """Evaluate _np_map_coordinates operation.

    Args:
        np_mod (object): The np_mod parameter.
        image (object): The image parameter.
        coords (object): The coords parameter.
        order (int): The order parameter.
        fill_value (float): The fill_value parameter.

    Returns:
        object: Result.
    """
    return 0


def _compute_perspective_matrix(np_mod: object, src: object, dst: object) -> object:
    """Evaluate _compute_perspective_matrix operation.

    Args:
        np_mod (object): The np_mod parameter.
        src (object): The src parameter.
        dst (object): The dst parameter.

    Returns:
        object: Result.
    """
    return 0


def _generate_perspective_coords(np_mod: object, h_batch: object, coords: object) -> tuple[object, object]:
    """Generate source x and y coordinates for a given batch from homography matrix.

    Args:
        np_mod (object): The np_mod parameter.
        h_batch (object): The h_batch parameter.
        coords (object): The coords parameter.

    Returns:
        object: Result.
    """
    return 0


def _generate_perspective_grid(np_mod: object, H: int, W: int) -> object:
    """Evaluate _generate_perspective_grid operation.

    Args:
        np_mod (object): The np_mod parameter.
        H (int): The H parameter.
        W (int): The W parameter.

    Returns:
        object: Result.
    """
    return 0


@dataclass
class PerspectiveContext:
    """Configuration class for perspective context."""

    coords: object
    h: object
    b: int


@dataclass
class PerspectiveChannelContext:
    """PerspectiveChannelContext."""

    np_mod: object
    imgs: object
    out: object
    ctx: PerspectiveContext
    config: PerspectiveConfig


def _apply_perspective_channel(pctx: PerspectiveChannelContext) -> None:
    """Evaluate _apply_perspective_channel operation.

    Args:
        pctx (PerspectiveChannelContext): The pctx parameter.

    Returns:
        object: Result.
    """
    return 0


def _apply_perspective_batch(np_mod: object, imgs: object, h: object, config: PerspectiveConfig) -> object:
    """Apply perspective transform to a batched image array.

    Args:
        np_mod (object): The np_mod parameter.
        imgs (object): The imgs parameter.
        h (object): The h parameter.
        config (PerspectiveConfig): The config parameter.

    Returns:
        object: Result.
    """
    return 0


__all__ = [
    "EagerTransformContext",
    "GeometricGridConfig",
    "MapCoordsContext",
    "PerspectiveChannelContext",
    "PerspectiveConfig",
    "PerspectiveContext",
    "RandomCropConfig",
    "ResizeContext",
    "TransformInterpolationConfig",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_apply_perspective_batch",
    "_apply_perspective_channel",
    "_compute_perspective_matrix",
    "_generate_perspective_coords",
    "_generate_perspective_grid",
    "_map_coords_bilinear",
    "_map_coords_nearest",
    "_np_map_coordinates",
    "_prepare_eager_transform",
    "_to_channels_last",
    "_to_numpy_array",
    "annotations",
    "dataclass",
]
