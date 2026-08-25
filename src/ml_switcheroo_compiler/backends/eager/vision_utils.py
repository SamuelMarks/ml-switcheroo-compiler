"""Module vision_utils.py."""

from __future__ import annotations

from typing import Any, Callable, Optional

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Vision utilities."""
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
    rng: Any


@dataclass
class GeometricGridConfig:
    """GeometricGridConfig."""

    H: int
    W: int
    rng: Any
    factor1: Any
    factor2: Any


@dataclass
class EagerTransformContext:
    """Configuration class for eager transform context."""

    np_mod: Any
    rng: Any
    imgs: Any
    B: int
    H: int
    W: int
    C: int
    name: str


def _prepare_eager_transform(backend_module: Any, images: Any, seed: Any, data_format: Any) -> Any:
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

    new_y: Any
    new_x: Any
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

    np_mod: Any
    image: Any
    y: Any
    x: Any
    valid: Any


def _map_coords_nearest(ctx: MapCoordsContext) -> Any:
    """Evaluate _map_coords_nearest operation.

    Args:
        ctx (MapCoordsContext): The ctx parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _map_coords_bilinear(ctx: MapCoordsContext) -> Any:
    """Evaluate _map_coords_bilinear operation.

    Args:
        ctx (MapCoordsContext): The ctx parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _np_map_coordinates(np_mod: Any, image: Any, coords: Any, order: int = 1, fill_value: float = 0.0) -> Any:
    """Evaluate _np_map_coordinates operation.

    Args:
        np_mod (object): The np_mod parameter.
        image (object): The image parameter.
        coords (object): The coords parameter.
        order (int): The order parameter.
        fill_value (float): The fill_value parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _compute_perspective_matrix(np_mod: Any, src: Any, dst: Any) -> Any:
    """Evaluate _compute_perspective_matrix operation.

    Args:
        np_mod (object): The np_mod parameter.
        src (object): The src parameter.
        dst (object): The dst parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _generate_perspective_coords(np_mod: Any, h_batch: Any, coords: Any) -> Any:
    """Generate source x and y coordinates for a given batch from homography matrix.

    Args:
        np_mod (object): The np_mod parameter.
        h_batch (object): The h_batch parameter.
        coords (object): The coords parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _generate_perspective_grid(np_mod: Any, H: int, W: int) -> Any:
    """Evaluate _generate_perspective_grid operation.

    Args:
        np_mod (object): The np_mod parameter.
        H (int): The H parameter.
        W (int): The W parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


@dataclass
class PerspectiveContext:
    """Configuration class for perspective context."""

    coords: Any
    h: Any
    b: int


@dataclass
class PerspectiveChannelContext:
    """PerspectiveChannelContext."""

    np_mod: Any
    imgs: Any
    out: Any
    ctx: PerspectiveContext
    config: PerspectiveConfig


def _apply_perspective_channel(pctx: PerspectiveChannelContext) -> Any:
    """Evaluate _apply_perspective_channel operation.

    Args:
        pctx (PerspectiveChannelContext): The pctx parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _apply_perspective_batch(np_mod: Any, imgs: Any, h: Any, config: PerspectiveConfig) -> Any:
    """Apply perspective transform to a batched image array.

    Args:
        np_mod (object): The np_mod parameter.
        imgs (object): The imgs parameter.
        h (object): The h parameter.
        config (PerspectiveConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
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
