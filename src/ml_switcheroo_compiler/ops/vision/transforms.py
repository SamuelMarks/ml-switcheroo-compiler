"""Module transforms.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Vision operations."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, get_op, register_op
from ml_switcheroo_compiler.ops.configs import ElasticConfig, PerspectiveConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def perspective_transform(
    images: Tensor,
    start_points: Tensor,
    end_points: Tensor,
    config_obj: object | None = None,
    **kwargs: object,
) -> object:
    """Apply a perspective transformation to the image(s).

    Args:
        images (Tensor): The images parameter.
        start_points (Tensor): The start_points parameter.
        end_points (Tensor): The end_points parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        config_obj: object = PerspectiveConfig(
            interpolation=kwargs.get("interpolation", "bilinear"),
            fill_value=kwargs.get("fill_value", 0.0),
            data_format=kwargs.get("data_format", None),
        )

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op(
            "PerspectiveTransform",
            images.data,
            start_points.data,
            end_points.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node(
        "PerspectiveTransform",
        [images, start_points, end_points],
        {"config": config_obj},
        (),
        DType.Int32,
    )


def elastic_transform(
    images: Tensor,
    displacement: Tensor,
    config_obj: object | None = None,
    **kwargs: object,
) -> object:
    """Apply an elastic transformation to the image(s).

    Args:
        images (Tensor): The images parameter.
        displacement (Tensor): The displacement parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        config_obj: object = ElasticConfig(
            interpolation=kwargs.get("interpolation", "bilinear"),
            fill_value=kwargs.get("fill_value", 0.0),
            data_format=kwargs.get("data_format", None),
        )

    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op(
            "ElasticTransform",
            images.data,
            displacement.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node(
        "ElasticTransform",
        [images, displacement],
        {"config": config_obj},
        (),
        DType.Int32,
    )


def flip_left_right(images: Tensor) -> object:
    """Flips images horizontally.

    Args:
        images (Tensor): The images parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op("FlipLeftRight", images.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))

    return get_op("FlipLeftRight")()(images, dtype=DType.Int32)


def flip_up_down(images: Tensor) -> object:
    """Flips images vertically.

    Args:
        images (Tensor): The images parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend: object = get_active_backend()
        data: object = backend.execute_op("FlipUpDown", images.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))

    return get_op("FlipUpDown")()(images, dtype=DType.Int32)


@register_op("ElasticTransform")
class ElasticTransform(OpDef):
    """ElasticTransform operation."""

    op_name: object = "ElasticTransform"

    def infer_shape(self, images: object, *args: object, **kwargs: object) -> object:
        """Infer the output shape for the infer_shape operation.

        Args:
        images (object): The images parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())
