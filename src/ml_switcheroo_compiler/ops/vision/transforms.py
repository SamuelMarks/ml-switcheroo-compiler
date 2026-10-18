"""Vision operations."""

from __future__ import annotations

from typing import Optional, cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, get_op, register_op
from ml_switcheroo_compiler.ops.configs import ElasticConfig, PerspectiveConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def perspective_transform(
    images: Tensor,
    start_points: Tensor,
    end_points: Tensor,
    config_obj: TensorOrScalar = None,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """Apply a perspective transformation to the image(s).

    Args:
        images (Tensor): The images parameter.
        start_points (Tensor): The start_points parameter.
        end_points (Tensor): The end_points parameter.
        config_obj (TensorOrScalar): The config_obj parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        config_obj = cast(
            TensorOrScalar,
            PerspectiveConfig(
                interpolation=cast("str", kwargs.get("interpolation", "bilinear")),
                fill_value=cast("float", kwargs.get("fill_value", 0.0)),
                data_format=cast("Optional[str]", kwargs.get("data_format", None)),
            ),
        )

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "PerspectiveTransform",
            images.data,
            start_points.data,
            end_points.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), TensorConfig(getattr(backend.array(data), "shape"), DType.Int32, images.device))
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
    config_obj: TensorOrScalar = None,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """Apply an elastic transformation to the image(s).

    Args:
        images (Tensor): The images parameter.
        displacement (Tensor): The displacement parameter.
        config_obj (TensorOrScalar): The config_obj parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        config_obj = cast(
            TensorOrScalar,
            ElasticConfig(
                interpolation=cast("str", kwargs.get("interpolation", "bilinear")),
                fill_value=cast("float", kwargs.get("fill_value", 0.0)),
                data_format=cast("Optional[str]", kwargs.get("data_format", None)),
            ),
        )

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "ElasticTransform",
            images.data,
            displacement.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), TensorConfig(getattr(backend.array(data), "shape"), DType.Int32, images.device))
    return _emit_shape_node(
        "ElasticTransform",
        [images, displacement],
        {"config": config_obj},
        (),
        DType.Int32,
    )


def flip_left_right(images: Tensor) -> TensorOrScalar:
    """Flips images horizontally.

    Args:
        images (Tensor): The images parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("FlipLeftRight", images.data)
        return Tensor(backend.array(data), TensorConfig(getattr(backend.array(data), "shape"), DType.Int32, images.device))

    return get_op("FlipLeftRight")()(images, dtype=DType.Int32)


def flip_up_down(images: Tensor) -> TensorOrScalar:
    """Flips images vertically.

    Args:
        images (Tensor): The images parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("FlipUpDown", images.data)
        return Tensor(backend.array(data), TensorConfig(getattr(backend.array(data), "shape"), DType.Int32, images.device))

    return get_op("FlipUpDown")()(images, dtype=DType.Int32)


@register_op("ElasticTransform")
class ElasticTransform(OpDef):
    """ElasticTransform operation."""

    op_name = "ElasticTransform"

    def infer_shape(self, images: EagerTensorProtocol, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape for the infer_shape operation.

        Args:
        images (EagerTensorProtocol): The images parameter.
        *args (EagerTensorProtocol): Positional args.
        **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())
