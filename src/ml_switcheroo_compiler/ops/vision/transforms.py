"""Vision operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, extract_shape
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
        images (Tensor): Input `images` argument (images).
        start_points (Tensor): Input `start_points` argument (start points).
        end_points (Tensor): Input `end_points` argument (end points).
        config_obj (TensorOrScalar): Input `config_obj` argument (config obj).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config_obj is None:
        config_obj = PerspectiveConfig(interpolation=kwargs.get("interpolation", "bilinear"), fill_value=kwargs.get("fill_value", 0.0), data_format=kwargs.get("data_format", None))

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "PerspectiveTransform",
            images.data,
            start_points.data,
            end_points.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), DType.Int32, images.device))
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
        images (Tensor): Input `images` argument (images).
        displacement (Tensor): Input `displacement` argument (displacement).
        config_obj (TensorOrScalar): Input `config_obj` argument (config obj).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config_obj is None:
        config_obj = ElasticConfig(interpolation=kwargs.get("interpolation", "bilinear"), fill_value=kwargs.get("fill_value", 0.0), data_format=kwargs.get("data_format", None))

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "ElasticTransform",
            images.data,
            displacement.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), DType.Int32, images.device))
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
        images (Tensor): Input `images` argument (images).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("FlipLeftRight", images.data)
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), DType.Int32, images.device))

    return get_op("FlipLeftRight")()(images, dtype=DType.Int32)


def flip_up_down(images: Tensor) -> TensorOrScalar:
    """Flips images vertically.

    Args:
        images (Tensor): Input `images` argument (images).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("FlipUpDown", images.data)
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), DType.Int32, images.device))

    return get_op("FlipUpDown")()(images, dtype=DType.Int32)


@register_op("ElasticTransform")
class ElasticTransform(OpDef):
    """ElasticTransform operation."""

    op_name = "ElasticTransform"

    def infer_shape(self, images: EagerTensorProtocol, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the output shape for the infer_shape operation.

        Args:
        images (EagerTensorProtocol): Input `images` argument (images).
        *args (EagerTensorProtocol): Positional args.
        **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        _typed_ret_1: TensorOrScalar = extract_shape(images)
        return _typed_ret_1
