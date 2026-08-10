from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Vision operations."""
from typing import Any

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.configs import BlurConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def gaussian_blur(images: Tensor, config_obj: Any | None = None, **kwargs: Any) -> Any:
    """Apply Gaussian blur to the image(s).

    Args:
        images (Tensor): The images parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        kernel_size = kwargs.get("kernel_size", (3, 3))
        sigma = kwargs.get("sigma", (1.0, 1.0))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(sigma, (float, int)):
            sigma = (float(sigma), float(sigma))
        config_obj = BlurConfig(kernel_size=kernel_size, sigma=sigma, data_format=kwargs.get("data_format", None))

    padding = kwargs.get("padding", "same")

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "GaussianBlur",
            images.data,
            config=config_obj,
            padding=padding,
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node(
        "GaussianBlur",
        [images],
        {"config": config_obj, "padding": padding},
        (),
        DType.Int32,
    )


def median_filter(
    images: Tensor,
    kernel_size: int | tuple[int, int],
    padding: str = "same",
    data_format: str | None = None,
) -> Any:
    """Apply a median filter to the image(s).

    Args:
        images (Tensor): The images parameter.
        kernel_size (object): The kernel_size parameter.
        padding (str): The padding parameter.
        data_format (object): The data_format parameter.

    Returns:
        Tensor: Result.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "MedianFilter",
            images.data,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
        )
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node(
        "MedianFilter",
        [images],
        {
            "kernel_size": kernel_size,
            "padding": padding,
            "data_format": data_format,
        },
        (),
        DType.Int32,
    )


def iou(
    boxes1: Tensor,
    boxes2: Tensor,
    bounding_box_format: str = "xyxy",
) -> Any:
    """Compute Intersection-Over-Union between two sets of bounding boxes.

    Args:
        boxes1 (Tensor): The boxes1 parameter.
        boxes2 (Tensor): The boxes2 parameter.
        bounding_box_format (str): The bounding_box_format parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "IoU",
            boxes1.data,
            boxes2.data,
            bounding_box_format=bounding_box_format,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, boxes1.dtype, boxes1.device),
        )
    return _emit_shape_node(
        "IoU",
        [boxes1, boxes2],
        {"bounding_box_format": bounding_box_format},
        (),
        boxes1.dtype,
    )


def non_max_suppression(
    boxes: Tensor,
    scores: Tensor,
    max_output_size: int,
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
) -> Any:
    """Greedily selects a subset of bounding boxes in descending order of score.

    Args:
        boxes (Tensor): The boxes parameter.
        scores (Tensor): The scores parameter.
        max_output_size (int): The max_output_size parameter.
        iou_threshold (float): The iou_threshold parameter.
        score_threshold (float): The score_threshold parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "NonMaxSuppression",
            boxes.data,
            scores.data,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        # Note: output of NMS is typically integer indices
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, boxes.device))
    return _emit_shape_node(
        "NonMaxSuppression",
        [boxes, scores],
        {
            "max_output_size": max_output_size,
            "iou_threshold": iou_threshold,
            "score_threshold": score_threshold,
        },
        (),
        DType.Int32,
    )


def sharpen(images: Tensor, factor: float = 1.0) -> Any:
    """Sharpen images.

    Args:
        images (Tensor): The images parameter.
        factor (float): The factor parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Sharpen", images.data, factor=factor)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    return get_op("Sharpen")()(images)


def random_gaussian_blur(
    images: Tensor,
    kernel_size: int | tuple[int, int],
    sigma: float | tuple[float, float],
    **kwargs: Any,
) -> Any:
    """Randomly apply Gaussian blur.

    Args:
        images (Tensor): The images parameter.
        kernel_size (object): The kernel_size parameter.
        sigma (object): The sigma parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RandomGaussianBlur",
            images.data,
            kernel_size=kernel_size,
            sigma=sigma,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomGaussianBlur",
        [images],
        {
            "kernel_size": kernel_size,
            "sigma": sigma,
            **kwargs,
        },
        (),
        images.dtype,
    )


def random_sharpness(images: Tensor, factor: float | tuple[float, float], **kwargs: Any) -> Any:
    """Randomly adjust sharpness.

    Args:
        images (Tensor): The images parameter.
        factor (object): The factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RandomSharpness",
            images.data,
            factor=factor,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomSharpness",
        [images],
        {
            "factor": factor,
            **kwargs,
        },
        (),
        images.dtype,
    )
