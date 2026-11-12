"""Vision operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.configs import BlurConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def gaussian_blur(images: Tensor, config_obj: TensorOrScalar = None, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Apply Gaussian blur to the image(s).

    Args:
        images (Tensor): Input `images` argument (images).
        config_obj (TensorOrScalar): Input `config_obj` argument (config obj).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
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
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), DType.Int32, images.device))
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
) -> TensorOrScalar:
    """Apply a median filter to the image(s).

    Args:
        images (Tensor): Input `images` argument (images).
        kernel_size (int | tuple[int, int]): Input `kernel_size` argument (kernel size).
        padding (str): Input `padding` argument (padding).
        data_format (str | None): Input `data_format` argument (data format).

    Returns:
        Tensor: Computed `Tensor` output.
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
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), DType.Int32, images.device))
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
) -> TensorOrScalar:
    """Compute Intersection-Over-Union between two sets of bounding boxes.

    Args:
        boxes1 (Tensor): Input `boxes1` argument (boxes1).
        boxes2 (Tensor): Input `boxes2` argument (boxes2).
        bounding_box_format (str): Input `bounding_box_format` argument (bounding box format).

    Returns:
        Tensor: Computed `Tensor` output.
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
            TensorConfig(extract_shape(backend.array(data)), boxes1.dtype, boxes1.device),
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
) -> TensorOrScalar:
    """Greedily selects a subset of bounding boxes in descending order of score.

    Args:
        boxes (Tensor): Input `boxes` argument (boxes).
        scores (Tensor): Input `scores` argument (scores).
        max_output_size (int): Input `max_output_size` argument (max output size).
        iou_threshold (float): Input `iou_threshold` argument (iou threshold).
        score_threshold (float): Input `score_threshold` argument (score threshold).

    Returns:
        Tensor: Computed `Tensor` output.
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
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), DType.Int32, boxes.device))
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


def sharpen(images: Tensor, factor: float = 1.0) -> TensorOrScalar:
    """Sharpen images.

    Args:
        images (Tensor): Input `images` argument (images).
        factor (float): Input `factor` argument (factor).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Sharpen", images.data, factor=factor)
        return Tensor(
            backend.array(data),
            TensorConfig(extract_shape(backend.array(data)), images.dtype, images.device),
        )

    return get_op("Sharpen")()(images)


def random_gaussian_blur(
    images: Tensor,
    kernel_size: int | tuple[int, int],
    sigma: float | tuple[float, float],
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """Randomly apply Gaussian blur.

    Args:
        images (Tensor): Input `images` argument (images).
        kernel_size (int | tuple[int, int]): Input `kernel_size` argument (kernel size).
        sigma (float | tuple[float, float]): Input `sigma` argument (sigma).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
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
            TensorConfig(extract_shape(backend.array(data)), images.dtype, images.device),
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


def random_sharpness(images: Tensor, factor: float | tuple[float, float], **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Randomly adjust sharpness.

    Args:
        images (Tensor): Input `images` argument (images).
        factor (float | tuple[float, float]): Input `factor` argument (factor).
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
        Tensor: Computed `Tensor` output.
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
            TensorConfig(extract_shape(backend.array(data)), images.dtype, images.device),
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
