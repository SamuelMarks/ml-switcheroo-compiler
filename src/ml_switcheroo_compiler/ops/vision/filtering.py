"""Vision operations."""

from __future__ import annotations


from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def gaussian_blur(images: Tensor, config_obj: object | None = None, **kwargs: object) -> Tensor:
    """Applies Gaussian blur to the image(s).

    Args:
        images (Tensor): Input images.
        config_obj (BlurConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: Blurred images.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import BlurConfig

        kernel_size = kwargs.get("kernel_size", (3, 3))
        sigma = kwargs.get("sigma", (1.0, 1.0))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(sigma, (float, int)):
            sigma = (float(sigma), float(sigma))
        config_obj = BlurConfig(
            kernel_size=kernel_size, sigma=sigma, data_format=kwargs.get("data_format", None)
        )

    padding = kwargs.get("padding", "same")

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "GaussianBlur",
            images.data,
            config=config_obj,
            padding=padding,
        )
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
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
) -> Tensor:
    """Applies a median filter to the image(s).

    Args:
        images (Tensor): Input images.
        kernel_size (int | tuple[int, int]): Size of the median filter kernel.
        padding (str): Padding mode ('same' or 'valid').
        data_format (str | None): Data format ('channels_last' or 'channels_first').

    Returns:
        Tensor: Filtered images.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "MedianFilter",
            images.data,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
        )
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
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
) -> Tensor:
    """Computes Intersection-Over-Union between two sets of bounding boxes.

    Args:
        boxes1 (Tensor): First set of bounding boxes [N, 4].
        boxes2 (Tensor): Second set of bounding boxes [M, 4].
        bounding_box_format (str): The format of the bounding boxes ('xyxy', 'yxyx', 'xywh', 'center_xywh').

    Returns:
        Tensor: IoU matrix of shape [N, M].
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

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
) -> Tensor:
    """Greedily selects a subset of bounding boxes in descending order of score.

    Args:
        boxes (Tensor): A 2-D float tensor of shape [num_boxes, 4].
        scores (Tensor): A 1-D float tensor of shape [num_boxes] representing a single score corresponding to each box.
        max_output_size (int): A scalar integer tensor representing the maximum number of boxes to be selected by non max suppression.
        iou_threshold (float): A float representing the threshold for deciding whether boxes overlap too much with respect to IOU.
        score_threshold (float): A float representing the threshold for deciding when to remove boxes based on score.

    Returns:
        Tensor: A 1-D integer tensor of shape [M] representing the selected indices from the boxes tensor, where M <= max_output_size.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

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
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, boxes.device)
        )
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


def sharpen(images: Tensor, factor: float = 1.0) -> Tensor:
    """Sharpen images.

    Args:
        images (Tensor): Input images.
        factor (float): Sharpening factor.

    Returns:
        Tensor: Sharpened images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Sharpen", images.data, factor=factor)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op

    kwargs = {"factor": factor}
    return get_op("Sharpen")()(images, **kwargs)
