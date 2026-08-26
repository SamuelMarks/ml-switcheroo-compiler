"""vision_filtering.py module."""

from dataclasses import dataclass
from typing import Any, Optional

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.ops.configs import BBoxConfig


@global_eager_registry.register("ExtractVolumePatches")
def _extract_volume_patches(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _extract_volume_patches operation.

    Args:
        backend_module (Any): The backend_module parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            Any: Result.
    """
    import numpy as np

    if not args:
        return backend_module.asarray([]) if hasattr(backend_module, "asarray") else np.array([])

    input_tensor = np.asarray(args[0])
    ksizes = kwargs.get("ksizes", args[1] if len(args) > 1 else [1, 1, 1, 1, 1])
    strides = kwargs.get("strides", args[2] if len(args) > 2 else [1, 1, 1, 1, 1])

    if input_tensor.ndim != 5:
        return backend_module.asarray(input_tensor) if hasattr(backend_module, "asarray") else input_tensor

    window_shape = tuple(ksizes)
    from numpy.lib.stride_tricks import sliding_window_view

    try:
        view = sliding_window_view(input_tensor, window_shape)
    except ValueError:
        return backend_module.asarray(input_tensor) if hasattr(backend_module, "asarray") else input_tensor

    step_view = view[
        :: strides[0],
        :: strides[1],
        :: strides[2],
        :: strides[3],
        :: strides[4],
    ]
    out_shape = step_view.shape[:5] + (-1,)
    res = step_view.reshape(out_shape)

    return backend_module.asarray(res) if hasattr(backend_module, "asarray") else res


@dataclass
class NMSConfig:
    """Config for NMS."""

    max_output_size: int
    iou_threshold: float = 0.5
    score_threshold: float = float("-inf")


def _get_box_coords(np_mod: Any, box_ctx: tuple[int, int, int, int], box: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    """Retrieve the box coords property or mapping.

    Args:
        np_mod (Any): NumPy module.
        box_ctx (tuple[int, int, int, int]): (height, width, crop_height, crop_width).
        box (Any): Bounding box.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: ((y1, x1), (y2, x2)).
    """
    h, w = box_ctx[0], box_ctx[1]
    y1 = int(box[0] * (h - 1))
    x1 = int(box[1] * (w - 1))
    y2 = int(box[2] * (h - 1))
    x2 = int(box[3] * (w - 1))
    return ((y1, x1), (y2, x2))


def _extract_box_channels(np_mod: Any, img: Any, out: Any, coords_and_i: tuple[tuple[tuple[int, int], tuple[int, int]], int], config: BBoxConfig) -> None:
    """Evaluate _extract_box_channels operation.

    Args:
        np_mod (Any): The np_mod parameter.
        img (Any): The img parameter.
        out (Any): The out parameter.
        coords_and_i (tuple[tuple[tuple[int, int], tuple[int, int]], int]): The coords_and_i parameter.
        config (BBoxConfig): The config parameter.

    Returns:
            None: Result.
    """
    (y1_i, x1_i), (y2_i, x2_i) = coords_and_i[0]
    i = coords_and_i[1]
    crop_h, crop_w = config.crop_size

    if y2_i <= y1_i or x2_i <= x1_i:
        return

    cropped = img[y1_i : y2_i + 1, x1_i : x2_i + 1]
    if cropped.size == 0:
        return

    y_indices = np_mod.linspace(0, cropped.shape[0] - 1, crop_h).astype(int)
    x_indices = np_mod.linspace(0, cropped.shape[1] - 1, crop_w).astype(int)
    resized = cropped[y_indices][:, x_indices]
    out[i] = resized


def _extract_single_box(np_mod: Any, batch_ctx: Any, i: int, config: BBoxConfig) -> None:
    """Evaluate _extract_single_box operation.

    Args:
        np_mod (Any): The np_mod parameter.
        batch_ctx (Any): The batch_ctx parameter.
        i (int): The i parameter.
        config (BBoxConfig): The config parameter.

    Returns:
            None: Result.
    """
    imgs, bxs, bxs_idx, out = batch_ctx
    img_idx = int(bxs_idx[i])
    if img_idx >= len(imgs):
        return
    img = imgs[img_idx]
    box = bxs[i]
    if img.ndim < 2:
        return
    h, w = img.shape[0], img.shape[1]
    box_ctx = (h, w, config.crop_size[0], config.crop_size[1])
    coords = _get_box_coords(np_mod, box_ctx, box)
    _extract_box_channels(np_mod, img, out, (coords, i), config)


def _extract_boxes_batch(np_mod: Any, imgs: Any, bxs: Any, bxs_idx: Any, config: BBoxConfig) -> Any:
    """Extract bounding boxes for a batch.

    Args:
        np_mod (Any): NumPy module.
        imgs (Any): Images.
        bxs (Any): Boxes.
        bxs_idx (Any): Box indices.
        config (BBoxConfig): Configuration.

    Returns: Any: Extracted boxes.
    """
    num_boxes = len(bxs)
    if imgs.ndim == 0:
        return np_mod.zeros((num_boxes, config.crop_size[0], config.crop_size[1], 1), dtype=imgs.dtype)

    channels = imgs.shape[-1] if imgs.ndim > 2 else 1
    out_shape = (num_boxes, config.crop_size[0], config.crop_size[1], channels)
    out = np_mod.zeros(out_shape, dtype=imgs.dtype)
    batch_ctx = (imgs, bxs, bxs_idx, out)
    for i in range(num_boxes):
        _extract_single_box(np_mod, batch_ctx, i, config)
    return out


def _extract_boxes_tf(backend_module: Any, images: Any, boxes: Any, box_indices: Any, config: BBoxConfig) -> Any:
    """Evaluate _extract_boxes_tf operation.

    Args:
        backend_module (Any): The backend_module parameter.
        images (Any): The images parameter.
        boxes (Any): The boxes parameter.
        box_indices (Any): The box_indices parameter.
        config (BBoxConfig): The config parameter.

    Returns:
            Any: Result.
    """
    import numpy as np

    images_np = np.asarray(images)
    boxes_np = np.asarray(boxes)
    box_indices_np = np.asarray(box_indices)
    res = _extract_boxes_batch(np, images_np, boxes_np, box_indices_np, config)
    return backend_module.asarray(res) if hasattr(backend_module, "asarray") else res


def extract_bounding_boxes_eager(backend_module: Any, images: Any, boxes: Any, box_indices: Any, config: BBoxConfig) -> Any:
    """Evaluate extract_bounding_boxes_eager operation.

    Args:
        backend_module (Any): The backend_module parameter.
        images (Any): The images parameter.
        boxes (Any): The boxes parameter.
        box_indices (Any): The box_indices parameter.
        config (BBoxConfig): The config parameter.

    Returns:
            Any: Result.
    """
    return _extract_boxes_tf(backend_module, images, boxes, box_indices, config)


def _to_xyxy_format(np_mod: Any, boxes: Any, format: str) -> Any:
    """Evaluate _to_xyxy_format operation.

    Args:
        np_mod (Any): The np_mod parameter.
        boxes (Any): The boxes parameter.
        format (str): The format parameter.

    Returns:
            Any: Result.

    Raises:
        ValueError: An exception.
    """
    boxes = np_mod.asarray(boxes)
    if format == "xyxy":
        return boxes
    elif format == "xyWH":
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return np_mod.stack([x, y, x + w, y + h], axis=-1)
    elif format == "cxcyWH":
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return np_mod.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
    else:
        msg = f"Unknown format {format}"
        raise ValueError(msg)


def _compute_iou(np_mod: Any, b1: Any, b2: Any) -> Any:
    """Evaluate _compute_iou operation.

    Args:
        np_mod (Any): The np_mod parameter.
        b1 (Any): The b1 parameter.
        b2 (Any): The b2 parameter.

    Returns:
            Any: Result.
    """
    b1 = np_mod.asarray(b1)
    b2 = np_mod.asarray(b2)

    if b1.size == 0 or b2.size == 0:
        return np_mod.zeros(b1.shape[:-1], dtype=b1.dtype)

    x11, y11, x12, y12 = np_mod.split(b1, 4, axis=-1)
    x21, y21, x22, y22 = np_mod.split(b2, 4, axis=-1)

    xA = np_mod.maximum(x11, x21)
    yA = np_mod.maximum(y11, y21)
    xB = np_mod.minimum(x12, x22)
    yB = np_mod.minimum(y12, y22)

    interArea = np_mod.maximum(0, xB - xA) * np_mod.maximum(0, yB - yA)
    box1Area = (x12 - x11) * (y12 - y11)
    box2Area = (x22 - x21) * (y22 - y21)

    iou = interArea / (box1Area + box2Area - interArea + 1e-9)
    return iou.reshape(iou.shape[:-1])


def iou_eager(backend_module: Any, boxes1: Any, boxes2: Any, bounding_box_format: str = "xyxy") -> Any:
    """Evaluate iou_eager operation.

    Args:
        backend_module (Any): The backend_module parameter.
        boxes1 (Any): The boxes1 parameter.
        boxes2 (Any): The boxes2 parameter.
        bounding_box_format (str): The bounding_box_format parameter.

    Returns:
            Any: Result.
    """
    import numpy as np

    b1 = _to_xyxy_format(np, boxes1, bounding_box_format)
    b2 = _to_xyxy_format(np, boxes2, bounding_box_format)
    res = _compute_iou(np, b1, b2)
    return backend_module.asarray(res) if hasattr(backend_module, "asarray") else res


def _sort_boxes_by_score(np_mod: Any, boxes: Any, scores: Any, score_threshold: float) -> tuple[Any, Any, Any, Any]:
    """Evaluate _sort_boxes_by_score operation.

    Args:
        np_mod (Any): The np_mod parameter.
        boxes (Any): The boxes parameter.
        scores (Any): The scores parameter.
        score_threshold (float): The score_threshold parameter.

    Returns:
            tuple[Any, Any, Any, Any]: Result.
    """
    boxes = np_mod.asarray(boxes)
    scores = np_mod.asarray(scores)

    if scores.size == 0:
        return boxes, scores, np_mod.array([], dtype=int), np_mod.array([], dtype=int)

    mask = scores >= score_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    indices = np_mod.where(mask)[0]

    order = filtered_scores.argsort()[::-1]
    return filtered_boxes, filtered_scores, indices, order


def _compute_overlap(np_mod: Any, bxs: Any, i: int, order: Any) -> Any:
    """Evaluate _compute_overlap operation.

    Args:
        np_mod (Any): The np_mod parameter.
        bxs (Any): The bxs parameter.
        i (int): The i parameter.
        order (Any): The order parameter.

    Returns:
            Any: Result.
    """
    b1 = bxs[order[i]]
    b2 = bxs[order[1:]]
    return _compute_iou(np_mod, b1, b2)


def _apply_suppression_threshold(np_mod: Any, bxs: Any, order: Any, max_output_size: int, iou_threshold: float) -> Any:
    """Evaluate _apply_suppression_threshold operation.

    Args:
        np_mod (Any): The np_mod parameter.
        bxs (Any): The bxs parameter.
        order (Any): The order parameter.
        max_output_size (int): The max_output_size parameter.
        iou_threshold (float): The iou_threshold parameter.

    Returns:
            Any: Result.
    """
    keep = []
    order_list = list(order)
    while len(order_list) > 0 and len(keep) < max_output_size:
        i = order_list.pop(0)
        keep.append(i)
        if len(order_list) == 0:
            break
        # Compute IoU of the kept box with the remaining boxes
        b1 = bxs[i]
        b2 = bxs[order_list]
        iou = _compute_iou(np_mod, b1, b2)

        # Keep those with IoU <= threshold
        to_keep = iou <= iou_threshold
        order_list = [order_list[j] for j in range(len(order_list)) if to_keep[j]]

    return np_mod.array(keep, dtype=np_mod.int64)


def _nms_tf(backend_module: Any, boxes: Any, scores: Any, config: Optional[NMSConfig] = None) -> Any:
    """Evaluate _nms_tf operation.

    Args:
        backend_module (Any): The backend_module parameter.
        boxes (Any): The boxes parameter.
        scores (Any): The scores parameter.
        config (Optional[NMSConfig]): The config parameter.

    Returns:
            Any: Result.
    """
    if config is None:
        config = NMSConfig(max_output_size=boxes.shape[0] if hasattr(boxes, "shape") else 100)
    res = _nms_torch(boxes, scores, config)
    return backend_module.asarray(res) if hasattr(backend_module, "asarray") else res


def _nms_torch(boxes: Any, scores: Any, config: NMSConfig) -> Any:
    """Evaluate _nms_torch operation.

    Args:
        boxes (Any): The boxes parameter.
        scores (Any): The scores parameter.
        config (NMSConfig): The config parameter.

    Returns:
            Any: Result.
    """
    import numpy as np

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    bxs, scs, idxs, order = _sort_boxes_by_score(np, boxes, scores, config.score_threshold)
    if len(bxs) == 0:
        return np.array([], dtype=np.int64)
    kept_local = _apply_suppression_threshold(np, bxs, order, config.max_output_size, config.iou_threshold)
    return idxs[kept_local]


def nms_eager(backend_module: Any, boxes: Any, scores: Any, config: Optional[NMSConfig] = None) -> Any:
    """Evaluate nms_eager operation.

    Args:
        backend_module (Any): The backend_module parameter.
        boxes (Any): The boxes parameter.
        scores (Any): The scores parameter.
        config (Optional[NMSConfig]): The config parameter.

    Returns:
            Any: Result.
    """
    return _nms_tf(backend_module, boxes, scores, config)
