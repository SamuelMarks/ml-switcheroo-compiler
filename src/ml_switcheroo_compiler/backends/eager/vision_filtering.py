"""Module docstring."""

from ml_switcheroo_compiler.backends.eager.utils import (
    _from_channels_last,
    _from_numpy_array,
    _to_channels_last,
    _to_numpy_array,
)
from ml_switcheroo_compiler.backends.eager.vision_geometric import _np_map_coordinates
from ml_switcheroo_compiler.ops.configs import BBoxConfig


def _extract_box_channels(
    np_mod: object,
    img: object,
    out: object,
    coords_and_i: tuple[tuple[object, object], int],
    config: BBoxConfig,
) -> None:
    order = 1 if config.interpolation == "bilinear" else 0
    coords, i = coords_and_i
    yy, xx = coords
    C = img.shape[-1]
    for c in range(C):
        out[i, ..., c] = _np_map_coordinates(
            np_mod, img[..., c], [yy, xx], order=order, fill_value=config.extrapolation_value
        )


def _get_box_coords(
    np_mod: object, box_ctx: tuple[int, int, int, int], box: object
) -> tuple[object, object]:
    H, W, out_H, out_W = box_ctx
    y1, x1, y2, x2 = box
    y1 = y1 * (H - 1)
    y2 = y2 * (H - 1)
    x1 = x1 * (W - 1)
    x2 = x2 * (W - 1)
    y_coords = np_mod.linspace(y1, y2, out_H)
    x_coords = np_mod.linspace(x1, x2, out_W)
    return np_mod.meshgrid(y_coords, x_coords, indexing="ij")


def _extract_single_box(np_mod: object, batch_ctx: tuple, i: int, config: BBoxConfig) -> None:
    imgs, bxs, bxs_idx, out = batch_ctx
    img = imgs[bxs_idx[i]]
    coords = _get_box_coords(
        np_mod, (img.shape[0], img.shape[1], config.crop_size[0], config.crop_size[1]), bxs[i]
    )
    _extract_box_channels(np_mod, img, out, (coords, i), config)


def _extract_boxes_batch(
    np_mod: object,
    imgs: object,
    bxs: object,
    bxs_idx: object,
    config: BBoxConfig,
) -> object:
    """Extract bounding boxes for a batch."""
    N = bxs.shape[0]
    out_H, out_W = config.crop_size
    C = imgs.shape[-1]
    out = np_mod.zeros((N, out_H, out_W, C), dtype=imgs.dtype)

    for i in range(N):
        _extract_single_box(np_mod, (imgs, bxs, bxs_idx, out), i, config)
    return out


def _extract_boxes_tf(
    backend_module: object,
    images: object,
    boxes: object,
    box_indices: object,
    config: BBoxConfig,
) -> object:
    import tensorflow as tf

    if config.data_format == "channels_first":
        images = backend_module.transpose(images, (0, 2, 3, 1))
    images_tf = tf.convert_to_tensor(images)
    boxes_tf = tf.convert_to_tensor(boxes)
    box_idx_tf = tf.convert_to_tensor(box_indices)
    res = tf.image.crop_and_resize(
        images_tf,
        boxes_tf,
        box_idx_tf,
        config.crop_size,
        method=config.interpolation,
        extrapolation_value=config.extrapolation_value,
    )
    res = backend_module.convert_to_tensor(res)
    if config.data_format == "channels_first":
        res = backend_module.transpose(res, (0, 3, 1, 2))
    return res


def extract_bounding_boxes_eager(
    backend_module: object,
    images: object,
    boxes: object,
    box_indices: object,
    config: BBoxConfig,
) -> object:
    """Evaluate extract bounding boxes eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    data_format = config.data_format

    if name == "keras.ops":
        return _extract_boxes_tf(
            backend_module,
            images,
            boxes,
            box_indices,
            config,
        )

    imgs = _to_numpy_array(np_mod, images, name)
    bxs = _to_numpy_array(np_mod, boxes, name)
    bxs_idx = _to_numpy_array(np_mod, box_indices, name).astype(np_mod.int32)

    imgs = _to_channels_last(np_mod, imgs, data_format)

    out = _extract_boxes_batch(np_mod, imgs, bxs, bxs_idx, config)

    out = _from_channels_last(np_mod, out, data_format)

    return _from_numpy_array(backend_module, out, name, images)


def _to_xyxy_format(np_mod: object, boxes: object, format: str) -> object:
    if format == "xyxy":
        return boxes
    elif format == "yxyx":
        return boxes[..., [1, 0, 3, 2]]
    elif format == "xywh":
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return np_mod.stack([x, y, x + w, y + h], axis=-1)
    elif format == "center_xywh":
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return np_mod.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
    return boxes


def _compute_iou(np_mod: object, b1: object, b2: object) -> object:
    b1_expanded = np_mod.expand_dims(b1, axis=1)  # [N, 1, 4]
    b2_expanded = np_mod.expand_dims(b2, axis=0)  # [1, M, 4]

    inter_mins = np_mod.maximum(b1_expanded[..., :2], b2_expanded[..., :2])
    inter_maxs = np_mod.minimum(b1_expanded[..., 2:], b2_expanded[..., 2:])
    inter_wh = np_mod.maximum(inter_maxs - inter_mins, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    b1_area = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    b2_area = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_area = (
        np_mod.expand_dims(b1_area, axis=1) + np_mod.expand_dims(b2_area, axis=0) - inter_area
    )

    return np_mod.where(union_area > 0, inter_area / union_area, 0.0)


def iou_eager(
    backend_module: object,
    boxes1: object,
    boxes2: object,
    bounding_box_format: str = "xyxy",
) -> object:
    """Evaluate IoU eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    b1 = _to_numpy_array(np_mod, boxes1, name)
    b2 = _to_numpy_array(np_mod, boxes2, name)

    b1 = _to_xyxy_format(np_mod, b1, bounding_box_format)
    b2 = _to_xyxy_format(np_mod, b2, bounding_box_format)

    iou_vals = _compute_iou(np_mod, b1, b2)

    return _from_numpy_array(backend_module, iou_vals, name, boxes1)


def _sort_boxes_by_score(
    np_mod: object, boxes: object, scores: object, score_threshold: float
) -> tuple[object, object, object, object]:
    valid_mask = scores > score_threshold
    bxs = boxes[valid_mask]
    scs = scores[valid_mask]
    original_idx = np_mod.arange(len(valid_mask))[valid_mask]
    order = scs.argsort()[::-1]
    return bxs, scs, original_idx, order


def _compute_overlap(np_mod: object, bxs: object, i: int, order: object) -> object:
    xx1 = np_mod.maximum(bxs[i, 0], bxs[order[1:], 0])
    yy1 = np_mod.maximum(bxs[i, 1], bxs[order[1:], 1])
    xx2 = np_mod.minimum(bxs[i, 2], bxs[order[1:], 2])
    yy2 = np_mod.minimum(bxs[i, 3], bxs[order[1:], 3])

    w = np_mod.maximum(0.0, xx2 - xx1)
    h = np_mod.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_i = (bxs[i, 2] - bxs[i, 0]) * (bxs[i, 3] - bxs[i, 1])
    area_others = (bxs[order[1:], 2] - bxs[order[1:], 0]) * (bxs[order[1:], 3] - bxs[order[1:], 1])
    return inter / (area_i + area_others - inter)


def _apply_suppression_threshold(
    np_mod: object, bxs: object, order: object, max_output_size: int, iou_threshold: float
) -> object:
    keep = []
    while order.size > 0 and len(keep) < max_output_size:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ovr = _compute_overlap(np_mod, bxs, i, order)

        inds = np_mod.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def _nms_tf(
    backend_module: object,
    boxes: object,
    scores: object,
    *,
    max_output_size: int,
    iou_threshold: float,
    score_threshold: float,
) -> object:
    import tensorflow as tf

    boxes_tf = tf.convert_to_tensor(boxes)
    scores_tf = tf.convert_to_tensor(scores)
    res = tf.image.non_max_suppression(
        boxes_tf,
        scores_tf,
        max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    return backend_module.convert_to_tensor(res)


def _nms_torch(  # pylint: disable=too-many-arguments
    boxes: object,
    scores: object,
    max_output_size: int,
    iou_threshold: float,
    score_threshold: float,
) -> object:
    import torch
    import torchvision.ops as tv_ops

    mask = scores > score_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    original_indices = torch.arange(len(scores), device=scores.device)[mask]
    keep = tv_ops.nms(filtered_boxes, filtered_scores, iou_threshold)
    keep = keep[:max_output_size]
    return original_indices[keep].to(torch.int32)


def nms_eager(
    backend_module: object,
    boxes: object,
    scores: object,
    *,
    max_output_size: int,
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
) -> object:
    """Evaluate non max suppression eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    if name == "keras.ops":
        return _nms_tf(
            backend_module,
            boxes,
            scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

    if name == "torch":
        return _nms_torch(boxes, scores, max_output_size, iou_threshold, score_threshold)

    bxs = _to_numpy_array(np_mod, boxes, name)
    scs = _to_numpy_array(np_mod, scores, name)

    bxs, scs, original_idx, order = _sort_boxes_by_score(np_mod, bxs, scs, score_threshold)

    keep = _apply_suppression_threshold(np_mod, bxs, order, max_output_size, iou_threshold)

    keep_indices = original_idx[keep].astype(np_mod.int32)

    return _from_numpy_array(backend_module, keep_indices, name)
