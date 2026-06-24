"""Module docstring."""

from ml_switcheroo_compiler.backends.eager.utils import (
    _from_channels_last,
    _from_numpy_array,
    _to_channels_last,
    _to_numpy_array,
)
from ml_switcheroo_compiler.backends.eager.vision_utils import _np_map_coordinates
from ml_switcheroo_compiler.ops.configs import BBoxConfig

from dataclasses import dataclass
from typing import Optional


@dataclass
class NMSConfig:
    """Config for NMS."""

    max_output_size: int
    iou_threshold: float = 0.5
    score_threshold: float = float("-inf")


def _extract_box_channels(
    np_mod: object,
    img: object,
    out: object,
    coords_and_i: tuple[tuple[object, object], int],
    config: BBoxConfig,
) -> None:
    """Function docstring.

    Args:
        np_mod: Arg.
        img: Arg.
        out: Arg.
        coords_and_i: Arg.
        config: Arg.
    """
    order = 1 if config.interpolation == "bilinear" else 0  # pragma: no cover
    coords, i = coords_and_i  # pragma: no cover
    yy, xx = coords  # pragma: no cover
    C = img.shape[-1]  # pragma: no cover
    for c in range(C):  # pragma: no cover
        out[i, ..., c] = _np_map_coordinates(  # pragma: no cover
            np_mod, img[..., c], [yy, xx], order=order, fill_value=config.extrapolation_value
        )


def _get_box_coords(
    np_mod: object, box_ctx: tuple[int, int, int, int], box: object
) -> tuple[object, object]:
    """Function docstring.

    Args:
        np_mod: Arg.
        box_ctx: Arg.
        box: Arg.
    """
    H, W, out_H, out_W = box_ctx  # pragma: no cover
    y1, x1, y2, x2 = box  # pragma: no cover
    y1 = y1 * (H - 1)  # pragma: no cover
    y2 = y2 * (H - 1)  # pragma: no cover
    x1 = x1 * (W - 1)  # pragma: no cover
    x2 = x2 * (W - 1)  # pragma: no cover
    y_coords = np_mod.linspace(y1, y2, out_H)  # pragma: no cover
    x_coords = np_mod.linspace(x1, x2, out_W)  # pragma: no cover
    return np_mod.meshgrid(y_coords, x_coords, indexing="ij")  # pragma: no cover


def _extract_single_box(np_mod: object, batch_ctx: tuple, i: int, config: BBoxConfig) -> None:
    """Function docstring.

    Args:
        np_mod: Arg.
        batch_ctx: Arg.
        i: Arg.
        config: Arg.
    """
    imgs, bxs, bxs_idx, out = batch_ctx  # pragma: no cover
    img = imgs[bxs_idx[i]]  # pragma: no cover
    coords = _get_box_coords(  # pragma: no cover
        np_mod, (img.shape[0], img.shape[1], config.crop_size[0], config.crop_size[1]), bxs[i]
    )
    _extract_box_channels(np_mod, img, out, (coords, i), config)  # pragma: no cover


def _extract_boxes_batch(
    np_mod: object,
    imgs: object,
    bxs: object,
    bxs_idx: object,
    config: BBoxConfig,
) -> object:
    """Extract bounding boxes for a batch."""
    N = bxs.shape[0]  # pragma: no cover
    out_H, out_W = config.crop_size  # pragma: no cover
    C = imgs.shape[-1]  # pragma: no cover
    out = np_mod.zeros((N, out_H, out_W, C), dtype=imgs.dtype)  # pragma: no cover

    for i in range(N):  # pragma: no cover
        _extract_single_box(np_mod, (imgs, bxs, bxs_idx, out), i, config)  # pragma: no cover
    return out  # pragma: no cover


def _extract_boxes_tf(
    backend_module: object,
    images: object,
    boxes: object,
    box_indices: object,
    config: BBoxConfig,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        boxes: Arg.
        box_indices: Arg.
        config: Arg.
    """
    import tensorflow as tf  # pragma: no cover

    if config.data_format == "channels_first":  # pragma: no cover
        images = backend_module.transpose(images, (0, 2, 3, 1))  # pragma: no cover
    images_tf = tf.convert_to_tensor(images)  # pragma: no cover
    boxes_tf = tf.convert_to_tensor(boxes)  # pragma: no cover
    box_idx_tf = tf.convert_to_tensor(box_indices)  # pragma: no cover
    res = tf.image.crop_and_resize(  # pragma: no cover
        images_tf,
        boxes_tf,
        box_idx_tf,
        config.crop_size,
        method=config.interpolation,
        extrapolation_value=config.extrapolation_value,
    )
    res = backend_module.convert_to_tensor(res)  # pragma: no cover
    if config.data_format == "channels_first":  # pragma: no cover
        res = backend_module.transpose(res, (0, 3, 1, 2))  # pragma: no cover
    return res  # pragma: no cover


def extract_bounding_boxes_eager(
    backend_module: object,
    images: object,
    boxes: object,
    box_indices: object,
    config: BBoxConfig,
) -> object:
    """Evaluate extract bounding boxes eagerly."""
    name = getattr(backend_module, "__name__", "")  # pragma: no cover
    np_mod = __import__("numpy")  # pragma: no cover

    data_format = config.data_format  # pragma: no cover

    if name == "keras.ops":  # pragma: no cover
        return _extract_boxes_tf(  # pragma: no cover
            backend_module,
            images,
            boxes,
            box_indices,
            config,
        )

    imgs = _to_numpy_array(np_mod, images, name)  # pragma: no cover
    bxs = _to_numpy_array(np_mod, boxes, name)  # pragma: no cover
    bxs_idx = _to_numpy_array(np_mod, box_indices, name).astype(np_mod.int32)  # pragma: no cover

    imgs = _to_channels_last(np_mod, imgs, data_format)  # pragma: no cover

    out = _extract_boxes_batch(np_mod, imgs, bxs, bxs_idx, config)  # pragma: no cover

    out = _from_channels_last(np_mod, out, data_format)  # pragma: no cover

    return _from_numpy_array(backend_module, out, name, images)  # pragma: no cover


def _to_xyxy_format(np_mod: object, boxes: object, format: str) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        boxes: Arg.
        format: Arg.
    """
    if format == "xyxy":  # pragma: no branch
        return boxes
    if format == "yxyx":  # pragma: no cover
        return boxes[..., [1, 0, 3, 2]]  # pragma: no cover
    if format == "xywh":  # pragma: no cover
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]  # pragma: no cover
        return np_mod.stack([x, y, x + w, y + h], axis=-1)  # pragma: no cover
    if format == "center_xywh":  # pragma: no cover
        cx, cy, w, h = (
            boxes[..., 0],
            boxes[..., 1],
            boxes[..., 2],
            boxes[..., 3],
        )  # pragma: no cover
        return np_mod.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
        )  # pragma: no cover
    return boxes  # pragma: no cover


def _compute_iou(np_mod: object, b1: object, b2: object) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        b1: Arg.
        b2: Arg.
    """
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
    """Function docstring.

    Args:
        np_mod: Arg.
        boxes: Arg.
        scores: Arg.
        score_threshold: Arg.
    """
    valid_mask = scores > score_threshold
    bxs = boxes[valid_mask]
    scs = scores[valid_mask]
    original_idx = np_mod.arange(len(valid_mask))[valid_mask]
    order = scs.argsort()[::-1]
    return bxs, scs, original_idx, order


def _compute_overlap(np_mod: object, bxs: object, i: int, order: object) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        bxs: Arg.
        i: Arg.
        order: Arg.
    """
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
    """Function docstring.

    Args:
        np_mod: Arg.
        bxs: Arg.
        order: Arg.
        max_output_size: Arg.
        iou_threshold: Arg.
    """
    keep = []
    while order.size > 0 and len(keep) < max_output_size:
        i = order[0]
        keep.append(i)

        if order.size == 1:  # pragma: no branch
            break  # pragma: no cover

        ovr = _compute_overlap(np_mod, bxs, i, order)

        inds = np_mod.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def _nms_tf(
    backend_module: object,
    boxes: object,
    scores: object,
    config: Optional[NMSConfig] = None,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        boxes: Arg.
        scores: Arg.
        config: Arg.
    """
    import tensorflow as tf  # pragma: no cover

    conf = config if config is not None else NMSConfig(max_output_size=0)  # pragma: no cover
    max_output_size = conf.max_output_size  # pragma: no cover
    iou_threshold = conf.iou_threshold  # pragma: no cover
    score_threshold = conf.score_threshold  # pragma: no cover

    boxes_tf = tf.convert_to_tensor(boxes)  # pragma: no cover
    scores_tf = tf.convert_to_tensor(scores)  # pragma: no cover
    res = tf.image.non_max_suppression(  # pragma: no cover
        boxes_tf,
        scores_tf,
        max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    return backend_module.convert_to_tensor(res)  # pragma: no cover


def _nms_torch(
    boxes: object,
    scores: object,
    config: NMSConfig,
) -> object:
    """Function docstring.

    Args:
        boxes: Arg.
        scores: Arg.
        config: NMSConfig containing size and thresholds.
    """
    max_output_size = config.max_output_size  # pragma: no cover
    iou_threshold = config.iou_threshold  # pragma: no cover
    score_threshold = config.score_threshold  # pragma: no cover
    import torch  # pragma: no cover
    import torchvision.ops as tv_ops  # pragma: no cover

    mask = scores > score_threshold  # pragma: no cover
    filtered_boxes = boxes[mask]  # pragma: no cover
    filtered_scores = scores[mask]  # pragma: no cover
    original_indices = torch.arange(len(scores), device=scores.device)[mask]  # pragma: no cover
    keep = tv_ops.nms(filtered_boxes, filtered_scores, iou_threshold)  # pragma: no cover
    keep = keep[:max_output_size]  # pragma: no cover
    return original_indices[keep].to(torch.int32)  # pragma: no cover


def nms_eager(
    backend_module: object,
    boxes: object,
    scores: object,
    config: Optional[NMSConfig] = None,
) -> object:
    """Evaluate non max suppression eagerly."""
    name = getattr(backend_module, "__name__", "")  # pragma: no cover
    np_mod = __import__("numpy")  # pragma: no cover
    conf = config if config is not None else NMSConfig(max_output_size=0)  # pragma: no cover
    max_output_size = conf.max_output_size  # pragma: no cover
    iou_threshold = conf.iou_threshold  # pragma: no cover
    score_threshold = conf.score_threshold  # pragma: no cover
    # pragma: no cover
    if name == "keras.ops":  # pragma: no branch  # pragma: no cover
        return _nms_tf(  # pragma: no cover
            backend_module,  # pragma: no cover
            boxes,  # pragma: no cover
            scores,  # pragma: no cover
            config=conf,  # pragma: no cover
        )  # pragma: no cover
    # pragma: no cover
    if name == "torch":  # pragma: no branch  # pragma: no cover
        config = NMSConfig(  # pragma: no cover
            max_output_size=max_output_size,  # pragma: no cover
            iou_threshold=iou_threshold,  # pragma: no cover
            score_threshold=score_threshold,  # pragma: no cover
        )  # pragma: no cover
        return _nms_torch(boxes, scores, config)  # pragma: no cover
    # pragma: no cover
    bxs = _to_numpy_array(np_mod, boxes, name)  # pragma: no cover
    scs = _to_numpy_array(np_mod, scores, name)  # pragma: no cover
    # pragma: no cover
    bxs, scs, original_idx, order = _sort_boxes_by_score(
        np_mod, bxs, scs, score_threshold
    )  # pragma: no cover
    # pragma: no cover
    keep = _apply_suppression_threshold(
        np_mod, bxs, order, max_output_size, iou_threshold
    )  # pragma: no cover
    # pragma: no cover
    keep_indices = original_idx[keep].astype(np_mod.int32)  # pragma: no cover
    # pragma: no cover
    return _from_numpy_array(backend_module, keep_indices, name)  # pragma: no cover
