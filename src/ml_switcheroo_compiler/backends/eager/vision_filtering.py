"""Vision utilities."""

import typing
from ml_switcheroo_compiler.ops.configs import BBoxConfig


def _to_numpy_array(np_mod: object, x: object, name: str) -> object:
    """Convert tensor to numpy array."""
    if name == "torch":
        return x.detach().cpu().numpy()
    if name == "mlx.core":
        return np_mod.array(x)
    if hasattr(x, "numpy"):
        return x.numpy()
    return np_mod.asarray(x)


def _from_numpy_array(
    backend_module: object, out: object, name: str, original_image: object = None
) -> object:
    """Convert numpy array back to backend tensor."""
    if name == "torch":
        import torch

        if original_image is not None:
            return torch.tensor(out, dtype=original_image.dtype, device=original_image.device)
        return torch.tensor(out)
    if name == "mlx.core":
        import mlx.core as mx

        if original_image is not None:
            return mx.array(out, dtype=original_image.dtype)
        return mx.array(out)
    if name == "jax.numpy":
        import jax.numpy as jnp

        if original_image is not None:
            return jnp.array(out, dtype=original_image.dtype)
        return jnp.array(out)

    if original_image is not None:
        return backend_module.array(out, dtype=original_image.dtype)
    return backend_module.array(out)


def _np_map_coordinates(
    np_mod: object, image: object, coords: object, order: int = 1, fill_value: float = 0.0
) -> object:
    y, x = coords[0], coords[1]
    out = np_mod.full(y.shape, fill_value, dtype=image.dtype)
    valid = (y >= 0) & (y <= image.shape[0] - 1) & (x >= 0) & (x <= image.shape[1] - 1)
    if order == 0:
        y_idx = np_mod.round(y[valid]).astype(np_mod.int32)
        x_idx = np_mod.round(x[valid]).astype(np_mod.int32)
        y_idx = np_mod.clip(y_idx, 0, image.shape[0] - 1)
        x_idx = np_mod.clip(x_idx, 0, image.shape[1] - 1)
        out[valid] = image[y_idx, x_idx]
    else:
        y0 = np_mod.floor(y[valid]).astype(np_mod.int32)
        x0 = np_mod.floor(x[valid]).astype(np_mod.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        y0 = np_mod.clip(y0, 0, image.shape[0] - 1)
        x0 = np_mod.clip(x0, 0, image.shape[1] - 1)
        y1 = np_mod.clip(y1, 0, image.shape[0] - 1)
        x1 = np_mod.clip(x1, 0, image.shape[1] - 1)
        dy = y[valid] - y0
        dx = x[valid] - x0
        w00 = (1 - dy) * (1 - dx)
        w01 = (1 - dy) * dx
        w10 = dy * (1 - dx)
        w11 = dy * dx
        val = image[y0, x0] * w00 + image[y0, x1] * w01 + image[y1, x0] * w10 + image[y1, x1] * w11
        out[valid] = val
    return out


def _to_channels_last(np_mod: object, imgs: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_first to channels_last if needed."""
    if data_format == "channels_first" and imgs.ndim >= 3:
        if imgs.ndim == 4:
            return np_mod.transpose(imgs, (0, 2, 3, 1))
        elif imgs.ndim == 3:
            return np_mod.transpose(imgs, (1, 2, 0))
    return imgs


def _from_channels_last(np_mod: object, out: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_last to channels_first if needed."""
    if data_format == "channels_first" and out.ndim >= 3:
        if out.ndim == 4:
            return np_mod.transpose(out, (0, 3, 1, 2))
        elif out.ndim == 3:
            return np_mod.transpose(out, (2, 0, 1))
    return out


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
    interpolation = config.interpolation
    order = 1 if interpolation == "bilinear" else 0
    extrapolation_value = config.extrapolation_value

    for i in range(N):
        idx = bxs_idx[i]
        img = imgs[idx]
        y1, x1, y2, x2 = bxs[i]

        H, W = img.shape[0], img.shape[1]

        y1 = y1 * (H - 1)
        y2 = y2 * (H - 1)
        x1 = x1 * (W - 1)
        x2 = x2 * (W - 1)

        y_coords = np_mod.linspace(y1, y2, out_H)
        x_coords = np_mod.linspace(x1, x2, out_W)
        yy, xx = np_mod.meshgrid(y_coords, x_coords, indexing="ij")

        for c in range(C):
            out[i, ..., c] = _np_map_coordinates(
                np_mod, img[..., c], [yy, xx], order=order, fill_value=extrapolation_value
            )
    return out


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

    crop_size = config.crop_size
    interpolation = config.interpolation
    extrapolation_value = config.extrapolation_value
    data_format = config.data_format

    if name == "keras.ops":
        import tensorflow as tf

        if data_format == "channels_first":
            images = backend_module.transpose(images, (0, 2, 3, 1))
        images_tf = tf.convert_to_tensor(images)
        boxes_tf = tf.convert_to_tensor(boxes)
        box_idx_tf = tf.convert_to_tensor(box_indices)
        res = tf.image.crop_and_resize(
            images_tf,
            boxes_tf,
            box_idx_tf,
            crop_size,
            method=interpolation,
            extrapolation_value=extrapolation_value,
        )
        res = backend_module.convert_to_tensor(res)
        if data_format == "channels_first":
            res = backend_module.transpose(res, (0, 3, 1, 2))
        return res

    imgs = _to_numpy_array(np_mod, images, name)
    bxs = _to_numpy_array(np_mod, boxes, name)
    bxs_idx = _to_numpy_array(np_mod, box_indices, name).astype(np_mod.int32)

    imgs = _to_channels_last(np_mod, imgs, data_format)

    out = _extract_boxes_batch(np_mod, imgs, bxs, bxs_idx, config)

    out = _from_channels_last(np_mod, out, data_format)

    return _from_numpy_array(backend_module, out, name, images)


def iou_eager(
    backend_module: object,
    boxes1: object,
    boxes2: object,
    bounding_box_format: str = "xyxy",
) -> object:
    """Evaluate IoU eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    is_torch = name == "torch"
    is_mlx = name == "mlx.core"

    def to_np(x: object) -> object:
        if is_torch:
            return x.detach().cpu().numpy()
        if is_mlx:
            return np_mod.array(x)
        if hasattr(x, "numpy"):
            return x.numpy()
        return np_mod.asarray(x)

    b1 = to_np(boxes1)
    b2 = to_np(boxes2)

    def to_xyxy(boxes: object, format: str) -> object:
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

    b1 = to_xyxy(b1, bounding_box_format)
    b2 = to_xyxy(b2, bounding_box_format)

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

    iou_vals = np_mod.where(union_area > 0, inter_area / union_area, 0.0)

    if is_torch:
        import torch

        return torch.tensor(iou_vals, dtype=boxes1.dtype, device=boxes1.device)
    if is_mlx:
        import mlx.core as mx

        return mx.array(iou_vals, dtype=boxes1.dtype)
    if name == "jax.numpy":
        import jax.numpy as jnp

        return jnp.array(iou_vals, dtype=boxes1.dtype)
    return np_mod.asarray(iou_vals, dtype=boxes1.dtype)


def nms_eager(
    backend_module: object,
    boxes: object,
    scores: object,
    max_output_size: int,
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
) -> object:
    """Evaluate non max suppression eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    is_torch = name == "torch"
    is_mlx = name == "mlx.core"

    if name == "keras.ops":
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

    if is_torch:
        import torchvision.ops as tv_ops
        import torch

        mask = scores > score_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        original_indices = torch.arange(len(scores), device=scores.device)[mask]
        keep = tv_ops.nms(filtered_boxes, filtered_scores, iou_threshold)
        keep = keep[:max_output_size]
        return original_indices[keep].to(torch.int32)

    def to_np(x: object) -> object:
        if is_torch:
            return x.detach().cpu().numpy()
        if is_mlx:
            return np_mod.array(x)
        if hasattr(x, "numpy"):
            return x.numpy()
        return np_mod.asarray(x)

    bxs = to_np(boxes)
    scs = to_np(scores)

    valid_mask = scs > score_threshold
    bxs = bxs[valid_mask]
    scs = scs[valid_mask]
    original_idx = np_mod.arange(len(valid_mask))[valid_mask]

    order = scs.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_output_size:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np_mod.maximum(bxs[i, 0], bxs[order[1:], 0])
        yy1 = np_mod.maximum(bxs[i, 1], bxs[order[1:], 1])
        xx2 = np_mod.minimum(bxs[i, 2], bxs[order[1:], 2])
        yy2 = np_mod.minimum(bxs[i, 3], bxs[order[1:], 3])

        w = np_mod.maximum(0.0, xx2 - xx1)
        h = np_mod.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (bxs[i, 2] - bxs[i, 0]) * (bxs[i, 3] - bxs[i, 1])
        area_others = (bxs[order[1:], 2] - bxs[order[1:], 0]) * (
            bxs[order[1:], 3] - bxs[order[1:], 1]
        )

        ovr = inter / (area_i + area_others - inter)

        inds = np_mod.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    keep_indices = original_idx[keep].astype(np_mod.int32)

    if is_torch:
        import torch

        return torch.tensor(keep_indices, dtype=torch.int32, device=boxes.device)
    if is_mlx:
        import mlx.core as mx

        return mx.array(keep_indices, dtype=mx.int32)
    if name == "jax.numpy":
        import jax.numpy as jnp

        return jnp.array(keep_indices, dtype=jnp.int32)
    return keep_indices
