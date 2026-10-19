"""Vision spatial patch extraction, bounding box cropping, IoU, and NMS eager kernels."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.ops.configs import BBoxConfig


def _resolve_filter_mod(backend_module: EagerValue | None) -> ModuleType:
    """Resolve an array module providing `zeros`, `linspace`, `stack`, and `where`.

    Args:
        backend_module: Candidate backend module or None.

    Returns:
        ModuleType: Resolved array module.
    """
    from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod

    return _resolve_eager_mod(backend_module)


@global_eager_registry.register("ExtractVolumePatches")
def _extract_volume_patches(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager extract volume patches operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        EagerValue: Result.
    """
    mod = _resolve_filter_mod(backend_module)
    if not args:
        return cast(EagerValue, getattr(backend_module, "asarray")([]) if hasattr(backend_module, "asarray") else mod.array([]))

    input_tensor: EagerTensorProtocol = mod.asarray(args[0])
    ksizes = cast("list[int]", kwargs.get("ksizes", args[1] if len(args) > 1 else [1, 1, 1, 1, 1]) if hasattr(kwargs, "get") else (args[1] if len(args) > 1 else [1, 1, 1, 1, 1]))
    strides = cast("list[int]", kwargs.get("strides", args[2] if len(args) > 2 else [1, 1, 1, 1, 1]) if hasattr(kwargs, "get") else (args[2] if len(args) > 2 else [1, 1, 1, 1, 1]))

    if input_tensor.ndim != 5:
        return cast(EagerValue, getattr(backend_module, "asarray")(input_tensor) if hasattr(backend_module, "asarray") else input_tensor)

    shape = input_tensor.shape
    if any(int(k) <= 0 or int(k) > int(s) for k, s in zip(ksizes, shape)):
        return cast(EagerValue, getattr(backend_module, "asarray")(input_tensor) if hasattr(backend_module, "asarray") else input_tensor)

    out_dims = [len(range(0, int(shape[d]) - int(ksizes[d]) + 1, int(strides[d]))) for d in range(5)]
    patch_vol = int(ksizes[0]) * int(ksizes[1]) * int(ksizes[2]) * int(ksizes[3]) * int(ksizes[4])
    res: EagerTensorProtocol = mod.zeros((*out_dims, patch_vol), dtype=input_tensor.dtype)

    for i0, s0 in enumerate(range(0, int(shape[0]) - int(ksizes[0]) + 1, int(strides[0]))):
        for i1, s1 in enumerate(range(0, int(shape[1]) - int(ksizes[1]) + 1, int(strides[1]))):
            for i2, s2 in enumerate(range(0, int(shape[2]) - int(ksizes[2]) + 1, int(strides[2]))):
                for i3, s3 in enumerate(range(0, int(shape[3]) - int(ksizes[3]) + 1, int(strides[3]))):
                    for i4, s4 in enumerate(range(0, int(shape[4]) - int(ksizes[4]) + 1, int(strides[4]))):
                        patch = input_tensor[
                            s0 : s0 + int(ksizes[0]),
                            s1 : s1 + int(ksizes[1]),
                            s2 : s2 + int(ksizes[2]),
                            s3 : s3 + int(ksizes[3]),
                            s4 : s4 + int(ksizes[4]),
                        ]
                        res[i0, i1, i2, i3, i4] = patch.reshape(-1)

    return cast(EagerValue, getattr(backend_module, "asarray")(res) if hasattr(backend_module, "asarray") else res)


@dataclass
class NMSConfig:
    """Config for NMS."""

    max_output_size: int
    iou_threshold: float = 0.5
    score_threshold: float = float("-inf")


def _get_box_coords(np_mod: EagerValue, box_ctx: tuple[int, int, int, int], box: EagerValue) -> tuple[tuple[int, int], tuple[int, int]]:
    """Retrieve the box coords property or mapping.

    Args:
        np_mod: NumPy module.
        box_ctx (tuple[int, int, int, int]): (height, width, crop_height, crop_width).
        box: Bounding box.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: ((y1, x1), (y2, x2)).
    """
    h, w = box_ctx[0], box_ctx[1]
    box_arr = cast(EagerTensorProtocol, box)
    y1 = int(box_arr[0] * (h - 1))
    x1 = int(box_arr[1] * (w - 1))
    y2 = int(box_arr[2] * (h - 1))
    x2 = int(box_arr[3] * (w - 1))
    return ((y1, x1), (y2, x2))


def _extract_box_channels(np_mod: EagerValue, img: EagerValue, out: EagerValue, coords_and_i: tuple[tuple[tuple[int, int], tuple[int, int]], int], config: BBoxConfig) -> None:
    """Execute the eager extract box channels operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        img: The img parameter.
        out: The out parameter.
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

    cropped = cast(EagerTensorProtocol, img)[y1_i : y2_i + 1, x1_i : x2_i + 1]
    if cropped.size == 0:
        return

    y_indices = getattr(np_mod, "linspace")(0, cropped.shape[0] - 1, crop_h).astype(int)
    x_indices = getattr(np_mod, "linspace")(0, cropped.shape[1] - 1, crop_w).astype(int)
    resized = cropped[y_indices][:, x_indices]
    cast(EagerTensorProtocol, out)[i] = resized


def _extract_single_box(np_mod: EagerValue, batch_ctx: EagerValue, i: int, config: BBoxConfig) -> None:
    """Execute the eager extract single box operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        batch_ctx: The batch_ctx parameter.
        i (int): The i parameter.
        config (BBoxConfig): The config parameter.

    Returns:
            None: Result.
    """
    imgs, bxs, bxs_idx, out = cast("tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]", batch_ctx)
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


def _extract_boxes_batch(np_mod: EagerValue, imgs: EagerValue, bxs: EagerValue, bxs_idx: EagerValue, config: BBoxConfig) -> EagerValue:
    """Extract bounding boxes for a batch.

    Args:
        np_mod: NumPy module.
        imgs: Images.
        bxs: Boxes.
        bxs_idx: Box indices.
        config (BBoxConfig): Configuration.

    Returns: Any: Extracted boxes.
    """
    imgs_arr = cast(EagerTensorProtocol, imgs)
    num_boxes = getattr(bxs, "__len__")()
    if imgs_arr.ndim == 0:
        return cast(EagerValue, getattr(np_mod, "zeros")((num_boxes, config.crop_size[0], config.crop_size[1], 1), dtype=imgs_arr.dtype))

    channels = imgs_arr.shape[-1] if imgs_arr.ndim > 2 else 1
    out_shape = (num_boxes, config.crop_size[0], config.crop_size[1], channels)
    out = getattr(np_mod, "zeros")(out_shape, dtype=imgs_arr.dtype)
    batch_ctx = (imgs, bxs, bxs_idx, out)
    for i in range(num_boxes):
        _extract_single_box(np_mod, batch_ctx, i, config)
    return cast(EagerValue, out)


def _extract_boxes_tf(backend_module: EagerValue, images: EagerValue, boxes: EagerValue, box_indices: EagerValue, config: BBoxConfig) -> EagerValue:
    """Execute the eager extract boxes tf operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        images: The images parameter.
        boxes: The boxes parameter.
        box_indices: The box_indices parameter.
        config (BBoxConfig): The config parameter.

    Returns:
            EagerValue: Result.
    """
    mod = _resolve_filter_mod(backend_module)
    images_np = mod.asarray(images)
    boxes_np = mod.asarray(boxes)
    box_indices_np = mod.asarray(box_indices)
    res = _extract_boxes_batch(mod, images_np, boxes_np, box_indices_np, config)
    return getattr(backend_module, "asarray")(res) if hasattr(backend_module, "asarray") else res


def extract_bounding_boxes_eager(backend_module: EagerValue, images: EagerValue, boxes: EagerValue, box_indices: EagerValue, config: BBoxConfig) -> EagerValue:
    """Execute the eager extract bounding boxes eager operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        images: The images parameter.
        boxes: The boxes parameter.
        box_indices: The box_indices parameter.
        config (BBoxConfig): The config parameter.

    Returns:
            EagerValue: Result.
    """
    return _extract_boxes_tf(backend_module, images, boxes, box_indices, config)


def _to_xyxy_format(np_mod: EagerValue, boxes: EagerValue, format: str) -> EagerValue:
    """Execute the eager to xyxy format operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        boxes: The boxes parameter.
        format (str): The format parameter.

    Returns:
            EagerValue: Result.

    Raises:
        ValueError: An exception.
    """
    boxes = getattr(np_mod, "asarray")(boxes)
    if format == "xyxy":
        return boxes
    elif format == "xyWH":
        boxes_arr = cast(EagerTensorProtocol, boxes)
        x, y, w, h = boxes_arr[..., 0], boxes_arr[..., 1], boxes_arr[..., 2], boxes_arr[..., 3]
        return cast(EagerValue, getattr(np_mod, "stack")([x, y, x + w, y + h], axis=-1))
    elif format == "cxcyWH":
        boxes_arr = cast(EagerTensorProtocol, boxes)
        cx, cy, w, h = boxes_arr[..., 0], boxes_arr[..., 1], boxes_arr[..., 2], boxes_arr[..., 3]
        return cast(EagerValue, getattr(np_mod, "stack")([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1))
    else:
        msg = f"Unknown format {format}"
        raise ValueError(msg)


def _compute_iou(np_mod: EagerValue, b1: EagerValue, b2: EagerValue) -> EagerValue:
    """Execute the eager compute iou operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        b1: The b1 parameter.
        b2: The b2 parameter.

    Returns:
            EagerValue: Result.
    """
    b1 = getattr(np_mod, "asarray")(b1)
    b2 = getattr(np_mod, "asarray")(b2)

    if getattr(b1, "size") == 0 or getattr(b2, "size") == 0:
        return cast(EagerValue, getattr(np_mod, "zeros")(cast(tuple[int, ...], getattr(b1, "shape"))[:-1], dtype=getattr(b1, "dtype")))

    x11, y11, x12, y12 = cast(
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol],
        getattr(np_mod, "split")(b1, 4, axis=-1),
    )
    x21, y21, x22, y22 = cast(
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol],
        getattr(np_mod, "split")(b2, 4, axis=-1),
    )

    xA = getattr(np_mod, "maximum")(x11, x21)
    yA = getattr(np_mod, "maximum")(y11, y21)
    xB = getattr(np_mod, "minimum")(x12, x22)
    yB = getattr(np_mod, "minimum")(y12, y22)

    interArea = getattr(np_mod, "maximum")(0, xB - xA) * getattr(np_mod, "maximum")(0, yB - yA)
    box1Area = (x12 - x11) * (y12 - y11)
    box2Area = (x22 - x21) * (y22 - y21)

    iou = interArea / (box1Area + box2Area - interArea + 1e-9)
    return cast(EagerValue, iou.reshape(iou.shape[:-1]))


def iou_eager(backend_module: EagerValue, boxes1: EagerValue, boxes2: EagerValue, bounding_box_format: str = "xyxy") -> EagerValue:
    """Execute the eager iou eager operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        boxes1: The boxes1 parameter.
        boxes2: The boxes2 parameter.
        bounding_box_format (str): The bounding_box_format parameter.

    Returns:
            EagerValue: Result.
    """
    mod = _resolve_filter_mod(backend_module)
    b1 = _to_xyxy_format(mod, boxes1, bounding_box_format)
    b2 = _to_xyxy_format(mod, boxes2, bounding_box_format)
    res = _compute_iou(mod, b1, b2)
    return getattr(backend_module, "asarray")(res) if hasattr(backend_module, "asarray") else res


def _sort_boxes_by_score(np_mod: EagerValue, boxes: EagerValue, scores: EagerValue, score_threshold: float) -> tuple[EagerValue, EagerValue, EagerValue, EagerValue]:
    """Execute the eager sort boxes by score operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        boxes: The boxes parameter.
        scores: The scores parameter.
        score_threshold (float): The score_threshold parameter.

    Returns:
            tuple[object, object, object, object]: Result.
    """
    boxes_arr: EagerTensorProtocol = getattr(np_mod, "asarray")(boxes)
    scores_arr: EagerTensorProtocol = getattr(np_mod, "asarray")(scores)

    if scores_arr.size == 0:
        return boxes_arr, scores_arr, getattr(np_mod, "array")([], dtype=int), getattr(np_mod, "array")([], dtype=int)

    mask = scores_arr >= score_threshold
    filtered_boxes = boxes_arr[mask]
    filtered_scores = scores_arr[mask]
    indices = getattr(np_mod, "where")(mask)[0]

    order = filtered_scores.argsort()[::-1]
    return filtered_boxes, filtered_scores, indices, order


def _compute_overlap(np_mod: EagerValue, bxs: EagerValue, i: int, order: EagerValue) -> EagerValue:
    """Execute the eager compute overlap operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        bxs: The bxs parameter.
        i (int): The i parameter.
        order: The order parameter.

    Returns:
            EagerValue: Result.
    """
    bxs_arr = cast(EagerTensorProtocol, bxs)
    order_arr = cast(EagerTensorProtocol, order)
    b1 = bxs_arr[order_arr[i]]
    b2 = bxs_arr[order_arr[1:]]
    return _compute_iou(np_mod, b1, b2)


def _apply_suppression_threshold(np_mod: EagerValue, bxs: EagerValue, order: EagerValue, max_output_size: int, iou_threshold: float) -> EagerValue:
    """Execute the eager apply suppression threshold operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        bxs: The bxs parameter.
        order: The order parameter.
        max_output_size (int): The max_output_size parameter.
        iou_threshold (float): The iou_threshold parameter.

    Returns:
            EagerValue: Result.
    """
    bxs_arr = cast(EagerTensorProtocol, bxs)
    keep: list[int] = []
    order_list = list(getattr(order, "__iter__")())
    while len(order_list) > 0 and len(keep) < max_output_size:
        i = order_list.pop(0)
        keep.append(i)
        if len(order_list) == 0:
            break
        # Compute IoU of the kept box with the remaining boxes
        b1 = bxs_arr[i]
        b2 = bxs_arr[order_list]
        iou = cast(EagerTensorProtocol, _compute_iou(np_mod, b1, b2))

        # Keep those with IoU <= threshold
        to_keep = iou <= iou_threshold
        order_list = [order_list[j] for j in range(len(order_list)) if to_keep[j]]

    return cast(EagerValue, getattr(np_mod, "array")(keep, dtype=getattr(np_mod, "int64")))


def _nms_tf(backend_module: EagerValue, boxes: EagerValue, scores: EagerValue, config: NMSConfig | None = None) -> EagerValue:
    """Execute the eager nms tf operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        boxes: The boxes parameter.
        scores: The scores parameter.
        config (Optional[NMSConfig]): The config parameter.

    Returns:
            EagerValue: Result.
    """
    if config is None:
        config = NMSConfig(max_output_size=cast(EagerTensorProtocol, boxes).shape[0] if hasattr(boxes, "shape") else 100)
    res = _nms_torch(boxes, scores, config, backend_module)
    return getattr(backend_module, "asarray")(res) if hasattr(backend_module, "asarray") else res


def _nms_torch(boxes: EagerValue, scores: EagerValue, config: NMSConfig, backend_module: EagerValue | None = None) -> EagerValue:
    """Execute the eager nms torch operation on the active backend array.

    Args:
        boxes: The boxes parameter.
        scores: The scores parameter.
        config (NMSConfig): The config parameter.
        backend_module: Optional execution backend module.

    Returns:
            EagerValue: Result.
    """
    mod = _resolve_filter_mod(backend_module)
    boxes = mod.asarray(boxes)
    scores = mod.asarray(scores)
    bxs, _, idxs, order = _sort_boxes_by_score(mod, boxes, scores, config.score_threshold)
    if getattr(bxs, "__len__")() == 0:
        return cast(EagerValue, mod.array([], dtype=mod.int64))
    kept_local = _apply_suppression_threshold(mod, bxs, order, config.max_output_size, config.iou_threshold)
    return cast(EagerTensorProtocol, idxs)[kept_local]


def nms_eager(backend_module: EagerValue, boxes: EagerValue, scores: EagerValue, config: NMSConfig | None = None) -> EagerValue:
    """Execute the eager nms eager operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        boxes: The boxes parameter.
        scores: The scores parameter.
        config (Optional[NMSConfig]): The config parameter.

    Returns:
            EagerValue: Result.
    """
    return _nms_tf(backend_module, boxes, scores, config)
