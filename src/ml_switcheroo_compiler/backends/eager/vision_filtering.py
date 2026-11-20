"""Vision spatial patch extraction, bounding box cropping, IoU, and NMS eager kernels."""

from __future__ import annotations

import types

import pydantic

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int_list,
    coerce_shape,
    global_eager_registry,
    unpack_quad,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.ops.configs import BBoxConfig


def _resolve_filter_mod(backend_module: BackendModuleProtocol | types.ModuleType | None) -> BackendModuleProtocol:
    """Resolve an array module providing `zeros`, `linspace`, `stack`, and `where`.

    Args:
        backend_module: Candidate backend module or None.

    Returns:
        BackendModuleProtocol: Resolved array module.
    """
    from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod

    return _resolve_eager_mod(backend_module)


@global_eager_registry.register("ExtractVolumePatches")
def _extract_volume_patches(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager extract volume patches operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    mod = _resolve_filter_mod(backend_module)
    if not args:
        return (
            adapt_backend_module(backend_module).require_symbol("asarray")([])
            if adapt_backend_module(backend_module).has_symbol("asarray")
            else mod.array([])
        )

    input_tensor: EagerTensorProtocol = mod.asarray(args[0])
    ksizes = coerce_int_list(
        kwargs.get("ksizes", args[1] if len(args) > 1 else [1, 1, 1, 1, 1])
        if adapt_backend_module(kwargs).has_symbol("get")
        else (args[1] if len(args) > 1 else [1, 1, 1, 1, 1])
    )
    strides = coerce_int_list(
        kwargs.get("strides", args[2] if len(args) > 2 else [1, 1, 1, 1, 1])
        if adapt_backend_module(kwargs).has_symbol("get")
        else (args[2] if len(args) > 2 else [1, 1, 1, 1, 1])
    )

    if input_tensor.ndim != 5:
        return (
            adapt_backend_module(backend_module).require_symbol("asarray")(input_tensor)
            if adapt_backend_module(backend_module).has_symbol("asarray")
            else input_tensor
        )

    shape = input_tensor.shape
    if any(int(k) <= 0 or int(k) > int(s) for k, s in zip(ksizes, shape)):
        return (
            adapt_backend_module(backend_module).require_symbol("asarray")(input_tensor)
            if adapt_backend_module(backend_module).has_symbol("asarray")
            else input_tensor
        )

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

    return (
        adapt_backend_module(backend_module).require_symbol("asarray")(res)
        if adapt_backend_module(backend_module).has_symbol("asarray")
        else res
    )


class NMSConfig(pydantic.BaseModel):
    """Config for NMS."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    max_output_size: int
    iou_threshold: float = 0.5
    score_threshold: float = float("-inf")


def _get_box_coords(
    np_mod: BackendModuleProtocol | types.ModuleType | None, box_ctx: tuple[int, int, int, int], box: BackendArg
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Retrieve the box coords property or mapping.

    Args:
        np_mod: NumPy module.
        box_ctx (tuple[int, int, int, int]): (height, width, crop_height, crop_width).
        box: Bounding box.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: ((y1, x1), (y2, x2)).
    """
    h, w = box_ctx[0], box_ctx[1]
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float, coerce_operand_list

    box_vals = [coerce_float(v) for v in coerce_operand_list(box)]
    y1 = int(box_vals[0] * (h - 1))
    x1 = int(box_vals[1] * (w - 1))
    y2 = int(box_vals[2] * (h - 1))
    x2 = int(box_vals[3] * (w - 1))
    return ((y1, x1), (y2, x2))


def _extract_box_channels(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    img: BackendArg,
    out: BackendArg,
    coords_and_i: tuple[tuple[tuple[int, int], tuple[int, int]], int],
    config: BBoxConfig,
) -> None:
    """Execute the eager extract box channels operation on the active backend array.

    Args:
        np_mod: Input `np_mod` argument (np mod).
        img: Input `img` argument (img).
        out: Input `out` argument (out).
        coords_and_i (tuple[tuple[tuple[int, int], tuple[int, int]], int]): Input `coords_and_i` argument (coords and
            i).
        config (BBoxConfig): Input `config` argument (config).

    Returns:
            None: Computed `None` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    (y1_i, x1_i), (y2_i, x2_i) = coords_and_i[0]
    i = coords_and_i[1]
    crop_h, crop_w = config.crop_size

    if y2_i <= y1_i or x2_i <= x1_i:
        return

    img_arr = coerce_array(np_mod, img)
    out_arr = coerce_array(np_mod, out)
    cropped = img_arr[y1_i : y2_i + 1, x1_i : x2_i + 1]
    if adapt_backend_module(cropped).get_symbol("size", 0) == 0:
        return

    y_indices = adapt_backend_module(np_mod).require_symbol("linspace")(0, cropped.shape[0] - 1, crop_h).astype(int)
    x_indices = adapt_backend_module(np_mod).require_symbol("linspace")(0, cropped.shape[1] - 1, crop_w).astype(int)
    resized = cropped[y_indices][:, x_indices]
    out_arr[i] = resized


def _extract_single_box(
    np_mod: BackendModuleProtocol | types.ModuleType | None, batch_ctx: BackendArg, i: int, config: BBoxConfig
) -> None:
    """Execute the eager extract single box operation on the active backend array.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        batch_ctx (BackendArg): Input `batch_ctx` argument (batch ctx).
        i (int): Input `i` argument (i).
        config (BBoxConfig): Input `config` argument (config).

    Returns:
        None: Computed `None` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_int, coerce_operand_list, coerce_shape

    imgs_raw, bxs_raw, bxs_idx_raw, out = unpack_quad(batch_ctx)
    imgs = coerce_operand_list(imgs_raw)
    bxs = coerce_operand_list(bxs_raw)
    bxs_idx = coerce_operand_list(bxs_idx_raw)
    img_idx = coerce_int(bxs_idx[i])
    if img_idx >= len(imgs):
        return
    img = imgs[img_idx]
    box = bxs[i]
    img_shape = coerce_shape(adapt_backend_module(img).require_symbol("shape"))
    if len(img_shape) < 2:
        return
    h, w = img_shape[0], img_shape[1]
    box_ctx = (h, w, config.crop_size[0], config.crop_size[1])
    coords = _get_box_coords(np_mod, box_ctx, box)
    _extract_box_channels(np_mod, img, out, (coords, i), config)


def _extract_boxes_batch(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    imgs: BackendArg,
    bxs: BackendArg,
    bxs_idx: BackendArg,
    config: BBoxConfig,
) -> BackendArg:
    """Extract bounding boxes for a batch.

    Args:
        np_mod: NumPy module.
        imgs: Images.
        bxs: Boxes.
        bxs_idx: Box indices.
        config (BBoxConfig): Configuration.

    Returns: BackendArg: Extracted boxes.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array, coerce_operand_list

    imgs_arr = coerce_array(np_mod, imgs)
    num_boxes = len(coerce_operand_list(bxs))
    if imgs_arr.ndim == 0:
        return adapt_backend_module(np_mod).require_symbol("zeros")(
            (num_boxes, config.crop_size[0], config.crop_size[1], 1), dtype=imgs_arr.dtype
        )

    channels = imgs_arr.shape[-1] if imgs_arr.ndim > 2 else 1
    out_shape = (num_boxes, config.crop_size[0], config.crop_size[1], channels)
    out = adapt_backend_module(np_mod).require_symbol("zeros")(out_shape, dtype=imgs_arr.dtype)
    batch_ctx = (imgs, bxs, bxs_idx, out)
    for i in range(num_boxes):
        _extract_single_box(np_mod, batch_ctx, i, config)
    return out


def _extract_boxes_tf(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    boxes: BackendArg,
    box_indices: BackendArg,
    config: BBoxConfig,
) -> BackendArg:
    """Execute the eager extract boxes tf operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        images: Input `images` argument (images).
        boxes: Input `boxes` argument (boxes).
        box_indices: Input `box_indices` argument (box indices).
        config (BBoxConfig): Input `config` argument (config).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = _resolve_filter_mod(backend_module)
    images_np = mod.asarray(images)
    boxes_np = mod.asarray(boxes)
    box_indices_np = mod.asarray(box_indices)
    res = _extract_boxes_batch(mod, images_np, boxes_np, box_indices_np, config)
    return (
        adapt_backend_module(backend_module).require_symbol("asarray")(res)
        if adapt_backend_module(backend_module).has_symbol("asarray")
        else res
    )


def extract_bounding_boxes_eager(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    boxes: BackendArg,
    box_indices: BackendArg,
    config: BBoxConfig,
) -> BackendArg:
    """Execute the eager extract bounding boxes eager operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        images: Input `images` argument (images).
        boxes: Input `boxes` argument (boxes).
        box_indices: Input `box_indices` argument (box indices).
        config (BBoxConfig): Input `config` argument (config).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _extract_boxes_tf(backend_module, images, boxes, box_indices, config)


def _to_xyxy_format(
    np_mod: BackendModuleProtocol | types.ModuleType | None, boxes: BackendArg, format: str
) -> BackendArg:
    """Execute the eager to xyxy format operation on the active backend array.

    Args:
        np_mod: Input `np_mod` argument (np mod).
        boxes: Input `boxes` argument (boxes).
        format (str): Input `format` argument (format).

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    boxes = adapt_backend_module(np_mod).require_symbol("asarray")(boxes)
    if format == "xyxy":
        return boxes
    elif format == "xyWH":
        boxes_arr = boxes
        x, y, w, h = boxes_arr[..., 0], boxes_arr[..., 1], boxes_arr[..., 2], boxes_arr[..., 3]
        return adapt_backend_module(np_mod).require_symbol("stack")([x, y, x + w, y + h], axis=-1)
    elif format == "cxcyWH":
        boxes_arr = boxes
        cx, cy, w, h = boxes_arr[..., 0], boxes_arr[..., 1], boxes_arr[..., 2], boxes_arr[..., 3]
        return adapt_backend_module(np_mod).require_symbol("stack")(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
        )
    else:
        msg = f"Unknown format {format}"
        raise ValueError(msg)


def _compute_iou(np_mod: BackendModuleProtocol | types.ModuleType | None, b1: BackendArg, b2: BackendArg) -> BackendArg:
    """Execute the eager compute iou operation on the active backend array.

    Args:
        np_mod: Input `np_mod` argument (np mod).
        b1: Input `b1` argument (b1).
        b2: Input `b2` argument (b2).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    b1 = adapt_backend_module(np_mod).require_symbol("asarray")(b1)
    b2 = adapt_backend_module(np_mod).require_symbol("asarray")(b2)

    if adapt_backend_module(b1).require_symbol("size") == 0 or adapt_backend_module(b2).require_symbol("size") == 0:
        return adapt_backend_module(np_mod).require_symbol("zeros")(
            coerce_shape(adapt_backend_module(b1).require_symbol("shape"))[:-1],
            dtype=adapt_backend_module(b1).require_symbol("dtype"),
        )

    x11, y11, x12, y12 = unpack_quad(adapt_backend_module(np_mod).require_symbol("split")(b1, 4, axis=-1))
    x21, y21, x22, y22 = unpack_quad(adapt_backend_module(np_mod).require_symbol("split")(b2, 4, axis=-1))

    xA = adapt_backend_module(np_mod).require_symbol("maximum")(x11, x21)
    yA = adapt_backend_module(np_mod).require_symbol("maximum")(y11, y21)
    xB = adapt_backend_module(np_mod).require_symbol("minimum")(x12, x22)
    yB = adapt_backend_module(np_mod).require_symbol("minimum")(y12, y22)

    interArea = adapt_backend_module(np_mod).require_symbol("maximum")(0, xB - xA) * adapt_backend_module(
        np_mod
    ).require_symbol("maximum")(0, yB - yA)
    box1Area = (x12 - x11) * (y12 - y11)
    box2Area = (x22 - x21) * (y22 - y21)

    iou = interArea / (box1Area + box2Area - interArea + 1e-9)
    return iou.reshape(iou.shape[:-1])


def iou_eager(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    boxes1: BackendArg,
    boxes2: BackendArg,
    bounding_box_format: str = "xyxy",
) -> BackendArg:
    """Execute the eager iou eager operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        boxes1: Input `boxes1` argument (boxes1).
        boxes2: Input `boxes2` argument (boxes2).
        bounding_box_format (str): Input `bounding_box_format` argument (bounding box format).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = _resolve_filter_mod(backend_module)
    b1 = _to_xyxy_format(mod, boxes1, bounding_box_format)
    b2 = _to_xyxy_format(mod, boxes2, bounding_box_format)
    res = _compute_iou(mod, b1, b2)
    return (
        adapt_backend_module(backend_module).require_symbol("asarray")(res)
        if adapt_backend_module(backend_module).has_symbol("asarray")
        else res
    )


def _sort_boxes_by_score(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    boxes: BackendArg,
    scores: BackendArg,
    score_threshold: float,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]:
    """Execute the eager sort boxes by score operation on the active backend array.

    Args:
        np_mod: Input `np_mod` argument (np mod).
        boxes: Input `boxes` argument (boxes).
        scores: Input `scores` argument (scores).
        score_threshold (float): Input `score_threshold` argument (score threshold).

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: Filtered and sorted
            boxes, scores, indices, and order.
    """
    boxes_arr: EagerTensorProtocol = adapt_backend_module(np_mod).require_symbol("asarray")(boxes)
    scores_arr: EagerTensorProtocol = adapt_backend_module(np_mod).require_symbol("asarray")(scores)

    if scores_arr.size == 0:
        return (
            boxes_arr,
            scores_arr,
            adapt_backend_module(np_mod).require_symbol("array")([], dtype=int),
            adapt_backend_module(np_mod).require_symbol("array")([], dtype=int),
        )

    mask = scores_arr >= score_threshold
    filtered_boxes = boxes_arr[mask]
    filtered_scores = scores_arr[mask]
    indices = adapt_backend_module(np_mod).require_symbol("where")(mask)[0]

    order = filtered_scores.argsort()[::-1]
    return filtered_boxes, filtered_scores, indices, order


def _compute_overlap(
    np_mod: BackendModuleProtocol | types.ModuleType | None, bxs: BackendArg, i: int, order: BackendArg
) -> BackendArg:
    """Execute the eager compute overlap operation on the active backend array.

    Args:
        np_mod: Input `np_mod` argument (np mod).
        bxs: Input `bxs` argument (bxs).
        i (int): Input `i` argument (i).
        order: Input `order` argument (order).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    bxs_arr = coerce_array(np_mod, bxs)
    order_arr = coerce_array(np_mod, order)
    b1 = bxs_arr[order_arr[i]]
    b2 = bxs_arr[order_arr[1:]]
    return _compute_iou(np_mod, b1, b2)


def _apply_suppression_threshold(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    bxs: BackendArg,
    order: BackendArg,
    max_output_size: int,
    iou_threshold: float,
) -> BackendArg:
    """Execute the eager apply suppression threshold operation on the active backend array.

    Args:
        np_mod: Input `np_mod` argument (np mod).
        bxs: Input `bxs` argument (bxs).
        order: Input `order` argument (order).
        max_output_size (int): Input `max_output_size` argument (max output size).
        iou_threshold (float): Input `iou_threshold` argument (iou threshold).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    bxs_arr = coerce_array(np_mod, bxs)
    keep: list[int] = []
    order_list = [int(x) for x in coerce_int_list(order)]
    while len(order_list) > 0 and len(keep) < max_output_size:
        i = order_list.pop(0)
        keep.append(i)
        if len(order_list) == 0:
            break
        # Compute IoU of the kept box with the remaining boxes
        b1 = bxs_arr[i]
        b2 = bxs_arr[order_list]
        iou = coerce_array(np_mod, _compute_iou(np_mod, b1, b2))

        # Keep those with IoU <= threshold
        to_keep = iou <= iou_threshold
        order_list = [order_list[j] for j in range(len(order_list)) if bool(to_keep[j])]

    return adapt_backend_module(np_mod).require_symbol("array")(
        keep, dtype=adapt_backend_module(np_mod).require_symbol("int64")
    )


def _nms_tf(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    boxes: BackendArg,
    scores: BackendArg,
    config: NMSConfig | None = None,
) -> BackendArg:
    """Execute the eager nms tf operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        boxes: Input `boxes` argument (boxes).
        scores: Input `scores` argument (scores).
        config (Optional[NMSConfig]): Input `config` argument (config).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    if config is None:
        config = NMSConfig(
            max_output_size=coerce_shape(adapt_backend_module(boxes).require_symbol("shape"))[0]
            if adapt_backend_module(boxes).has_symbol("shape")
            else 100
        )
    res = _nms_torch(boxes, scores, config, backend_module)
    return (
        adapt_backend_module(backend_module).require_symbol("asarray")(res)
        if adapt_backend_module(backend_module).has_symbol("asarray")
        else res
    )


def _nms_torch(
    boxes: BackendArg,
    scores: BackendArg,
    config: NMSConfig,
    backend_module: BackendModuleProtocol | types.ModuleType | None = None,
) -> BackendArg:
    """Execute the eager nms torch operation on the active backend array.

    Args:
        boxes: Input `boxes` argument (boxes).
        scores: Input `scores` argument (scores).
        config (NMSConfig): Input `config` argument (config).
        backend_module: Optional execution backend module.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = _resolve_filter_mod(backend_module)
    boxes = mod.asarray(boxes)
    scores = mod.asarray(scores)
    bxs, _, idxs, order = _sort_boxes_by_score(mod, boxes, scores, config.score_threshold)
    if len(bxs) == 0:
        return mod.array([], dtype=mod.int64)
    kept_local = _apply_suppression_threshold(mod, bxs, order, config.max_output_size, config.iou_threshold)
    return idxs[kept_local]


def nms_eager(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    boxes: BackendArg,
    scores: BackendArg,
    config: NMSConfig | None = None,
) -> BackendArg:
    """Execute the eager nms eager operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        boxes: Input `boxes` argument (boxes).
        scores: Input `scores` argument (scores).
        config (Optional[NMSConfig]): Input `config` argument (config).

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _nms_tf(backend_module, boxes, scores, config)
