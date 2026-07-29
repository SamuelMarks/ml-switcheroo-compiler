import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.vision_filtering import (
    NMSConfig,
    _apply_suppression_threshold,
    _compute_iou,
    _compute_overlap,
    _extract_box_channels,
    _extract_boxes_batch,
    _extract_boxes_tf,
    _extract_single_box,
    _extract_volume_patches,
    _get_box_coords,
    _nms_tf,
    _nms_torch,
    _sort_boxes_by_score,
    _to_xyxy_format,
    extract_bounding_boxes_eager,
    iou_eager,
    nms_eager,
)
from ml_switcheroo_compiler.ops.configs import BBoxConfig


class DummyBackend:
    @staticmethod
    def asarray(x):
        return np.asarray(x)


def test_vision_filtering_volume_patches():
    # Empty args
    assert len(_extract_volume_patches(DummyBackend())) == 0

    # 1D array - should just return array
    arr_1d = np.array([1, 2, 3])
    assert np.array_equal(_extract_volume_patches(DummyBackend(), arr_1d), arr_1d)

    # 5D array - normal case
    arr_5d = np.arange(32).reshape(1, 2, 2, 2, 4)
    patches = _extract_volume_patches(DummyBackend(), arr_5d, ksizes=[1, 1, 1, 1, 1], strides=[1, 1, 1, 1, 1])
    assert patches.shape == (1, 2, 2, 2, 4, 1)

    # Invalid window size (ValueError caught)
    patches = _extract_volume_patches(DummyBackend(), arr_5d, ksizes=[1, 1, 1, 1, 5])
    assert np.array_equal(patches, arr_5d)


def test_vision_filtering_box_coords():
    coords = _get_box_coords(np, (10, 10, 5, 5), [0.0, 0.0, 1.0, 1.0])
    assert coords == ((0, 0), (9, 9))


def test_vision_filtering_box_channels():
    img = np.arange(100).reshape(10, 10)
    out = np.zeros((1, 5, 5))
    config = BBoxConfig(crop_size=(5, 5))

    # Normal
    _extract_box_channels(np, img, out, (((0, 0), (9, 9)), 0), config)
    assert out[0].shape == (5, 5)

    # Empty crop
    _extract_box_channels(np, img, out, (((5, 5), (4, 4)), 0), config)

    # 0 size
    img_empty = np.array([[]])
    _extract_box_channels(np, img_empty, out, (((0, 0), (9, 9)), 0), config)


def test_vision_filtering_single_box():
    imgs = [np.arange(100).reshape(10, 10)]
    bxs = [[0.0, 0.0, 1.0, 1.0]]
    bxs_idx = [0]
    out = np.zeros((1, 5, 5))
    config = BBoxConfig(crop_size=(5, 5))

    batch_ctx = (imgs, bxs, bxs_idx, out)
    _extract_single_box(np, batch_ctx, 0, config)

    # Out of bounds image index
    bxs_idx_oob = [1]
    _extract_single_box(np, (imgs, bxs, bxs_idx_oob, out), 0, config)

    # Image < 2D
    imgs_1d = [np.array([1])]
    _extract_single_box(np, (imgs_1d, bxs, bxs_idx, out), 0, config)


def test_vision_filtering_boxes_batch():
    imgs = np.arange(100).reshape(1, 10, 10, 1)
    bxs = np.array([[0.0, 0.0, 1.0, 1.0]])
    bxs_idx = np.array([0])
    config = BBoxConfig(crop_size=(5, 5))

    out = _extract_boxes_batch(np, imgs, bxs, bxs_idx, config)
    assert out.shape == (1, 5, 5, 1)

    # scalar imgs
    imgs_scalar = np.array(0)
    out2 = _extract_boxes_batch(np, imgs_scalar, bxs, bxs_idx, config)
    assert out2.shape == (1, 5, 5, 1)

    res = _extract_boxes_tf(DummyBackend(), imgs, bxs, bxs_idx, config)
    assert res.shape == (1, 5, 5, 1)

    res2 = extract_bounding_boxes_eager(DummyBackend(), imgs, bxs, bxs_idx, config)
    assert res2.shape == (1, 5, 5, 1)


def test_vision_filtering_xyxy_format():
    boxes = np.array([[0, 0, 10, 10]])

    assert np.array_equal(_to_xyxy_format(np, boxes, "xyxy"), boxes)
    assert np.array_equal(_to_xyxy_format(np, boxes, "xyWH"), np.array([[0, 0, 10, 10]]))
    assert np.array_equal(_to_xyxy_format(np, boxes, "cxcyWH"), np.array([[-5, -5, 5, 5]]))

    with pytest.raises(ValueError):
        _to_xyxy_format(np, boxes, "invalid")


def test_vision_filtering_iou():
    b1 = np.array([[0, 0, 10, 10]])
    b2 = np.array([[0, 0, 10, 10]])

    iou = _compute_iou(np, b1, b2)
    assert np.allclose(iou, 1.0)

    iou2 = iou_eager(DummyBackend(), b1, b2)
    assert np.allclose(iou2, 1.0)

    # Empty
    iou3 = _compute_iou(np, np.empty((0, 4)), np.empty((0, 4)))
    assert iou3.size == 0


def test_vision_filtering_sort_and_nms():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]])
    scores = np.array([0.9, 0.8, 0.95])

    b, s, i, o = _sort_boxes_by_score(np, boxes, scores, 0.85)
    assert len(b) == 2
    assert o[0] == 1  # 0.95 is max, which is index 1 of the filtered array (which had 0.9 and 0.95)

    # empty sort
    b2, s2, i2, o2 = _sort_boxes_by_score(np, np.array([]), np.array([]), 0.5)
    assert len(b2) == 0

    # NMS overlap
    ov = _compute_overlap(np, boxes, 0, [0, 1, 2])
    assert len(ov) == 2

    # suppression
    keep = _apply_suppression_threshold(np, boxes, [0, 1, 2], 2, 0.5)
    assert len(keep) <= 2

    config = NMSConfig(max_output_size=2, iou_threshold=0.5, score_threshold=0.85)
    res_torch = _nms_torch(boxes, scores, config)
    assert len(res_torch) == 2

    res_tf = _nms_tf(DummyBackend(), boxes, scores, config)
    assert len(res_tf) == 2

    # None config
    res_tf2 = _nms_tf(DummyBackend(), boxes, scores, None)
    assert len(res_tf2) > 0

    # Eager nms
    res_eager = nms_eager(DummyBackend(), boxes, scores, config)
    assert len(res_eager) == 2

    # Empty nms
    res_empty = _nms_torch(np.array([]), np.array([]), config)
    assert len(res_empty) == 0
