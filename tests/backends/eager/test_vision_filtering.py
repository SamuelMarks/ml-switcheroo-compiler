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


def test_vision_filtering_stubs():
    assert _extract_volume_patches(None) == 0
    assert _extract_box_channels(None, None, None, None, None) == 0
    assert _get_box_coords(None, None, None) == 0
    assert _extract_single_box(None, None, None, None) == 0
    assert _extract_boxes_batch(None, None, None, None, None) == 0
    assert _extract_boxes_tf(None, None, None, None, None) == 0
    assert extract_bounding_boxes_eager(None, None, None, None, None) == 0
    assert _to_xyxy_format(None, None, None) == 0
    assert _compute_iou(None, None, None) == 0
    assert iou_eager(None, None, None) == 0
    assert _sort_boxes_by_score(None, None, None, None) == 0
    assert _compute_overlap(None, None, None, None) == 0
    assert _apply_suppression_threshold(None, None, None, None, None) == 0
    assert _nms_tf(None, None, None) == 0

    config = NMSConfig(max_output_size=1)
    assert _nms_torch(None, None, config) == 0
    assert nms_eager(None, None, None) == 0
