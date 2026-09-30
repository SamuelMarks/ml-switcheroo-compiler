"""Tests for test_vision_geometry_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_geometry as geom_mod


def test_vision_geometry_ops() -> None:
    """Test all vision geometry operations and branches in vision_geometry.

    Returns:
        None
    """
    # AffineGenerator
    aff_gen = geom_mod._np_affine_generator(np, 2, 0.0, 0.0, 1.0)
    assert aff_gen.shape == (2, 8)
    assert numpy_eager_registry.get("AffineGenerator")(np, 2, 0.0, 0.0, 1.0).shape == (2, 8)

    # ElasticTransform
    # Branch 1: empty displacement
    img_2d = np.ones((4, 4))
    assert np.array_equal(geom_mod._np_elastic_transform(np, img_2d, np.array([])), img_2d)

    # Branch 2: 1D array
    img_1d = np.ones(4)
    disp_valid = np.ones((4, 4, 2))
    assert np.array_equal(geom_mod._np_elastic_transform(np, img_1d, disp_valid), img_1d)

    # Branch 3: 2D array with 2-channel displacement
    el_res = geom_mod._np_elastic_transform(np, img_2d, disp_valid)
    assert el_res.shape == (4, 4)

    # Branch 4: 2D array with 1-channel displacement
    disp_1ch = np.ones((4, 4, 1))
    el_res_1ch = geom_mod._np_elastic_transform(np, img_2d, disp_1ch)
    assert el_res_1ch.shape == (4, 4)
    assert numpy_eager_registry.get("ElasticTransform")(np, img_2d, disp_valid).shape == (4, 4)

    # ExtractBoundingBoxes
    boxes = np.array([[0, 0, 1, 1]])
    box_idx = np.array([0])
    assert np.array_equal(geom_mod._np_extract_bounding_boxes(np, img_2d, boxes, box_idx), img_2d)
    assert numpy_eager_registry.get("ExtractBoundingBoxes")(np, img_2d, boxes, box_idx) is not None

    # IoU
    boxes1 = np.array([[0.0, 0.0, 1.0, 1.0]])
    boxes2 = np.array([[0.0, 0.0, 1.0, 1.0]])
    iou = geom_mod._np_iou(np, boxes1, boxes2)
    assert iou.shape == (1,)
    assert numpy_eager_registry.get("IoU")(np, boxes1, boxes2) is not None

    # NonMaxSuppression
    scores = np.array([0.9])
    nms_res = geom_mod._np_nms(np, boxes1, scores, 1)
    assert len(nms_res) >= 1
    assert numpy_eager_registry.get("NonMaxSuppression")(np, boxes1, scores, 1) is not None

    # PerspectiveTransform
    pts1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    pts2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    persp = geom_mod._np_perspective_transform(np, img_2d, pts1, pts2, None)
    assert np.array_equal(persp, img_2d)
    assert numpy_eager_registry.get("PerspectiveTransform")(np, img_2d, pts1, pts2, None) is not None

    # AffineGrid
    # Array with len(size) == 4
    theta = np.zeros((1, 2, 3), dtype=np.float32)
    grid4 = geom_mod._np_affine_grid(np, theta, (1, 3, 4, 4))
    assert grid4.shape == (1, 4, 4, 2)

    # Array with len(size) != 4
    grid3 = geom_mod._np_affine_grid(np, theta, (1, 4, 4))
    assert grid3.shape == (1, 4, 4, 2)

    # Non-array theta
    assert geom_mod._np_affine_grid(np, 5.0, (1, 4, 4)) == 5.0
    assert numpy_eager_registry.get("AffineGrid")(np, theta, (1, 3, 4, 4)).shape == (1, 4, 4, 2)

    # AffineTransform
    trans = np.eye(3)
    aff_res = geom_mod._np_affine_transform(np, img_2d, trans)
    assert np.array_equal(aff_res, img_2d)
    assert numpy_eager_registry.get("AffineTransform")(np, img_2d, trans) is not None
