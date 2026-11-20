"""Behavioral unit tests for vision geometry."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager as eager_mod
from ml_switcheroo_compiler.backends.eager.vision_ops import geometry as geom_mod
from ml_switcheroo_compiler.backends.eager.vision_ops.geometry import (
    _np_affine_generator,
    _np_affine_grid,
    _np_affine_transform,
    _np_elastic_transform,
    _np_extract_bounding_boxes,
    _np_iou,
    _np_nms,
    _np_perspective_transform,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import extract_test_shape, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_np_affine_generator() -> None:
    """Test function test_np_affine_generator."""
    res = _np_affine_generator(np, 2, None, None, None)
    assert res.shape == (2, 8)


def test_np_elastic_transform() -> None:
    """Test function test_np_elastic_transform."""
    images = np.ones((2, 2))
    res = _np_elastic_transform(np, images, ensure_ndarray(None))
    np.testing.assert_allclose(res, images)


def test_np_extract_bounding_boxes() -> None:
    """Test function test_np_extract_bounding_boxes."""
    images = np.ones((2, 2))
    res = _np_extract_bounding_boxes(np, images, ensure_ndarray(None), ensure_ndarray(None))
    np.testing.assert_allclose(res, images)


def test_np_iou() -> None:
    """Test function test_np_iou."""

    @global_eager_registry.register("IoU")
    def _dummy_iou(bm: BackendArg, b1: BackendArg, b2: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function _dummy_iou.

        Args:
            bm: Test parameter `bm`.
            b1: Test parameter `b1`.
            b2: Test parameter `b2`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `_dummy_iou`.
        """
        return np.ones((1,))

    with pytest.raises(IndexError):
        _np_iou(np, ensure_ndarray(None), ensure_ndarray(None))


def test_np_nms() -> None:
    """Test function test_np_nms."""

    @global_eager_registry.register("NonMaxSuppression")
    def _dummy_nms(
        bm: BackendArg, b: BackendArg, s: BackendArg, max_out: BackendArg, **kwargs: BackendArg
    ) -> BackendArg:
        """Test function _dummy_nms.

        Args:
            bm: Test parameter `bm`.
            b: Test parameter `b`.
            s: Test parameter `s`.
            max_out: Test parameter `max_out`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `_dummy_nms`.
        """
        return np.ones((1,))

    with pytest.raises(TypeError):
        _np_nms(np, ensure_ndarray(None), ensure_ndarray(None), 1)


def test_np_perspective_transform() -> None:
    """Test function test_np_perspective_transform."""
    images = np.ones((2, 2))
    res = _np_perspective_transform(np, images, ensure_ndarray(None), ensure_ndarray(None), None)
    np.testing.assert_allclose(res, images)


def test_np_affine_grid() -> None:
    """Test function test_np_affine_grid."""
    res1 = _np_affine_grid(np, np.ones((2, 2)), (2, 2))
    assert extract_test_shape(res1) == (2, 2, 2)
    res2 = _np_affine_grid(np, "dummy", (2, 2))
    assert res2 == "dummy"


def test_np_affine_transform() -> None:
    """Test function test_np_affine_transform."""
    images = np.ones((2, 2))
    res = _np_affine_transform(np, images, ensure_ndarray(None))
    np.testing.assert_allclose(res, images)


def test_numpy_vision_geometry_eager_extra() -> None:
    """Test the numpy vision geometry eager extra behavior."""
    el_out = _np_elastic_transform(np, np.ones((2, 2, 3)), np.ones((2, 2, 2)))
    assert extract_test_shape(el_out) == (2, 2, 3)
    bb_out = _np_extract_bounding_boxes(np, np.ones((2, 2, 3)), np.array([0, 0, 10, 10]), np.array([0]))
    assert bb_out is not None
    original_iou = eager_mod.iou_eager

    def mock_iou(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Evaluate and process the mock iou operation.

        Args:
            *args (BackendArg): Variable positional arguments.
            **kwargs (BackendArg): Arbitrary keyword arguments.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        return args[1]

    eager_mod.iou_eager = mock_iou
    try:
        iou_out = _np_iou(np, np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
        assert iou_out is not None
    finally:
        eager_mod.iou_eager = original_iou
    original_nms = eager_mod.nms_eager

    def mock_nms(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Evaluate and process the mock nms operation.

        Args:
            *args (BackendArg): Variable positional arguments.
            **kwargs (BackendArg): Arbitrary keyword arguments.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        return args[1]

    eager_mod.nms_eager = mock_nms
    try:
        nms_out = _np_nms(np, np.array([[0, 0, 10, 10]]), np.array([0.9]), max_output_size=1, threshold=0.5)
        assert nms_out is not None
    finally:
        eager_mod.nms_eager = original_nms
    pt_out = _np_perspective_transform(np, np.ones((2, 2, 3)), np.eye(3), np.eye(3), config=None)
    assert extract_test_shape(pt_out) == (2, 2, 3)
    ag_out = _np_affine_grid(np, np.eye(2, 3)[None, ...], size=(1, 1, 4, 4))
    assert ag_out is not None
    at_out = _np_affine_transform(np, np.ones((2, 2, 3)), np.eye(2, 3))
    assert extract_test_shape(at_out) == (2, 2, 3)
    gen_out = _np_affine_generator(np, 2, np.ones(1), np.ones(1), np.ones(1))
    assert extract_test_shape(gen_out) == (2, 8)


def test_vision_geometry_ops() -> None:
    """Test all vision geometry operations and branches in vision_geometry.

    Returns:
        None
    """
    # AffineGenerator
    aff_gen = geom_mod._np_affine_generator(np, 2, 0.0, 0.0, 1.0)
    assert aff_gen.shape == (2, 8)
    assert require_callable(numpy_eager_registry.get)("AffineGenerator")(np, 2, 0.0, 0.0, 1.0).shape == (2, 8)

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
    assert require_callable(numpy_eager_registry.get)("ElasticTransform")(np, img_2d, disp_valid).shape == (4, 4)

    # ExtractBoundingBoxes
    boxes = np.array([[0, 0, 1, 1]])
    box_idx = np.array([0])
    assert np.array_equal(geom_mod._np_extract_bounding_boxes(np, img_2d, boxes, box_idx), img_2d)
    assert require_callable(numpy_eager_registry.get)("ExtractBoundingBoxes")(np, img_2d, boxes, box_idx) is not None

    # IoU
    boxes1 = np.array([[0.0, 0.0, 1.0, 1.0]])
    boxes2 = np.array([[0.0, 0.0, 1.0, 1.0]])
    iou = geom_mod._np_iou(np, boxes1, boxes2)
    assert iou.shape == (1,)
    assert require_callable(numpy_eager_registry.get)("IoU")(np, boxes1, boxes2) is not None

    # NonMaxSuppression
    scores = np.array([0.9])
    nms_res = geom_mod._np_nms(np, boxes1, scores, 1)
    assert len(nms_res) >= 1
    assert require_callable(numpy_eager_registry.get)("NonMaxSuppression")(np, boxes1, scores, 1) is not None

    # PerspectiveTransform
    pts1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    pts2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    persp = geom_mod._np_perspective_transform(np, img_2d, pts1, pts2, None)
    assert np.array_equal(persp, img_2d)
    assert require_callable(numpy_eager_registry.get)("PerspectiveTransform")(np, img_2d, pts1, pts2, None) is not None

    # AffineGrid
    # Array with len(size) == 4
    theta = np.zeros((1, 2, 3), dtype=np.float32)
    grid4 = geom_mod._np_affine_grid(np, theta, (1, 3, 4, 4))
    assert extract_test_shape(grid4) == (1, 4, 4, 2)

    # Array with len(size) != 4
    grid3 = geom_mod._np_affine_grid(np, theta, (1, 4, 4))
    assert extract_test_shape(grid3) == (1, 4, 4, 2)

    # Non-array theta
    assert geom_mod._np_affine_grid(np, 5.0, (1, 4, 4)) == 5.0
    assert require_callable(numpy_eager_registry.get)("AffineGrid")(np, theta, (1, 3, 4, 4)).shape == (1, 4, 4, 2)

    # AffineTransform
    trans = np.eye(3)
    aff_res = geom_mod._np_affine_transform(np, img_2d, trans)
    assert np.array_equal(aff_res, img_2d)
    assert require_callable(numpy_eager_registry.get)("AffineTransform")(np, img_2d, trans) is not None
