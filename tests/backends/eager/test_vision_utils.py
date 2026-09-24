"""Comprehensive unit tests for vision coordinate mapping and geometric utilities."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.vision_utils import (
    EagerTransformContext,
    GeometricGridConfig,
    MapCoordsContext,
    PerspectiveChannelContext,
    PerspectiveConfig,
    PerspectiveContext,
    RandomCropConfig,
    ResizeContext,
    TransformInterpolationConfig,
    _apply_perspective_batch,
    _apply_perspective_channel,
    _check_collinear_quad,
    _compute_perspective_matrix,
    _cubic_kernel,
    _generate_perspective_coords,
    _generate_perspective_grid,
    _map_coords_bicubic,
    _map_coords_bilinear,
    _map_coords_nearest,
    _np_map_coordinates,
    _prepare_eager_transform,
)


def test_prepare_eager_transform() -> None:
    """Test _prepare_eager_transform across tensor ranks, data formats, and seeds."""
    # None images returns safe zero-dimension context
    ctx_none = _prepare_eager_transform(None, None, seed=42, data_format=None)
    assert ctx_none.B == 0
    assert ctx_none.imgs.shape == (0, 0, 0, 0)
    assert ctx_none.rng is not None

    # 2D image (H, W)
    img_2d = np.ones((8, 12), dtype=np.float32)
    ctx_2d = _prepare_eager_transform(np, img_2d, seed=None, data_format=None)
    assert ctx_2d.B == 1
    assert ctx_2d.H == 8
    assert ctx_2d.W == 12
    assert ctx_2d.C == 1

    # 3D image channels-last (H, W, C)
    img_3d_last = np.ones((8, 12, 3), dtype=np.float32)
    ctx_3d_last = _prepare_eager_transform(np, img_3d_last, seed=123, data_format="channels_last")
    assert ctx_3d_last.B == 1
    assert ctx_3d_last.H == 8
    assert ctx_3d_last.W == 12
    assert ctx_3d_last.C == 3

    # 3D image channels-first (C, H, W)
    img_3d_first = np.ones((3, 8, 12), dtype=np.float32)
    ctx_3d_first = _prepare_eager_transform(np, img_3d_first, seed=None, data_format="channels_first")
    assert ctx_3d_first.B == 1
    assert ctx_3d_first.H == 8
    assert ctx_3d_first.W == 12
    assert ctx_3d_first.C == 3

    # 4D image channels-first (B, C, H, W)
    img_4d_first = np.ones((2, 3, 8, 12), dtype=np.float32)
    ctx_4d_first = _prepare_eager_transform(np, img_4d_first, seed=None, data_format="channels_first")
    assert ctx_4d_first.B == 2
    assert ctx_4d_first.H == 8
    assert ctx_4d_first.W == 12
    assert ctx_4d_first.C == 3

    # Pass list input to test non-ndarray conversion
    ctx_list = _prepare_eager_transform(np, [[1.0, 2.0], [3.0, 4.0]], seed=None, data_format=None)
    assert ctx_list.B == 1


def test_collinear_check_and_perspective_matrix() -> None:
    """Test homography estimation, DLT solver, collinearity detection, and projective invariance."""
    # None inputs
    assert _compute_perspective_matrix(None, None, None) == 0

    # Unit square mapping to 2x scaled square
    src_pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    dst_pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float32)

    h_mat = _compute_perspective_matrix(np, src_pts, dst_pts)
    assert h_mat.shape == (3, 3)

    # Validate projective mapping: p_dst ~ H @ p_src
    for s_pt, d_pt in zip(src_pts, dst_pts):
        homog_s = np.array([s_pt[0], s_pt[1], 1.0])
        mapped = h_mat @ homog_s
        mapped_xy = mapped[:2] / mapped[2]
        assert np.allclose(mapped_xy, d_pt, atol=1e-5)

    # Batched point correspondences (B=2, 4, 2)
    src_batch = np.stack([src_pts, src_pts], axis=0)
    dst_batch = np.stack([dst_pts, dst_pts], axis=0)
    h_batch = _compute_perspective_matrix(np, src_batch, dst_batch)
    assert h_batch.shape == (2, 3, 3)

    # Collinear points rejection
    collinear_src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 1.0]], dtype=np.float32)
    assert _check_collinear_quad(collinear_src) is True
    assert _check_collinear_quad(src_pts) is False

    with pytest.raises(ValueError, match="Collinear points detected"):
        _compute_perspective_matrix(np, collinear_src, dst_pts)

    with pytest.raises(ValueError, match="Collinear points detected"):
        _compute_perspective_matrix(np, src_pts, collinear_src)

    # Mock SVD singular matrix branch
    class MockSingularLinalg:
        @staticmethod
        def svd(a):
            return np.eye(8), np.zeros(8), np.eye(9)

    class MockNP:
        zeros = np.zeros
        float64 = np.float64
        float32 = np.float32
        asarray = staticmethod(np.asarray)
        stack = staticmethod(np.stack)
        linalg = MockSingularLinalg

    with pytest.raises(ValueError, match="Singular matrix encountered"):
        _compute_perspective_matrix(MockNP, src_pts, dst_pts)


def test_perspective_grid_and_coords() -> None:
    """Test perspective grid creation and backward coordinate mapping."""
    # Degenerate grid sizes
    assert _generate_perspective_grid(None, 0, 0) == 0
    assert _generate_perspective_grid(np, -2, 5) == 0

    grid = _generate_perspective_grid(np, 4, 6)
    assert grid.shape == (4, 6, 3)
    assert np.all(grid[..., 2] == 1.0)
    assert grid[0, 0, 0] == 0.0 and grid[0, 0, 1] == 0.0
    assert grid[3, 5, 0] == 5.0 and grid[3, 5, 1] == 3.0

    # Coordinate mapping with None inputs
    assert _generate_perspective_coords(None, None, None) == 0

    # Identity 2D homography
    h_eye = np.eye(3, dtype=np.float32)
    ys, xs = _generate_perspective_coords(np, h_eye, grid)
    assert ys.shape == (4, 6)
    assert xs.shape == (4, 6)
    assert np.allclose(ys, grid[..., 1])
    assert np.allclose(xs, grid[..., 0])

    # Batched homography
    h_batch = np.stack([h_eye, h_eye], axis=0)
    ys_b, xs_b = _generate_perspective_coords(np, h_batch, grid)
    assert ys_b.shape == (2, 4, 6)
    assert xs_b.shape == (2, 4, 6)


def test_map_coordinates_and_cubic_kernel() -> None:
    """Test nearest, bilinear, and bicubic coordinate mapping and cubic interpolation kernel."""
    # None inputs
    assert _map_coords_nearest(None) == 0
    assert _map_coords_bilinear(None) == 0
    assert _map_coords_bicubic(None) == 0
    assert _np_map_coordinates(None, None, None) == 0

    # Cubic kernel evaluation
    t_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.5])
    w = _cubic_kernel(t_vals, a=-0.5)
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[2], 0.0)
    assert np.isclose(w[4], 0.0)

    # 4x4 image
    img = np.arange(16, dtype=np.float32).reshape(4, 4)

    # In-bounds and out-of-bounds coordinates
    y_coords = np.array([0.0, 1.5, 5.0])
    x_coords = np.array([0.0, 1.5, -2.0])

    # Nearest neighbor (order=0)
    res_nearest = _np_map_coordinates(np, img, (y_coords, x_coords), order=0, fill_value=-1.0)
    assert res_nearest[0] == img[0, 0]
    assert res_nearest[2] == -1.0

    # Bilinear (order=1)
    res_bilinear = _np_map_coordinates(np, img, (y_coords, x_coords), order=1, fill_value=99.0)
    expected_mid = (img[1, 1] + img[1, 2] + img[2, 1] + img[2, 2]) / 4.0
    assert np.isclose(res_bilinear[1], expected_mid)
    assert res_bilinear[2] == 99.0

    # Bicubic (order=3)
    res_bicubic = _np_map_coordinates(np, img, (y_coords, x_coords), order=3, fill_value=0.0)
    assert res_bicubic.shape == (3,)
    assert np.isclose(res_bicubic[0], img[0, 0], atol=1e-4)


def test_perspective_apply_batch() -> None:
    """Test applying perspective transformation over channels and batches."""
    assert _apply_perspective_channel(None) == 0
    assert _apply_perspective_batch(None, None, None, None) == 0

    imgs = np.ones((2, 8, 8, 3), dtype=np.float32)
    h_eye = np.eye(3, dtype=np.float32)
    cfg = PerspectiveConfig(interpolation="bilinear", fill_value=0.0)

    out = _apply_perspective_batch(np, imgs, h_eye, cfg)
    assert out.shape == (2, 8, 8, 3)
    assert np.allclose(out, imgs)


def test_dataclasses_repr() -> None:
    """Test instantiation and attribute storage of vision utility dataclasses."""
    c1 = RandomCropConfig(crop_h=2, crop_w=3, b=0, c=1, H=10, W=10)
    assert c1.crop_h == 2

    c2 = GeometricGridConfig(H=5, W=5)
    assert c2.H == 5

    c3 = EagerTransformContext(B=1, H=4, W=4, C=3, name="test")
    assert c3.B == 1

    c4 = TransformInterpolationConfig(order=1, fill_value=0.0)
    assert c4.order == 1

    c5 = ResizeContext(H=10, W=10, new_H=20, new_W=20, align_corners=True)
    assert c5.align_corners is True

    c6 = MapCoordsContext(image=None)
    assert c6.image is None

    c7 = PerspectiveContext(b=0)
    assert c7.b == 0

    c8 = PerspectiveChannelContext(ctx=c7, config=PerspectiveConfig())
    assert c8.config.interpolation == "bilinear"
