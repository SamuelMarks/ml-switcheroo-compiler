"""Exhaustive unit tests for vision transforms and upsampling eager operations."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.vision_augmentation import RotationConfig, random_rotation_eager
from ml_switcheroo_compiler.backends.eager.vision_transforms import (
    ElasticGridContext,
    _apply_elastic_batch,
    _apply_resize_batch,
    _compute_elastic_grid,
    _compute_resize_grid,
    _extract_upsample_size,
    _get_resize_interpolation_order,
    _invert_homography,
    _standardize_image_input,
    _upsample_1d_linear,
    _upsample_3d_trilinear,
    _upsample_bicubic_eager,
    _upsample_linear_eager,
    _upsample_nearest_eager,
    elastic_transform_eager,
    perspective_transform_eager,
    resize_eager,
)
from ml_switcheroo_compiler.backends.eager.vision_utils import (
    ResizeContext,
)
from ml_switcheroo_compiler.ops.configs import ElasticConfig, PerspectiveConfig, ResizeOptions


def test_standardize_and_homography_helpers() -> None:
    """Test image layout normalization and homography inversion routines."""
    # 2D -> 4D
    arr_2d = np.ones((4, 6), dtype=np.float32)
    std_2d, was_3d = _standardize_image_input(arr_2d, None)
    assert std_2d.shape == (1, 4, 6, 1)
    assert was_3d is False

    # 3D channels-last -> 4D
    arr_3d = np.ones((4, 6, 3), dtype=np.float32)
    std_3d, was_3d = _standardize_image_input(arr_3d, "channels_last")
    assert std_3d.shape == (1, 4, 6, 3)
    assert was_3d is True

    # 3D channels-first -> 4D
    arr_3d_first = np.ones((3, 4, 6), dtype=np.float32)
    std_3d_f, was_3d_f = _standardize_image_input(arr_3d_first, "channels_first")
    assert std_3d_f.shape == (1, 4, 6, 3)
    assert was_3d_f is True

    # 4D channels-first -> 4D channels-last
    arr_4d = np.ones((2, 3, 4, 6), dtype=np.float32)
    std_4d, was_3d_4d = _standardize_image_input(arr_4d, "channels_first")
    assert std_4d.shape == (2, 4, 6, 3)
    assert was_3d_4d is False

    # Homography inversion single and batched
    h_single = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    h_inv = _invert_homography(np, h_single)
    assert np.allclose(h_single @ h_inv, np.eye(3), atol=1e-5)

    h_batch = np.stack([h_single, h_single], axis=0)
    h_batch_inv = _invert_homography(np, h_batch)
    assert h_batch_inv.shape == (2, 3, 3)


def test_perspective_transform_eager() -> None:
    """Test perspective transform eager execution across ranks, formats, and point correspondences."""
    # None images returns 0
    assert perspective_transform_eager(None, None, None, None, None) == 0

    img = np.ones((8, 8, 3), dtype=np.float32)
    cfg = PerspectiveConfig(interpolation="bilinear", fill_value=0.0)

    # None start_points or end_points returns copy
    out_copy = perspective_transform_eager(np, img, None, None, cfg)
    assert out_copy.shape == (8, 8, 3)
    assert np.allclose(out_copy, img)

    # Valid points mapping
    src_pts = np.array([[0.0, 0.0], [7.0, 0.0], [7.0, 7.0], [0.0, 7.0]], dtype=np.float32)
    dst_pts = np.array([[0.0, 0.0], [7.0, 0.0], [7.0, 7.0], [0.0, 7.0]], dtype=np.float32)
    out_id = perspective_transform_eager(np, img, src_pts, dst_pts, cfg)
    assert out_id.shape == (8, 8, 3)
    assert np.allclose(out_id, img)

    # Channels first layout
    img_cf = np.ones((2, 3, 8, 8), dtype=np.float32)
    cfg_cf = PerspectiveConfig(interpolation="nearest", fill_value=0.0, data_format="channels_first")
    out_cf = perspective_transform_eager(np, img_cf, src_pts, dst_pts, cfg_cf)
    assert out_cf.shape == (2, 3, 8, 8)


def test_elastic_transform_eager() -> None:
    """Test elastic distortion grid generation, interpolation, and eager execution."""
    # None inputs
    assert elastic_transform_eager(None, None, None, None) == 0
    assert _apply_elastic_batch(None, None, None) == 0
    assert _compute_elastic_grid(None) == 0

    # Grid context with zero displacement preserves base grid
    ctx_zero = ElasticGridContext(np_mod=np, H=6, W=8, B=1, disp=0)
    y_g, x_g = _compute_elastic_grid(ctx_zero)
    assert y_g.shape == (6, 8)
    assert x_g.shape == (6, 8)

    # Grid with 3D displacement (2, H, W)
    disp_3d = np.zeros((2, 6, 8), dtype=np.float32)
    ctx_disp = ElasticGridContext(np_mod=np, H=6, W=8, B=1, disp=disp_3d)
    y_d, x_d = _compute_elastic_grid(ctx_disp)
    assert np.allclose(y_d, y_g)

    # Grid with 3D displacement (H, W, 2)
    disp_last = np.zeros((6, 8, 2), dtype=np.float32)
    ctx_last = ElasticGridContext(np_mod=np, H=6, W=8, B=1, disp=disp_last)
    y_l, x_l = _compute_elastic_grid(ctx_last)
    assert np.allclose(y_l, y_g)

    # Grid with 4D displacement (B, 2, H, W)
    disp_4d_ch = np.zeros((1, 2, 6, 8), dtype=np.float32)
    ctx_4d_ch = ElasticGridContext(np_mod=np, H=6, W=8, B=1, disp=disp_4d_ch)
    y_4ch, x_4ch = _compute_elastic_grid(ctx_4d_ch)
    assert np.allclose(y_4ch, y_g)

    # Grid with 4D displacement (B, H, W, 2)
    disp_4d = np.zeros((1, 6, 8, 2), dtype=np.float32)
    ctx_4d = ElasticGridContext(np_mod=np, H=6, W=8, B=1, disp=disp_4d)
    y_4, x_4 = _compute_elastic_grid(ctx_4d)
    assert np.allclose(y_4, y_g)

    # Grid with non-standard displacement shape triggers fallback zeros
    ctx_bad = ElasticGridContext(np_mod=np, H=6, W=8, B=1, disp=np.zeros((5,)))
    y_bad, x_bad = _compute_elastic_grid(ctx_bad)
    assert np.allclose(y_bad, y_g)

    # Eager execution with None displacement preserves image
    img = np.ones((2, 6, 8, 3), dtype=np.float32)
    cfg = ElasticConfig(interpolation="bilinear", fill_value=0.0)
    res_none = elastic_transform_eager(np, img, None, cfg)
    assert np.allclose(res_none, img)

    # Eager execution with zero displacement
    res_zero = elastic_transform_eager(np, img, disp_4d, cfg)
    assert res_zero.shape == (2, 6, 8, 3)
    assert np.allclose(res_zero, img)

    # 3D image input to elastic transform returns 3D output
    img_single_3d = np.ones((6, 8, 3), dtype=np.float32)
    res_3d = elastic_transform_eager(np, img_single_3d, disp_4d, cfg)
    assert res_3d.shape == (6, 8, 3)

    # Channels first layout
    img_cf = np.ones((2, 3, 6, 8), dtype=np.float32)
    cfg_cf = ElasticConfig(interpolation="bicubic", fill_value=0.0, data_format="channels_first")
    res_cf = elastic_transform_eager(np, img_cf, disp_4d, cfg_cf)
    assert res_cf.shape == (2, 3, 6, 8)


def test_resize_and_interpolation_orders() -> None:
    """Test interpolation order mapping and spatial resize eager kernel."""
    assert _get_resize_interpolation_order("nearest") == 0
    assert _get_resize_interpolation_order("bicubic") == 3
    assert _get_resize_interpolation_order("lanczos3") == 3
    assert _get_resize_interpolation_order("bilinear") == 1
    assert _get_resize_interpolation_order("linear") == 1

    # Resize None cases
    assert _compute_resize_grid(None, None) == 0
    assert _apply_resize_batch(None, None, None, None, 1) == 0
    assert resize_eager(None, None, None, None) == 0

    img = np.ones((2, 8, 8, 3), dtype=np.float32)
    cfg = ResizeOptions(size=(16, 16), interpolation="bilinear", align_corners=False)

    res = resize_eager(np, img, (16, 16), cfg)
    assert res.shape == (2, 16, 16, 3)

    # Resize with align_corners=True and half_pixel_centers=True
    cfg_corners = ResizeOptions(size=(4, 4), interpolation="nearest", align_corners=True)
    res_corners = resize_eager(np, img, (4, 4), cfg_corners)
    assert res_corners.shape == (2, 4, 4, 3)

    # Resize channels_first layout
    img_cf = np.ones((2, 3, 8, 8), dtype=np.float32)
    cfg_cf = ResizeOptions(size=(12, 12), interpolation="bicubic", data_format="channels_first")
    res_cf = resize_eager(np, img_cf, (12, 12), cfg_cf)
    assert res_cf.shape == (2, 3, 12, 12)

    # 3D image input to resize_eager returns 3D output
    img_single = np.ones((8, 8, 3), dtype=np.float32)
    res_3d_resize = resize_eager(np, img_single, (16, 16), cfg)
    assert res_3d_resize.shape == (16, 16, 3)

    # Test _compute_resize_grid half_pixel_centers branch
    ctx_half = ResizeContext(H=8, W=8, new_H=16, new_W=16, align_corners=False, half_pixel_centers=True)
    grid_y, grid_x = _compute_resize_grid(np, ctx_half)
    assert grid_y.shape == (16, 16)


def test_upsampling_operations() -> None:
    """Test 1D, 2D, and 3D nearest, linear/bilinear/trilinear, and bicubic upsampling."""
    # Size extraction helper
    assert _extract_upsample_size((10, 20), size=15, scale_factor=None) == (15, 15)
    assert _extract_upsample_size((10, 20), size=(12, 24), scale_factor=None) == (12, 24)
    assert _extract_upsample_size((10, 20), size=None, scale_factor=2.0) == (20, 40)
    assert _extract_upsample_size((10, 20), size=None, scale_factor=(1.5, 2.0)) == (15, 40)
    assert _extract_upsample_size((10, 20), size=None, scale_factor=None) == (10, 20)

    # None inputs for eager upsampling
    assert _upsample_nearest_eager(None) == 0
    assert _upsample_linear_eager(None) == 0
    assert _upsample_bicubic_eager(None) == 0

    # 1D nearest upsample
    x_1d = np.ones((2, 4, 10), dtype=np.float32)
    up_1d_near = _upsample_nearest_eager(np, x_1d, size=(20,))
    assert up_1d_near.shape == (2, 4, 20)

    # 1D linear upsample
    up_1d_lin = _upsample_linear_eager(np, x_1d, size=(20,), align_corners=True)
    assert up_1d_lin.shape == (2, 4, 20)

    up_1d_unalign = _upsample_1d_linear(x_1d, 20, align_corners=False)
    assert up_1d_unalign.shape == (2, 4, 20)

    # 1D linear same length returns copy
    up_1d_same = _upsample_1d_linear(x_1d, 10, align_corners=False)
    assert np.allclose(up_1d_same, x_1d)

    # 2D nearest upsample
    x_2d = np.ones((2, 4, 8, 8), dtype=np.float32)
    up_2d_near = _upsample_nearest_eager(np, x_2d, scale_factor=2.0)
    assert up_2d_near.shape == (2, 4, 16, 16)

    # 2D bilinear upsample
    up_2d_bi = _upsample_linear_eager(np, x_2d, size=(12, 14), align_corners=False)
    assert up_2d_bi.shape == (2, 4, 12, 14)

    # 2D bicubic upsample
    up_2d_bic = _upsample_bicubic_eager(np, x_2d, size=(16, 16), align_corners=True)
    assert up_2d_bic.shape == (2, 4, 16, 16)

    # 3D nearest upsample
    x_3d = np.ones((1, 2, 4, 6, 8), dtype=np.float32)
    up_3d_near = _upsample_nearest_eager(np, x_3d, size=(8, 12, 16))
    assert up_3d_near.shape == (1, 2, 8, 12, 16)

    # 3D trilinear upsample
    up_3d_tri = _upsample_linear_eager(np, x_3d, size=(8, 12, 16), align_corners=True)
    assert up_3d_tri.shape == (1, 2, 8, 12, 16)

    up_3d_tri_noalign = _upsample_3d_trilinear(x_3d, (8, 12, 16), align_corners=False)
    assert up_3d_tri_noalign.shape == (1, 2, 8, 12, 16)

    # 4D spatial fallback copy branch
    x_4d_sp = np.ones((1, 2, 2, 2, 2, 2), dtype=np.float32)
    assert _upsample_nearest_eager(np, x_4d_sp).shape == x_4d_sp.shape
    assert _upsample_linear_eager(np, x_4d_sp).shape == x_4d_sp.shape


def test_rotation_eager_preserved() -> None:
    """Test compatibility with random_rotation_eager."""
    images = np.ones((2, 10, 10, 3), dtype=np.float32)
    cfg_rot = RotationConfig(
        factor=0.5,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=42,
        fill_value=0.0,
        data_format="channels_last",
    )
    res_rot = random_rotation_eager(np, images, cfg_rot)
    assert res_rot is not None

    cfg_near = RotationConfig(
        factor=0.5,
        fill_mode="reflect",
        interpolation="nearest",
        seed=42,
        fill_value=0.0,
        data_format="channels_last",
    )
    res_near = random_rotation_eager(np, images, cfg_near)
    assert res_near is not None
