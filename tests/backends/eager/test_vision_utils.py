"""Test module."""

from ml_switcheroo_compiler.backends.eager.vision_utils import (
    EagerTransformContext,
    GeometricGridConfig,
    MapCoordsContext,
    PerspectiveChannelContext,
    PerspectiveContext,
    RandomCropConfig,
    ResizeContext,
    TransformInterpolationConfig,
    _apply_perspective_batch,
    _apply_perspective_channel,
    _compute_perspective_matrix,
    _generate_perspective_coords,
    _generate_perspective_grid,
    _map_coords_bilinear,
    _map_coords_nearest,
    _np_map_coordinates,
    _prepare_eager_transform,
)


def test_vision_utils():
    assert _prepare_eager_transform(None, None, None, None) is not None
    assert _map_coords_nearest(None) == 0
    assert _map_coords_bilinear(None) == 0
    assert _np_map_coordinates(None, None, None) == 0
    assert _compute_perspective_matrix(None, None, None) == 0
    assert _generate_perspective_coords(None, None, None) == 0
    assert _generate_perspective_grid(None, 0, 0) == 0
    assert _apply_perspective_channel(None) == 0
    assert _apply_perspective_batch(None, None, None, None) == 0

    c1 = RandomCropConfig(crop_h=1, crop_w=2, b=3, c=4, H=5, W=6, rng=None)
    assert c1.crop_h == 1
    c2 = GeometricGridConfig(H=1, W=2, rng=None, factor1=None, factor2=None)
    assert c2.H == 1
    c3 = EagerTransformContext(np_mod=None, rng=None, imgs=None, B=1, H=2, W=3, C=4, name="n")
    assert c3.B == 1
    c4 = TransformInterpolationConfig(new_y=None, new_x=None, order=1, fill_value=0.0)
    assert c4.order == 1
    c5 = ResizeContext(H=1, W=2, new_H=3, new_W=4, align_corners=False)
    assert c5.H == 1
    c6 = MapCoordsContext(np_mod=None, image=None, y=None, x=None, valid=None)
    assert c6.image is None
    c7 = PerspectiveContext(coords=None, h=None, b=1)
    assert c7.b == 1
    c8 = PerspectiveChannelContext(np_mod=None, imgs=None, out=None, ctx=None, config=None)
    assert c8.imgs is None
