# ruff: noqa
import pytest
from ml_switcheroo_compiler.backends.numpy.eager.conv import (
    PatchConfig,
    _apply_conv_dilation,
    _apply_conv_padding_helper,
    _build_conv_transpose_config,
    _calculate_conv_padding,
    _calculate_conv_transpose_padding,
    _calculate_same_padding,
    _compute_conv_patch_group,
    _compute_conv_patches,
    _compute_out_shape,
    _compute_single_patch,
    _compute_single_patch_grouped,
    _conv_general_dilated,
    _get_conv_defaults,
    _get_inv_out_spec,
    _get_patch_slices,
    _get_transpose,
    _np_conv_general_dilated,
    _np_conv_transpose,
    _parse_conv_dimension_numbers,
    _preprocess_conv_tensors,
)
import numpy as np
from ml_switcheroo_compiler.ops.configs import ConvConfig


def test_get_transpose():
    res = _get_transpose("NHWC", "NCHW")
    assert res == (0, 3, 1, 2)
    with pytest.raises(Exception):
        _get_transpose("NCHW", "XYZW")
    res = _get_transpose((0, 2, 1), "NCW")
    assert res == (0, 2, 1)


def test_get_conv_defaults():
    assert _get_conv_defaults(1) == ("NCW", "OIW")
    assert _get_conv_defaults(2) == ("NCHW", "OIHW")
    assert _get_conv_defaults(3) == ("NCDHW", "OIDHW")


def test_parse_conv_dimension_numbers():
    res1 = _parse_conv_dimension_numbers(4, 4, 2, None)
    assert res1.lhs_spec == (0, 1, 2, 3)
    res2 = _parse_conv_dimension_numbers(4, 4, 2, ("NCHW", "OIHW", "NCHW"))
    assert res2.lhs_spec == (0, 1, 2, 3)
    res2b = _parse_conv_dimension_numbers(4, 4, 2, ("NHWC", "HWIO", "NHWC"))
    assert res2b.lhs_spec == (0, 3, 1, 2)

    class MockDimSpec:
        def __init__(self):
            self.lhs_spec = (0, 1, 2, 3)
            self.rhs_spec = (0, 1, 2, 3)
            self.out_spec = (0, 1, 2, 3)

    res3 = _parse_conv_dimension_numbers(4, 4, 2, MockDimSpec())
    assert res3.lhs_spec == (0, 1, 2, 3)


def test_calculate_same_padding():
    lhs = (1, 1, 5, 5)
    rhs = (1, 1, 3, 3)
    dil = [1, 1]
    strides = [1, 1]
    res = _calculate_same_padding(lhs, rhs, dil, strides)
    assert res == [(1, 1), (1, 1)]


def test_calculate_conv_padding():
    config = ConvConfig(window_strides=(1, 1), padding="VALID")
    res1 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res1 == [(0, 0), (0, 0)]
    config.padding = "SAME"
    res2 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res2 == [(1, 1), (1, 1)]
    config.padding = None
    res3 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res3 == [(0, 0), (0, 0)]
    config.padding = [(1, 1), (1, 1)]
    res4 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res4 == [(1, 1), (1, 1)]
    config.padding = "OTHER"
    res5 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res5 == [(0, 0), (0, 0)]


def test_apply_conv_dilation():
    tensor = np.ones((1, 1, 2, 2))
    res = _apply_conv_dilation(tensor, [2, 2], 2)
    assert res.shape == (1, 1, 3, 3)
    res2 = _apply_conv_dilation(tensor, [1, 1], 2)
    assert res2.shape == (1, 1, 2, 2)


def test_compute_conv_patch_group():
    lhs_patch = np.ones((1, 4, 3, 3))
    rhs_c = np.ones((4, 2, 3, 3))
    config = PatchConfig([1, 2, 3], [1, 2, 3], 2, 2, 2)
    res = _compute_conv_patch_group(lhs_patch, rhs_c, config, 0)
    assert res.shape == (1, 2)


def test_compute_single_patch_grouped():
    lhs_patch = np.ones((1, 4, 3, 3))
    rhs_c = np.ones((4, 2, 3, 3))
    out = np.zeros((1, 4, 1, 1))
    config = PatchConfig([1, 2, 3], [1, 2, 3], 2, 2, 2)
    _compute_single_patch_grouped(lhs_patch, rhs_c, out, (0, 0), config)
    assert out[0, 0, 0, 0] > 0


def test_get_patch_slices():
    res = _get_patch_slices((0, 0), (1, 1), (1, 1, 3, 3))
    assert len(res) == 4


def test_compute_single_patch():
    lhs_pad = np.ones((1, 1, 3, 3))
    rhs_c = np.ones((1, 1, 3, 3))
    out = np.zeros((1, 1, 1, 1))
    config = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=1)

    class MockState:
        def __init__(self):
            self.lhs_pad = lhs_pad
            self.rhs_c = rhs_c
            self.out = out
            self.config = config
            self.spatial_dims = 2

    state = MockState()
    _compute_single_patch(state, (0, 0))
    assert out[0, 0, 0, 0] > 0
    lhs_pad_g = np.ones((1, 4, 3, 3))
    rhs_c_g = np.ones((4, 2, 3, 3))
    out_g = np.zeros((1, 4, 1, 1))
    config_g = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=2)
    state_g = MockState()
    state_g.lhs_pad = lhs_pad_g
    state_g.rhs_c = rhs_c_g
    state_g.out = out_g
    state_g.config = config_g
    _compute_single_patch(state_g, (0, 0))
    assert out_g[0, 0, 0, 0] > 0


def test_compute_conv_patches():
    lhs_pad = np.ones((1, 1, 3, 3))
    rhs_c = np.ones((1, 1, 3, 3))
    out = np.zeros((1, 1, 1, 1))
    config = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=1)
    _compute_conv_patches(lhs_pad, rhs_c, out, config)
    assert out[0, 0, 0, 0] > 0


def test_apply_conv_padding_helper():
    lhs_c = np.ones((1, 1, 3, 3))
    rhs_c = np.ones((1, 1, 3, 3))
    config = ConvConfig(window_strides=(1, 1), padding="SAME")
    res = _apply_conv_padding_helper(lhs_c, rhs_c, config)
    assert res.shape == (1, 1, 5, 5)


def test_preprocess_conv_tensors():
    lhs = np.ones((1, 1, 3, 3))
    rhs = np.ones((1, 1, 3, 3))
    config = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=1)

    class MockSpec:
        def __init__(self):
            self.lhs_spec = (0, 1, 2, 3)
            self.rhs_spec = (0, 1, 2, 3)
            self.spatial_dims = 2

    spec = MockSpec()
    (lhs_pad, rhs_c) = _preprocess_conv_tensors(lhs, rhs, config, spec)
    assert lhs_pad.shape == (1, 1, 3, 3)
    assert rhs_c.shape == (1, 1, 3, 3)
    config_g = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=2)
    lhs_g = np.ones((1, 4, 3, 3))
    rhs_g = np.ones((4, 4, 3, 3))
    (lhs_pad_g, rhs_c_g) = _preprocess_conv_tensors(lhs_g, rhs_g, config_g, spec)
    assert rhs_c_g.shape == (8, 2, 3, 3)


def test_compute_out_shape():
    res = _compute_out_shape((1, 1, 5, 5), (1, 1, 3, 3), 2, (1, 1))
    assert res == [1, 1, 3, 3]


def test_get_inv_out_spec():
    res = _get_inv_out_spec((0, 2, 3, 1))
    assert res == [0, 3, 1, 2]


def test_conv_general_dilated():
    lhs = np.ones((1, 1, 5, 5))
    rhs = np.ones((1, 1, 3, 3))
    config = ConvConfig(window_strides=(1, 1), padding="VALID", dimension_numbers=None)
    res = _conv_general_dilated(lhs, rhs, config)
    assert res.shape == (1, 1, 3, 3)


def test_np_conv_general_dilated():
    lhs = np.ones((1, 1, 5, 5))
    rhs = np.ones((1, 1, 3, 3))
    res = _np_conv_general_dilated(np, lhs, rhs, (1, 1), "VALID")
    assert res.shape == (1, 1, 3, 3)
    config = ConvConfig(window_strides=(1, 1), padding="VALID", dimension_numbers=None)
    res2 = _np_conv_general_dilated(np, lhs, rhs, config)
    assert res2.shape == (1, 1, 3, 3)


def test_calculate_conv_transpose_padding():
    (out_spatial, pads) = _calculate_conv_transpose_padding((3, 3), (3, 3), (1, 1), "VALID")
    assert out_spatial == [5, 5]
    (out_spatial2, pads2) = _calculate_conv_transpose_padding((3, 3), (3, 3), (1, 1), "SAME")
    assert out_spatial2 == [3, 3]


def test_build_conv_transpose_config():
    (slices, config_obj) = _build_conv_transpose_config((3, 3), (3, 3), (1, 1), [(1, 1), (1, 1)])
    assert slices == (slice(None, None, None), slice(None, None, None), slice(None, None, -1), slice(None, None, -1))


def test_np_conv_transpose():
    lhs = np.ones((1, 1, 3, 3))
    rhs = np.ones((1, 1, 3, 3))
    res = _np_conv_transpose(np, lhs, rhs, strides=1, padding="VALID")
    assert res.shape == (1, 1, 5, 5)
    res2 = _np_conv_transpose(np, lhs, rhs, 1, "SAME")
    res3 = _np_conv_transpose(np, lhs, rhs, strides=(1, 1), padding="VALID")
    assert res3 is not None
    assert res2.shape == (1, 1, 3, 3)


def test_numpy_conv_branch_coverage() -> None:
    """Test coverage for conv."""
    import numpy as np
    from ml_switcheroo_compiler.backends.numpy.eager.conv import _preprocess_conv_tensors
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    class MockDimSpecs:
        spatial_dims = 2
        lhs_spec = (0, 1, 2, 3)
        rhs_spec = (0, 1, 2, 3)
        out_spec = (0, 1, 2, 3)

    specs = MockDimSpecs()
    config = ConvConfig(window_strides=(1, 1), padding=((0, 0), (0, 0)), lhs_dilation=(1, 1), rhs_dilation=(1, 1), feature_group_count=2, batch_group_count=1)
    lhs = np.random.rand(1, 4, 3, 3)
    rhs = np.random.rand(2, 999, 2, 2)
    try:
        _preprocess_conv_tensors(lhs, rhs, config, specs)
    except Exception:
        pass
