# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for test_conv."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import (
    ConvDimSpecs,
    ConvExecutionState,
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
from ml_switcheroo_compiler.ops.configs import ConvConfig
from tests.testing_utils import MockTestStub, adapt_backend_module, extract_test_shape

if typing.TYPE_CHECKING:
    pass


def test_get_transpose() -> None:
    """Test function test_get_transpose."""
    res = _get_transpose("NHWC", "NCHW")
    assert res == (0, 3, 1, 2)
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=r".*"
    ):
        _get_transpose("NCHW", "XYZW")
    res = _get_transpose((0, 2, 1), "NCW")
    assert res == (0, 2, 1)


def test_get_conv_defaults() -> None:
    """Test function test_get_conv_defaults."""
    assert _get_conv_defaults(1) == ("NCW", "OIW")
    assert _get_conv_defaults(2) == ("NCHW", "OIHW")
    assert _get_conv_defaults(3) == ("NCDHW", "OIDHW")


def test_parse_conv_dimension_numbers() -> None:
    """Test function test_parse_conv_dimension_numbers."""
    res1 = _parse_conv_dimension_numbers(4, 4, 2, None)
    assert tuple(res1.lhs_spec) == (0, 1, 2, 3)
    res2 = _parse_conv_dimension_numbers(4, 4, 2, ("NCHW", "OIHW", "NCHW"))
    assert tuple(res2.lhs_spec) == (0, 1, 2, 3)
    res2b = _parse_conv_dimension_numbers(4, 4, 2, ("NHWC", "HWIO", "NHWC"))
    assert tuple(res2b.lhs_spec) == (0, 3, 1, 2)

    class MockDimSpec(MockTestStub):
        """Test class MockDimSpec."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.lhs_spec = (0, 1, 2, 3)
            self.rhs_spec = (0, 1, 2, 3)
            self.out_spec = (0, 1, 2, 3)

    res3 = _parse_conv_dimension_numbers(4, 4, 2, MockDimSpec())
    assert tuple(res3.lhs_spec) == (0, 1, 2, 3)


def test_calculate_same_padding() -> None:
    """Test function test_calculate_same_padding."""
    lhs = (1, 1, 5, 5)
    rhs = (1, 1, 3, 3)
    dil = [1, 1]
    strides = [1, 1]
    res = _calculate_same_padding(lhs, rhs, dil, strides)
    assert res == [(1, 1), (1, 1)]


def test_calculate_conv_padding() -> None:
    """Test function test_calculate_conv_padding."""
    config = ConvConfig(window_strides=(1, 1), padding="VALID")
    res1 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res1 == [(0, 0), (0, 0)]
    config = ConvConfig(window_strides=(1, 1), padding="SAME")
    res2 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res2 == [(1, 1), (1, 1)]
    res3 = _calculate_conv_padding(config, (1, 1), (1, 1))
    assert res3 == []
    res3 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res3 == [(1, 1), (1, 1)]
    config = ConvConfig(window_strides=(1, 1), padding=[(1, 1), (1, 1)])
    res4 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res4 == [(1, 1), (1, 1)]
    config = ConvConfig(window_strides=(1, 1), padding="OTHER")
    res5 = _calculate_conv_padding(config, (1, 1, 5, 5), (1, 1, 3, 3))
    assert res5 == [(0, 0), (0, 0)]


def test_apply_conv_dilation() -> None:
    """Test function test_apply_conv_dilation."""
    tensor = np.ones((1, 1, 2, 2))
    res = _apply_conv_dilation(tensor, [2, 2], 2)
    assert extract_test_shape(res) == (1, 1, 3, 3)
    res2 = _apply_conv_dilation(tensor, [1, 1], 2)
    assert extract_test_shape(res2) == (1, 1, 2, 2)


def test_compute_conv_patch_group() -> None:
    """Test function test_compute_conv_patch_group."""
    lhs_patch = np.ones((1, 4, 3, 3))
    rhs_c = np.ones((4, 2, 3, 3))
    config = PatchConfig(axes_lhs=[1, 2, 3], axes_rhs=[1, 2, 3], group_in_c=2, group_out_c=2, feature_group_count=2)
    res = _compute_conv_patch_group(lhs_patch, rhs_c, config, 0)
    assert extract_test_shape(res) == (1, 2)


def test_compute_single_patch_grouped() -> None:
    """Test function test_compute_single_patch_grouped."""
    lhs_patch = np.ones((1, 4, 3, 3))
    rhs_c = np.ones((4, 2, 3, 3))
    out = np.zeros((1, 4, 1, 1))
    config = PatchConfig(axes_lhs=[1, 2, 3], axes_rhs=[1, 2, 3], group_in_c=2, group_out_c=2, feature_group_count=2)
    _compute_single_patch_grouped(lhs_patch, rhs_c, out, (0, 0), config)
    assert out[0, 0, 0, 0] > 0


def test_get_patch_slices() -> None:
    """Test function test_get_patch_slices."""
    res = _get_patch_slices((0, 0), (1, 1), (1, 1, 3, 3))
    assert len(res) == 4


def test_compute_single_patch() -> None:
    """Test function test_compute_single_patch."""
    lhs_pad = np.ones((1, 1, 3, 3))
    rhs_c = np.ones((1, 1, 3, 3))
    out = np.zeros((1, 1, 1, 1))
    config = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=1)
    np_adp = adapt_backend_module(np)
    state = ConvExecutionState(
        lhs_pad=np_adp.asarray(lhs_pad),
        rhs_c=np_adp.asarray(rhs_c),
        out=np_adp.asarray(out),
        config=config,
        spatial_dims=2,
    )
    _compute_single_patch(state, (0, 0))
    assert out[0, 0, 0, 0] > 0
    lhs_pad_g = np.ones((1, 4, 3, 3))
    rhs_c_g = np.ones((4, 2, 3, 3))
    out_g = np.zeros((1, 4, 1, 1))
    config_g = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=2)
    state_g = ConvExecutionState(
        lhs_pad=np_adp.asarray(lhs_pad_g),
        rhs_c=np_adp.asarray(rhs_c_g),
        out=np_adp.asarray(out_g),
        config=config_g,
        spatial_dims=2,
    )
    _compute_single_patch(state_g, (0, 0))
    assert out_g[0, 0, 0, 0] > 0


def test_compute_conv_patches() -> None:
    """Test function test_compute_conv_patches."""
    lhs_pad = np.ones((1, 1, 3, 3))
    rhs_c = np.ones((1, 1, 3, 3))
    out = np.zeros((1, 1, 1, 1))
    config = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=1)
    _compute_conv_patches(lhs_pad, rhs_c, out, config)
    assert out[0, 0, 0, 0] > 0


def test_apply_conv_padding_helper() -> None:
    """Test function test_apply_conv_padding_helper."""
    lhs_c = np.ones((1, 1, 3, 3))
    rhs_c = np.ones((1, 1, 3, 3))
    config = ConvConfig(window_strides=(1, 1), padding="SAME")
    res = _apply_conv_padding_helper(lhs_c, rhs_c, config)
    assert extract_test_shape(res) == (1, 1, 5, 5)


def test_preprocess_conv_tensors() -> None:
    """Test function test_preprocess_conv_tensors."""
    lhs = np.ones((1, 1, 3, 3))
    rhs = np.ones((1, 1, 3, 3))
    config = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=1)
    spec = ConvDimSpecs(lhs_spec=(0, 1, 2, 3), rhs_spec=(0, 1, 2, 3), out_spec=(0, 1, 2, 3), spatial_dims=2)
    (lhs_pad, rhs_c) = _preprocess_conv_tensors(lhs, rhs, config, spec)
    assert lhs_pad.shape == (1, 1, 3, 3)
    assert rhs_c.shape == (1, 1, 3, 3)
    config_g = ConvConfig(window_strides=(1, 1), padding="VALID", feature_group_count=2)
    lhs_g = np.ones((1, 4, 3, 3))
    rhs_g = np.ones((4, 4, 3, 3))
    (lhs_pad_g, rhs_c_g) = _preprocess_conv_tensors(lhs_g, rhs_g, config_g, spec)
    assert rhs_c_g.shape == (8, 2, 3, 3)


def test_compute_out_shape() -> None:
    """Test function test_compute_out_shape."""
    res = _compute_out_shape((1, 1, 5, 5), (1, 1, 3, 3), 2, (1, 1))
    assert res == [1, 1, 3, 3]


def test_get_inv_out_spec() -> None:
    """Test function test_get_inv_out_spec."""
    res = _get_inv_out_spec((0, 2, 3, 1))
    assert res == [0, 3, 1, 2]


def test_conv_general_dilated() -> None:
    """Test function test_conv_general_dilated."""
    lhs = np.ones((1, 1, 5, 5))
    rhs = np.ones((1, 1, 3, 3))
    config = ConvConfig(window_strides=(1, 1), padding="VALID", dimension_numbers=None)
    res = _conv_general_dilated(lhs, rhs, config)
    assert extract_test_shape(res) == (1, 1, 3, 3)


def test_np_conv_general_dilated() -> None:
    """Test function test_np_conv_general_dilated."""
    lhs = np.ones((1, 1, 5, 5))
    rhs = np.ones((1, 1, 3, 3))
    res = _np_conv_general_dilated(np, lhs, rhs, (1, 1), "VALID")
    assert extract_test_shape(res) == (1, 1, 3, 3)
    config = ConvConfig(window_strides=(1, 1), padding="VALID", dimension_numbers=None)
    res2 = _np_conv_general_dilated(np, lhs, rhs, config)
    assert extract_test_shape(res2) == (1, 1, 3, 3)


def test_calculate_conv_transpose_padding() -> None:
    """Test function test_calculate_conv_transpose_padding."""
    (out_spatial, pads) = _calculate_conv_transpose_padding((3, 3), (3, 3), (1, 1), "VALID")
    assert out_spatial == [5, 5]
    (out_spatial2, pads2) = _calculate_conv_transpose_padding((3, 3), (3, 3), (1, 1), "SAME")
    assert out_spatial2 == [3, 3]


def test_build_conv_transpose_config() -> None:
    """Test function test_build_conv_transpose_config."""
    (slices, config_obj) = _build_conv_transpose_config((3, 3), (3, 3), (1, 1), [(1, 1), (1, 1)])
    assert slices == (slice(None, None, None), slice(None, None, None), slice(None, None, -1), slice(None, None, -1))


def test_np_conv_transpose() -> None:
    """Test function test_np_conv_transpose."""
    lhs = np.ones((1, 1, 3, 3))
    rhs = np.ones((1, 1, 3, 3))
    res = _np_conv_transpose(np, lhs, rhs, strides=1, padding="VALID")
    assert extract_test_shape(res) == (1, 1, 5, 5)
    res2 = _np_conv_transpose(np, lhs, rhs, 1, "SAME")
    res3 = _np_conv_transpose(np, lhs, rhs, strides=(1, 1), padding="VALID")
    assert res3 is not None
    assert extract_test_shape(res2) == (1, 1, 3, 3)


def test_numpy_conv_branch_coverage() -> None:
    """Test coverage for conv."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _np_conv2d, _preprocess_conv_tensors
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    specs = ConvDimSpecs(spatial_dims=2, lhs_spec=(0, 1, 2, 3), rhs_spec=(0, 1, 2, 3), out_spec=(0, 1, 2, 3))
    config = ConvConfig(
        window_strides=(1, 1),
        padding=((0, 0), (0, 0)),
        lhs_dilation=(1, 1),
        rhs_dilation=(1, 1),
        feature_group_count=2,
        batch_group_count=1,
    )
    lhs = np.random.rand(1, 4, 3, 3)
    rhs = np.random.rand(2, 999, 2, 2)
    _preprocess_conv_tensors(lhs, rhs, config, specs)
    rhs_exact = np.random.rand(2, 2, 2, 2)
    (res_pad, res_rhs) = _preprocess_conv_tensors(lhs, rhs_exact, config, specs)
    assert tuple(res_pad.shape) == (1, 4, 3, 3)
    assert tuple(res_rhs.shape) == (2, 2, 2, 2)
    lhs_conv = np.ones((1, 1, 5, 5))
    rhs_conv = np.ones((1, 1, 3, 3))
    out1 = _np_conv2d(np, lhs_conv, rhs_conv, stride=1, padding=(1, 1))
    assert tuple(extract_test_shape(out1)) == (1, 1, 5, 5)
    out2 = _np_conv2d(np, lhs_conv, rhs_conv, strides=[1, 1], padding="VALID")
    assert tuple(extract_test_shape(out2)) == (1, 1, 3, 3)


def test_np_conv2d_groups() -> None:
    """Test _np_conv2d execution with grouped convolutions."""
    from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _np_conv2d

    lhs = np.ones((2, 4, 8, 8), dtype=np.float32)
    rhs = np.ones((4, 2, 3, 3), dtype=np.float32)
    out = _np_conv2d(np, lhs, rhs, stride=1, padding="SAME", groups=2)
    assert extract_test_shape(out) == (2, 4, 8, 8)


def test_compute_conv2d_padding_values() -> None:
    """Test _compute_conv2d_padding_values under different padding spec types."""
    from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _compute_conv2d_padding_values

    assert _compute_conv2d_padding_values((10, 10), (3, 3), (1, 1), (1, 1), (2, 3), (10, 10)) == (2, 2, 3, 3)
    assert _compute_conv2d_padding_values((10, 10), (3, 3), (1, 1), (1, 1), (1, 2, 3, 4), (10, 10)) == (1, 2, 3, 4)
    assert _compute_conv2d_padding_values((10, 10), (3, 3), (1, 1), (1, 1), 2, (10, 10)) == (2, 2, 2, 2)
    assert _compute_conv2d_padding_values((10, 10), (3, 3), (1, 1), (1, 1), "SAME", (10, 10)) == (1, 1, 1, 1)
    assert _compute_conv2d_padding_values((10, 10), (3, 3), (1, 1), (1, 1), "same", (10, 10)) == (1, 1, 1, 1)
    assert _compute_conv2d_padding_values((10, 10), (3, 3), (1, 1), (1, 1), [1, 2, 3], (10, 10)) == (0, 0, 0, 0)
    assert _compute_conv2d_padding_values((10, 10), (3, 3), (1, 1), (1, 1), "VALID", (8, 8)) == (0, 0, 0, 0)
    assert _compute_conv2d_padding_values((10, 10), (3, 3), (1, 1), (1, 1), None, (10, 10)) == (0, 0, 0, 0)


def test_np_conv2d_input_grad() -> None:
    """Test _np_conv2d_input_grad gradient evaluation for grouped and standard convolutions."""
    from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _np_conv2d_input_grad

    d_out = np.ones((2, 4, 8, 8), dtype=np.float32)
    weight = np.ones((4, 2, 3, 3), dtype=np.float32)
    dx = _np_conv2d_input_grad(
        np,
        d_out,
        weight,
        stride=1,
        dilation=1,
        groups=2,
        padding="SAME",
        target_shape=(2, 4, 8, 8),
    )
    assert dx.shape == (2, 4, 8, 8)
    weight_g1 = np.ones((4, 3, 1, 1), dtype=np.float32)
    d_out_g1 = np.ones((2, 4, 6, 6), dtype=np.float32)
    dx_g1 = _np_conv2d_input_grad(np, d_out_g1, weight_g1, strides=(1, 1), dilations=(1, 1), groups=1, padding="VALID")
    assert dx_g1.shape == (2, 3, 6, 6)


def test_np_conv2d_weight_grad() -> None:
    """Test _np_conv2d_weight_grad gradient evaluation for grouped and standard convolutions."""
    from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _np_conv2d_weight_grad

    x = np.ones((2, 4, 8, 8), dtype=np.float32)
    d_out = np.ones((2, 4, 8, 8), dtype=np.float32)
    dw = _np_conv2d_weight_grad(np, x, d_out, stride=1, dilation=1, groups=2, padding="SAME", target_shape=(4, 2, 3, 3))
    assert dw.shape == (4, 2, 3, 3)
    x_valid = np.ones((2, 3, 8, 8), dtype=np.float32)
    d_out_valid = np.ones((2, 6, 6, 6), dtype=np.float32)
    dw_valid = _np_conv2d_weight_grad(
        np, x_valid, d_out_valid, strides=(1, 1), dilations=(1, 1), groups=1, padding="VALID"
    )
    assert dw_valid.shape == (6, 3, 3, 3)


def test_np_conv2d_bias_grad() -> None:
    """Test _np_conv2d_bias_grad gradient evaluation."""
    from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _np_conv2d_bias_grad

    d_out = np.ones((2, 4, 8, 8), dtype=np.float32)
    db = _np_conv2d_bias_grad(np, d_out)
    assert db.shape == (4,)
    assert np.allclose(db, 2 * 8 * 8)
