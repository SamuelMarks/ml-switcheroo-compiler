# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy array repr, convolution helpers, correlation, and finite differences."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mm_raw
from tests.testing_utils import coerce_operand_list, require_callable

if typing.TYPE_CHECKING:
    pass

mm = _mm_raw


def test__np_append() -> None:
    """Test function test__np_append."""
    with pytest.raises(TypeError):
        mm._np_append(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_append(np)


def test__np_apply_along_axis() -> None:
    """Test function test__np_apply_along_axis."""
    with pytest.raises(TypeError):
        mm._np_apply_along_axis(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_apply_along_axis(np)


def test__np_apply_over_axes() -> None:
    """Test function test__np_apply_over_axes."""
    with pytest.raises(TypeError):
        mm._np_apply_over_axes(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_apply_over_axes(np)


def test__np_arange_() -> None:
    """Test function test__np_arange_."""
    with pytest.raises(ValueError):
        mm._np_arange_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_arange_(np)


def test__np_argpartition() -> None:
    """Test function test__np_argpartition."""
    with pytest.raises(TypeError):
        mm._np_argpartition(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_argpartition(np)


def test__np_argwhere() -> None:
    """Test function test__np_argwhere."""
    mm._np_argwhere(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_argwhere(np)


def test__np_array_equiv_() -> None:
    """Test function test__np_array_equiv_."""
    with pytest.raises(TypeError):
        mm._np_array_equiv_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_array_equiv_(np)


def test__np_array_repr_() -> None:
    """Test function test__np_array_repr_."""
    mm._np_array_repr_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_array_repr_(np)


def test__np_array_str_() -> None:
    """Test function test__np_array_str_."""
    mm._np_array_str_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_array_str_(np)


def test__np_assertop() -> None:
    """Test function test__np_assertop."""
    mm._np_assertop(np, np.ones((2, 2)))
    _out = mm._np_assertop(np)
    assert np.asarray(_out).shape == (1,) and bool(np.allclose(np.asarray(_out, dtype=float), [0.0], atol=1e-3))


def test__np_asstringconfig() -> None:
    """Test function test__np_asstringconfig."""
    try:
        cfg_first = mm._np_asstringconfig(np, np.ones((2, 2)))
        assert cfg_first is not None
    except ValueError as err:
        assert str(err) != ""
    cfg_default = mm._np_asstringconfig(np)
    assert cfg_default is not None


def test__np_atleast_1d() -> None:
    """Test function test__np_atleast_1d."""
    mm._np_atleast_1d(np, np.ones((2, 2)))
    _out = mm._np_atleast_1d(np)
    assert len(coerce_operand_list(_out)) == 0


def test__np_atleast_2d() -> None:
    """Test function test__np_atleast_2d."""
    mm._np_atleast_2d(np, np.ones((2, 2)))
    _out = mm._np_atleast_2d(np)
    assert len(coerce_operand_list(_out)) == 0


def test__np_atleast_3d() -> None:
    """Test function test__np_atleast_3d."""
    mm._np_atleast_3d(np, np.ones((2, 2)))
    _out = mm._np_atleast_3d(np)
    assert len(coerce_operand_list(_out)) == 0


def test__np_average() -> None:
    """Test function test__np_average."""
    mm._np_average(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_average(np)


def test__np_bessel_i0e() -> None:
    """Test function test__np_bessel_i0e."""
    mm._np_bessel_i0e(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_bessel_i0e(np)


def test__np_bessel_i1e() -> None:
    """Test function test__np_bessel_i1e."""
    mm._np_bessel_i1e(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_bessel_i1e(np)


def test__np_betainc() -> None:
    """Test function test__np_betainc."""
    with pytest.raises(TypeError):
        mm._np_betainc(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_betainc(np)


def test__np_bitcast_convert_type() -> None:
    """Test function test__np_bitcast_convert_type."""
    with pytest.raises(TypeError):
        require_callable(mm._np_bitcast_convert_type)(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_bitcast_convert_type)(np)


def test__np_bitwise_count() -> None:
    """Test function test__np_bitwise_count."""
    with pytest.raises(TypeError):
        mm._np_bitwise_count(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_bitwise_count(np)


def test__np_blackman_() -> None:
    """Test function test__np_blackman_."""
    with pytest.raises(ValueError):
        mm._np_blackman_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_blackman_(np)


def test__np_block() -> None:
    """Test function test__np_block."""
    mm._np_block(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_block(np)


def test__np_blurconfig() -> None:
    """Test function test__np_blurconfig."""
    with pytest.raises(TypeError):
        mm._np_blurconfig(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_blurconfig(np)


def test__np_broadcast_arrays_() -> None:
    """Test function test__np_broadcast_arrays_."""
    mm._np_broadcast_arrays_(np, np.ones((2, 2)))
    _out = mm._np_broadcast_arrays_(np)
    assert len(coerce_operand_list(_out)) == 0


def test__np_callable() -> None:
    """Test function test__np_callable."""
    mm._np_callable(np, np.ones((2, 2)))
    _out = mm._np_callable(np)
    assert not bool(_out)


def test__np_can_cast_() -> None:
    """Test function test__np_can_cast_."""
    with pytest.raises(TypeError):
        mm._np_can_cast_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_can_cast_(np)


def test__np_chebyshev_polynomial_t() -> None:
    """Test function test__np_chebyshev_polynomial_t."""
    with pytest.raises(ValueError):
        mm._np_chebyshev_polynomial_t(np, np.ones((2, 2)))
    with pytest.raises(ValueError):
        mm._np_chebyshev_polynomial_t(np)


def test__np_chebyshev_polynomial_u() -> None:
    """Test function test__np_chebyshev_polynomial_u."""
    with pytest.raises(ValueError):
        mm._np_chebyshev_polynomial_u(np, np.ones((2, 2)))
    with pytest.raises(ValueError):
        mm._np_chebyshev_polynomial_u(np)


def test__np_choose() -> None:
    """Test function test__np_choose."""
    with pytest.raises(TypeError):
        mm._np_choose(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_choose(np)


def test__np_clip() -> None:
    """Test function test__np_clip."""
    assert mm._np_clip(np, np.ones((2, 2)), 0.0, 1.0) is not None
    assert mm._np_clip(np) is None


def test__np_clz() -> None:
    """Test function test__np_clz."""
    with pytest.raises(TypeError):
        mm._np_clz(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_clz)(np)


def test__np_column_stack() -> None:
    """Test function test__np_column_stack."""
    mm._np_column_stack(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_column_stack(np)


def test__np_compress() -> None:
    """Test function test__np_compress."""
    with pytest.raises(TypeError):
        mm._np_compress(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_compress(np)


def test__np_confusion_matrix() -> None:
    """Test function test__np_confusion_matrix."""
    mm._np_confusion_matrix(np, np.ones((2, 2)))
    _out = mm._np_confusion_matrix(np)
    assert _out is None


def test__np_convgeneraldilatedlocal() -> None:
    """Test function test__np_convgeneraldilatedlocal."""
    with pytest.raises(IndexError):
        mm._np_convgeneraldilatedlocal(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_convgeneraldilatedlocal(np)


def test__np_convgeneraldilatedpatches() -> None:
    """Test function test__np_convgeneraldilatedpatches."""
    with pytest.raises(IndexError):
        mm._np_convgeneraldilatedpatches(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_convgeneraldilatedpatches(np)


def test__np_convolve() -> None:
    """Test function test__np_convolve."""
    with pytest.raises(TypeError):
        mm._np_convolve(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_convolve(np)


def test__np_convwithgeneralpadding() -> None:
    """Test function test__np_convwithgeneralpadding."""
    with pytest.raises(IndexError):
        mm._np_convwithgeneralpadding(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_convwithgeneralpadding(np)


def test__np_corrcoef() -> None:
    """Test function test__np_corrcoef."""
    mm._np_corrcoef(np, np.array([[1.0, 2.0], [3.0, 5.0]]))
    with pytest.raises(TypeError):
        mm._np_corrcoef(np)


def test__np_correlate() -> None:
    """Test function test__np_correlate."""
    with pytest.raises(TypeError):
        mm._np_correlate(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_correlate(np)


def test__np_cov() -> None:
    """Test function test__np_cov."""
    mm._np_cov(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_cov(np)


def test__np_customlinearsolve() -> None:
    """Test function test__np_customlinearsolve."""
    with pytest.raises(IndexError):
        mm._np_customlinearsolve(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_customlinearsolve(np)


def test__np_customroot() -> None:
    """Test function test__np_customroot."""
    with pytest.raises(IndexError):
        mm._np_customroot(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_customroot(np)


def test__np_debuginfs() -> None:
    """Test function test__np_debuginfs."""
    mm._np_debuginfs(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_debuginfs(np)


def test__np_debugnans() -> None:
    """Test function test__np_debugnans."""
    mm._np_debugnans(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_debugnans(np)


def test__np_decode_csv() -> None:
    """Test function test__np_decode_csv."""
    mm._np_decode_csv(np, np.ones((2, 2)))
    _out = mm._np_decode_csv(np)
    assert len(coerce_operand_list(_out)) == 1


def test__np_decode_image() -> None:
    """Test function test__np_decode_image."""
    with pytest.raises(RuntimeError):
        mm._np_decode_image(np, np.ones((2, 2)))
    mm._np_decode_image(np)


def test__np_delete_() -> None:
    """Test function test__np_delete_."""
    with pytest.raises(TypeError):
        mm._np_delete_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_delete_(np)


def test__np_descriptive() -> None:
    """Test function test__np_descriptive."""
    mm._np_descriptive(np, np.ones((2, 2)))
    _out = mm._np_descriptive(np)
    assert np.asarray(_out).shape == (3,) and bool(
        np.allclose(np.asarray(_out, dtype=float), [0.0, 0.0, 0.0], atol=1e-3)
    )


def test__np_diag_indices_() -> None:
    """Test function test__np_diag_indices_."""
    with pytest.raises(ValueError):
        mm._np_diag_indices_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_diag_indices_(np)


def test__np_diag_indices_from_() -> None:
    """Test function test__np_diag_indices_from_."""
    mm._np_diag_indices_from_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_diag_indices_from_(np)


def test__np_diagflat_() -> None:
    """Test function test__np_diagflat_."""
    mm._np_diagflat_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_diagflat_(np)


def test__np_diagonal_() -> None:
    """Test function test__np_diagonal_."""
    mm._np_diagonal_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_diagonal_(np)


def test__np_diff_() -> None:
    """Test function test__np_diff_."""
    mm._np_diff_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_diff_(np)


def test__np_digitize_() -> None:
    """Test function test__np_digitize_."""
    with pytest.raises(TypeError):
        mm._np_digitize_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_digitize_(np)


def test__np_distributions() -> None:
    """Test function test__np_distributions."""
    mm._np_distributions(np, np.ones((2, 2)))
    _out = mm._np_distributions(np)
    assert _out is None


def test__np_dotgeneral() -> None:
    """Test function test__np_dotgeneral."""
    with pytest.raises(IndexError):
        mm._np_dotgeneral(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_dotgeneral(np)


def test__np_ediff1d_() -> None:
    """Test function test__np_ediff1d_."""
    mm._np_ediff1d_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_ediff1d_(np)


def test__np_einsum_path_() -> None:
    """Test function test__np_einsum_path_."""
    with pytest.raises(TypeError):
        mm._np_einsum_path_(np, np.ones((2, 2)))
    with pytest.raises(ValueError):
        mm._np_einsum_path_(np)


def test__np_elasticconfig() -> None:
    """Test function test__np_elasticconfig."""
    try:
        cfg_first = mm._np_elasticconfig(np, np.ones((2, 2)))
        assert cfg_first is not None
    except ValueError as err:
        assert str(err) != ""
    cfg_default = mm._np_elasticconfig(np)
    assert cfg_default is not None


def test__np_empty_() -> None:
    """Test function test__np_empty_."""
    with pytest.raises(TypeError):
        mm._np_empty_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_empty_(np)


def test__np_empty_like_() -> None:
    """Test function test__np_empty_like_."""
    mm._np_empty_like_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_empty_like_(np)


def test__np_expand_dims_() -> None:
    """Test function test__np_expand_dims_."""
    with pytest.raises(TypeError):
        mm._np_expand_dims_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_expand_dims_(np)


def test__np_extract_() -> None:
    """Test function test__np_extract_."""
    with pytest.raises(TypeError):
        mm._np_extract_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_extract_(np)


def test__np_extractpatchesoptions() -> None:
    """Test function test__np_extractpatchesoptions."""
    try:
        cfg_first = mm._np_extractpatchesoptions(np, np.ones((2, 2)))
        assert cfg_first is not None
    except ValueError as err:
        assert str(err) != ""
    cfg_default = mm._np_extractpatchesoptions(np)
    assert cfg_default is not None


def test__np_fabs_() -> None:
    """Test function test__np_fabs_."""
    mm._np_fabs_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_fabs_(np)


def test__np_fftn() -> None:
    """Test function test__np_fftn."""
    mm._np_fftn(np, np.ones((2, 2)))
    _out = mm._np_fftn(np)
    assert _out is None


def test__np_fftnd() -> None:
    """Test function test__np_fftnd."""
    mm._np_fftnd(np, np.ones((2, 2)))
    _out = mm._np_fftnd(np)
    assert _out is None


def test__np_fftshift() -> None:
    """Test function test__np_fftshift."""
    mm._np_fftshift(np, np.ones((2, 2)))
    _out = mm._np_fftshift(np)
    assert _out is None


def test__np_flatnonzero_() -> None:
    """Test function test__np_flatnonzero_."""
    mm._np_flatnonzero_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_flatnonzero_(np)


def test__np_flip_op_() -> None:
    """Test function test__np_flip_op_."""
    mm._np_flip_op_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_flip_op_(np)


def test__np_flip_reverse_() -> None:
    """Test function test__np_flip_reverse_."""
    mm._np_flip_reverse_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_flip_reverse_(np)


def test__np_fliplr_() -> None:
    """Test function test__np_fliplr_."""
    mm._np_fliplr_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_fliplr_(np)


def test__np_flipud_() -> None:
    """Test function test__np_flipud_."""
    mm._np_flipud_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_flipud_(np)


def test__np_frombuffer() -> None:
    """Test function test__np_frombuffer."""
    mm._np_frombuffer(np, np.ones((2, 2)))
    _out = mm._np_frombuffer(np)
    assert _out is None


def test__np_fromdlpack() -> None:
    """Test function test__np_fromdlpack."""
    mm._np_fromdlpack(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_fromdlpack(np)


def test__np_fromfunction_() -> None:
    """Test function test__np_fromfunction_."""
    with pytest.raises(TypeError):
        mm._np_fromfunction_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_fromfunction_(np)


def test__np_fromiter_() -> None:
    """Test function test__np_fromiter_."""
    with pytest.raises(TypeError):
        mm._np_fromiter_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_fromiter_(np)


def test__np_frompyfunc_() -> None:
    """Test function test__np_frompyfunc_."""
    with pytest.raises(TypeError):
        mm._np_frompyfunc_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_frompyfunc_(np)


def test__np_fromstring_() -> None:
    """Test function test__np_fromstring_."""
    mm._np_fromstring_(np, "1 2", sep=" ")
    with pytest.raises(TypeError):
        mm._np_fromstring_(np)


def test__np_full_() -> None:
    """Test function test__np_full_."""
    with pytest.raises(TypeError):
        mm._np_full_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_full_(np)


def test__np_full_like_() -> None:
    """Test function test__np_full_like_."""
    with pytest.raises(TypeError):
        mm._np_full_like_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_full_like_(np)


def test__np_geomspace_() -> None:
    """Test function test__np_geomspace_."""
    with pytest.raises(TypeError):
        mm._np_geomspace_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_geomspace_(np)


def test__np_get_printoptions_() -> None:
    """Test function test__np_get_printoptions_."""
    with pytest.raises(TypeError):
        mm._np_get_printoptions_(np, np.ones((2, 2)))
    mm._np_get_printoptions_(np)


def test__np_hamming_() -> None:
    """Test function test__np_hamming_."""
    with pytest.raises(ValueError):
        mm._np_hamming_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_hamming_(np)
