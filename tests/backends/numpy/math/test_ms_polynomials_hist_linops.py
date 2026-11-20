# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy orthogonal polynomials, histograms, and LinearOperator classes."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mm_raw
from tests.testing_utils import require_callable

if typing.TYPE_CHECKING:
    pass

mm = _mm_raw


def test__np_hanning_() -> None:
    """Test function test__np_hanning_."""
    with pytest.raises(ValueError):
        mm._np_hanning_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_hanning_(np)


def test__np_hermite_polynomial_h() -> None:
    """Test function test__np_hermite_polynomial_h."""
    with pytest.raises(ValueError):
        mm._np_hermite_polynomial_h(np, np.ones((2, 2)))
    with pytest.raises(ValueError):
        mm._np_hermite_polynomial_h(np)


def test__np_hermite_polynomial_he() -> None:
    """Test function test__np_hermite_polynomial_he."""
    with pytest.raises(ValueError):
        mm._np_hermite_polynomial_he(np, np.ones((2, 2)))
    with pytest.raises(ValueError):
        mm._np_hermite_polynomial_he(np)


def test__np_hfft() -> None:
    """Test function test__np_hfft."""
    mm._np_hfft(np, np.ones((2, 2)))
    _out = mm._np_hfft(np)
    assert _out is None


def test__np_histogram2d_() -> None:
    """Test function test__np_histogram2d_."""
    with pytest.raises(TypeError):
        mm._np_histogram2d_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_histogram2d_(np)


def test__np_histogram_() -> None:
    """Test function test__np_histogram_."""
    mm._np_histogram_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_histogram_(np)


def test__np_histogram_bin_edges_() -> None:
    """Test function test__np_histogram_bin_edges_."""
    mm._np_histogram_bin_edges_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_histogram_bin_edges_(np)


def test__np_histogramdd_() -> None:
    """Test function test__np_histogramdd_."""
    mm._np_histogramdd_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_histogramdd_(np)


def test__np_ifft() -> None:
    """Test function test__np_ifft."""
    mm._np_ifft(np, np.ones((2, 2)))
    _out = mm._np_ifft(np)
    assert _out is None


def test__np_ifft2() -> None:
    """Test function test__np_ifft2."""
    mm._np_ifft2(np, np.ones((2, 2)))
    _out = mm._np_ifft2(np)
    assert _out is None


def test__np_ifftn() -> None:
    """Test function test__np_ifftn."""
    mm._np_ifftn(np, np.ones((2, 2)))
    _out = mm._np_ifftn(np)
    assert _out is None


def test__np_ifftnd() -> None:
    """Test function test__np_ifftnd."""
    mm._np_ifftnd(np, np.ones((2, 2)))
    _out = mm._np_ifftnd(np)
    assert _out is None


def test__np_ifftshift() -> None:
    """Test function test__np_ifftshift."""
    mm._np_ifftshift(np, np.ones((2, 2)))
    _out = mm._np_ifftshift(np)
    assert _out is None


def test__np_indices_() -> None:
    """Test function test__np_indices_."""
    with pytest.raises(TypeError):
        mm._np_indices_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_indices_(np)


def test__np_insert_() -> None:
    """Test function test__np_insert_."""
    with pytest.raises(TypeError):
        mm._np_insert_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_insert_(np)


def test__np_interp_() -> None:
    """Test function test__np_interp_."""
    with pytest.raises(TypeError):
        mm._np_interp_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_interp_(np)


def test__np_intersect1d_() -> None:
    """Test function test__np_intersect1d_."""
    with pytest.raises(TypeError):
        mm._np_intersect1d_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_intersect1d_(np)


def test__np_irfft2() -> None:
    """Test function test__np_irfft2."""
    mm._np_irfft2(np, np.ones((2, 2)))
    _out = mm._np_irfft2(np)
    assert _out is None


def test__np_irfftn() -> None:
    """Test function test__np_irfftn."""
    mm._np_irfftn(np, np.ones((2, 2)))
    _out = mm._np_irfftn(np)
    assert _out is None


def test__np_irfftnd() -> None:
    """Test function test__np_irfftnd."""
    mm._np_irfftnd(np, np.ones((2, 2)))
    _out = mm._np_irfftnd(np)
    assert _out is None


def test__np_iscomplex_() -> None:
    """Test function test__np_iscomplex_."""
    mm._np_iscomplex_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_iscomplex_(np)


def test__np_iscomplexobj_() -> None:
    """Test function test__np_iscomplexobj_."""
    mm._np_iscomplexobj_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_iscomplexobj_(np)


def test__np_isin_() -> None:
    """Test function test__np_isin_."""
    with pytest.raises(TypeError):
        mm._np_isin_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_isin_(np)


def test__np_isreal_() -> None:
    """Test function test__np_isreal_."""
    mm._np_isreal_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_isreal_(np)


def test__np_isrealobj_() -> None:
    """Test function test__np_isrealobj_."""
    mm._np_isrealobj_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_isrealobj_(np)


def test__np_isscalar_() -> None:
    """Test function test__np_isscalar_."""
    mm._np_isscalar_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_isscalar_(np)


def test__np_issubdtype_issubdtype_() -> None:
    """Test function test__np_issubdtype_issubdtype_."""
    with pytest.raises(TypeError):
        mm._np_issubdtype_issubdtype_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_issubdtype_issubdtype_(np)


def test__np_issubdtype_op_() -> None:
    """Test function test__np_issubdtype_op_."""
    with pytest.raises(TypeError):
        mm._np_issubdtype_op_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_issubdtype_op_(np)


def test__np_iterable_() -> None:
    """Test function test__np_iterable_."""
    mm._np_iterable_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_iterable_(np)


def test__np_ix__() -> None:
    """Test function test__np_ix__."""
    with pytest.raises(ValueError):
        mm._np_ix__(np, np.ones((2, 2)))
    mm._np_ix__(np)


def test__np_kaiser_() -> None:
    """Test function test__np_kaiser_."""
    with pytest.raises(TypeError):
        mm._np_kaiser_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_kaiser_(np)


def test__np_key() -> None:
    """Test function test__np_key."""
    with pytest.raises(ValueError):
        mm._np_key(np, np.ones((2, 2)))
    mm._np_key(np)


def test__np_kron_() -> None:
    """Test function test__np_kron_."""
    with pytest.raises(TypeError):
        mm._np_kron_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_kron_(np)


def test__np_laguerre_polynomial_l() -> None:
    """Test function test__np_laguerre_polynomial_l."""
    with pytest.raises(ValueError):
        mm._np_laguerre_polynomial_l(np, np.ones((2, 2)))
    with pytest.raises(ValueError):
        mm._np_laguerre_polynomial_l(np)


def test__np_legendre_polynomial_p() -> None:
    """Test function test__np_legendre_polynomial_p."""
    with pytest.raises(ValueError):
        mm._np_legendre_polynomial_p(np, np.ones((2, 2)))
    with pytest.raises(ValueError):
        mm._np_legendre_polynomial_p(np)


def test__np_lexsort_() -> None:
    """Test function test__np_lexsort_."""
    mm._np_lexsort_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_lexsort_(np)


def test__np_linalg_cholesky_() -> None:
    """Test function test__np_linalg_cholesky_."""
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=".*"
    ):
        mm._np_linalg_cholesky_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_linalg_cholesky_(np)


def test__np_linalg_det_() -> None:
    """Test function test__np_linalg_det_."""
    mm._np_linalg_det_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_linalg_det_(np)


def test__np_linalg_inv_() -> None:
    """Test function test__np_linalg_inv_."""
    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=".*"
    ):
        mm._np_linalg_inv_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_linalg_inv_(np)


def test__np_linalg_matrix_power_() -> None:
    """Test function test__np_linalg_matrix_power_."""
    with pytest.raises(TypeError):
        mm._np_linalg_matrix_power_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_linalg_matrix_power_(np)


def test__np_linalg_pinv_() -> None:
    """Test function test__np_linalg_pinv_."""
    mm._np_linalg_pinv_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_linalg_pinv_(np)


def test__np_linalg_svd_() -> None:
    """Test function test__np_linalg_svd_."""
    mm._np_linalg_svd_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_linalg_svd_(np)


def test__np_linearoperator() -> None:
    """Test function test__np_linearoperator."""
    with pytest.raises(TypeError):
        mm._np_linearoperator(np, np.ones((2, 2)))
    mm._np_linearoperator(np)


def test__np_linearoperatoradjoint() -> None:
    """Test function test__np_linearoperatoradjoint."""
    with pytest.raises(TypeError):
        mm._np_linearoperatoradjoint(np, np.ones((2, 2)))
    mm._np_linearoperatoradjoint(np)


def test__np_linearoperatorblockdiag() -> None:
    """Test function test__np_linearoperatorblockdiag."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorblockdiag(np, np.ones((2, 2)))
    mm._np_linearoperatorblockdiag(np)


def test__np_linearoperatorblocklowertriangular() -> None:
    """Test function test__np_linearoperatorblocklowertriangular."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorblocklowertriangular(np, np.ones((2, 2)))
    mm._np_linearoperatorblocklowertriangular(np)


def test__np_linearoperatorcirculant() -> None:
    """Test function test__np_linearoperatorcirculant."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorcirculant(np, np.ones((2, 2)))
    mm._np_linearoperatorcirculant(np)


def test__np_linearoperatorcirculant2d() -> None:
    """Test function test__np_linearoperatorcirculant2d."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorcirculant2d(np, np.ones((2, 2)))
    mm._np_linearoperatorcirculant2d(np)


def test__np_linearoperatorcirculant3d() -> None:
    """Test function test__np_linearoperatorcirculant3d."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorcirculant3d(np, np.ones((2, 2)))
    mm._np_linearoperatorcirculant3d(np)


def test__np_linearoperatorcomposition() -> None:
    """Test function test__np_linearoperatorcomposition."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorcomposition(np, np.ones((2, 2)))
    mm._np_linearoperatorcomposition(np)


def test__np_linearoperatordiag() -> None:
    """Test function test__np_linearoperatordiag."""
    with pytest.raises(TypeError):
        mm._np_linearoperatordiag(np, np.ones((2, 2)))
    mm._np_linearoperatordiag(np)


def test__np_linearoperatorfullmatrix() -> None:
    """Test function test__np_linearoperatorfullmatrix."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorfullmatrix(np, np.ones((2, 2)))
    mm._np_linearoperatorfullmatrix(np)


def test__np_linearoperatorhouseholder() -> None:
    """Test function test__np_linearoperatorhouseholder."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorhouseholder(np, np.ones((2, 2)))
    mm._np_linearoperatorhouseholder(np)


def test__np_linearoperatoridentity() -> None:
    """Test function test__np_linearoperatoridentity."""
    with pytest.raises(TypeError):
        mm._np_linearoperatoridentity(np, np.ones((2, 2)))
    mm._np_linearoperatoridentity(np)


def test__np_linearoperatorinversion() -> None:
    """Test function test__np_linearoperatorinversion."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorinversion(np, np.ones((2, 2)))
    mm._np_linearoperatorinversion(np)


def test__np_linearoperatorkronecker() -> None:
    """Test function test__np_linearoperatorkronecker."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorkronecker(np, np.ones((2, 2)))
    mm._np_linearoperatorkronecker(np)


def test__np_linearoperatorlowertriangular() -> None:
    """Test function test__np_linearoperatorlowertriangular."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorlowertriangular(np, np.ones((2, 2)))
    mm._np_linearoperatorlowertriangular(np)


def test__np_linearoperatorlowrankupdate() -> None:
    """Test function test__np_linearoperatorlowrankupdate."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorlowrankupdate(np, np.ones((2, 2)))
    mm._np_linearoperatorlowrankupdate(np)


def test__np_linearoperatorpermutation() -> None:
    """Test function test__np_linearoperatorpermutation."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorpermutation(np, np.ones((2, 2)))
    mm._np_linearoperatorpermutation(np)


def test__np_linearoperatorscaledidentity() -> None:
    """Test function test__np_linearoperatorscaledidentity."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorscaledidentity(np, np.ones((2, 2)))
    mm._np_linearoperatorscaledidentity(np)


def test__np_linearoperatortoeplitz() -> None:
    """Test function test__np_linearoperatortoeplitz."""
    with pytest.raises(TypeError):
        mm._np_linearoperatortoeplitz(np, np.ones((2, 2)))
    mm._np_linearoperatortoeplitz(np)


def test__np_linearoperatortridiag() -> None:
    """Test function test__np_linearoperatortridiag."""
    with pytest.raises(TypeError):
        mm._np_linearoperatortridiag(np, np.ones((2, 2)))
    mm._np_linearoperatortridiag(np)


def test__np_linearoperatorzeros() -> None:
    """Test function test__np_linearoperatorzeros."""
    with pytest.raises(TypeError):
        mm._np_linearoperatorzeros(np, np.ones((2, 2)))
    mm._np_linearoperatorzeros(np)


def test__np_load_() -> None:
    """Test function test__np_load_."""
    with pytest.raises(TypeError):
        mm._np_load_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_load_(np)


def test__np_log1p2() -> None:
    """Test function test__np_log1p2."""
    mm._np_log1p2(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_log1p2(np)


def test__np_log_softmax() -> None:
    """Test function test__np_log_softmax."""
    mm._np_log_softmax(np, np.ones((2, 2)))
    _out = mm._np_log_softmax(np)
    assert _out is None


def test__np_logsumexp() -> None:
    """Test function test__np_logsumexp."""
    mm._np_logsumexp(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_logsumexp)(np)


def test__np_mask_indices_() -> None:
    """Test function test__np_mask_indices_."""
    with pytest.raises(TypeError):
        mm._np_mask_indices_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_mask_indices_(np)


def test__np_median_() -> None:
    """Test function test__np_median_."""
    mm._np_median_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_median_(np)


def test__np_modf_() -> None:
    """Test function test__np_modf_."""
    mm._np_modf_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_modf_(np)


def test__np_modified_bessel_i0() -> None:
    """Test function test__np_modified_bessel_i0."""
    mm._np_modified_bessel_i0(np, np.ones((2, 2)))
    _out = mm._np_modified_bessel_i0(np)
    assert _out is None


def test__np_modified_bessel_i1() -> None:
    """Test function test__np_modified_bessel_i1."""
    mm._np_modified_bessel_i1(np, np.ones((2, 2)))
    _out = mm._np_modified_bessel_i1(np)
    assert _out is None


def test__np_modified_bessel_k0() -> None:
    """Test function test__np_modified_bessel_k0."""
    mm._np_modified_bessel_k0(np, np.ones((2, 2)))
    _out = mm._np_modified_bessel_k0(np)
    assert _out is None


def test__np_modified_bessel_k1() -> None:
    """Test function test__np_modified_bessel_k1."""
    mm._np_modified_bessel_k1(np, np.ones((2, 2)))
    _out = mm._np_modified_bessel_k1(np)
    assert _out is None


def test__np_mvlgamma() -> None:
    """Test function test__np_mvlgamma."""
    with pytest.raises(AttributeError):
        mm._np_mvlgamma(np, np.ones((2, 2)))
    with pytest.raises(AttributeError):
        mm._np_mvlgamma(np)


def test__np_nonzero_() -> None:
    """Test function test__np_nonzero_."""
    mm._np_nonzero_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_nonzero_(np)


def test__np_one_hot() -> None:
    """Test function test__np_one_hot."""
    mm._np_one_hot(np, np.ones((2, 2)))
    _out = mm._np_one_hot(np)
    assert _out is None


def test__np_ones_() -> None:
    """Test function test__np_ones_."""
    with pytest.raises(TypeError):
        mm._np_ones_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_ones_(np)


def test__np_ones_like_() -> None:
    """Test function test__np_ones_like_."""
    mm._np_ones_like_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_ones_like_(np)


def test__np_packbits() -> None:
    """Test function test__np_packbits."""
    with pytest.raises(TypeError):
        mm._np_packbits(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_packbits(np)


def test__np_parse_tensor() -> None:
    """Test function test__np_parse_tensor."""
    with pytest.raises(RuntimeError):
        mm._np_parse_tensor(np, np.ones((2, 2)))
    mm._np_parse_tensor(np)


def test__np_perspectiveconfig() -> None:
    """Test function test__np_perspectiveconfig."""
    try:
        cfg_first = mm._np_perspectiveconfig(np, np.ones((2, 2)))
        assert cfg_first is not None
    except ValueError as err:
        assert str(err) != ""
    cfg_default = mm._np_perspectiveconfig(np)
    assert cfg_default is not None
