# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy real FFTs, sparse segment/reshape ops, and tensor array utilities."""

from __future__ import annotations

import os
import typing

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mm_raw
from tests.testing_utils import coerce_int, coerce_operand_list, require_callable

if typing.TYPE_CHECKING:
    pass

mm = _mm_raw


def test__np_piecewise() -> None:
    """Test function test__np_piecewise."""
    with pytest.raises(IndexError):
        mm._np_piecewise(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_piecewise(np)


def test__np_pmean() -> None:
    """Test function test__np_pmean."""
    with pytest.raises(TypeError):
        require_callable(mm._np_pmean)(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_pmean)(np)


def test__np_population_count() -> None:
    """Test function test__np_population_count."""
    with pytest.raises(TypeError):
        mm._np_population_count(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_population_count)(np)


def test__np_promotetypes() -> None:
    """Test function test__np_promotetypes."""
    with pytest.raises(IndexError):
        mm._np_promotetypes(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_promotetypes(np)


def test__np_psum() -> None:
    """Test function test__np_psum."""
    mm._np_psum(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_psum(np)


def test__np_raggeddot() -> None:
    """Test function test__np_raggeddot."""
    with pytest.raises(IndexError):
        mm._np_raggeddot(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_raggeddot(np)


def test__np_randombernoulli() -> None:
    """Test function test__np_randombernoulli."""
    with pytest.raises(TypeError):
        mm._np_randombernoulli(np, np.ones((2, 2)))
    mm._np_randombernoulli(np)


def test__np_randomcategorical() -> None:
    """Test function test__np_randomcategorical."""
    mm._np_randomcategorical(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_randomcategorical(np)


def test__np_randompermutation() -> None:
    """Test function test__np_randompermutation."""
    mm._np_randompermutation(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_randompermutation(np)


def test__np_randomtruncatednormal() -> None:
    """Test function test__np_randomtruncatednormal."""
    with pytest.raises(TypeError):
        mm._np_randomtruncatednormal(np, np.ones((2, 2)))
    mm._np_randomtruncatednormal(np)


def test__np_ravel_multi_index_() -> None:
    """Test function test__np_ravel_multi_index_."""
    with pytest.raises(TypeError):
        mm._np_ravel_multi_index_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_ravel_multi_index_(np)


def test__np_rawconv2d() -> None:
    """Test function test__np_rawconv2d."""
    with pytest.raises(IndexError):
        mm._np_rawconv2d(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_rawconv2d(np)


def test__np_rawmatmul() -> None:
    """Test function test__np_rawmatmul."""
    with pytest.raises(IndexError):
        mm._np_rawmatmul(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_rawmatmul(np)


def test__np_rawmerge() -> None:
    """Test function test__np_rawmerge."""
    mm._np_rawmerge(np, np.ones((2, 2)))
    _out = mm._np_rawmerge(np)
    assert len(coerce_operand_list(_out)) == 2


def test__np_rawop() -> None:
    """Test function test__np_rawop."""
    mm._np_rawop(np, np.ones((2, 2)))
    _out = mm._np_rawop(np)
    assert _out is None


def test__np_rawswitch() -> None:
    """Test function test__np_rawswitch."""
    mm._np_rawswitch(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_rawswitch(np)


def test__np_read_file() -> None:
    """Verify _np_read_file raises on missing path and returns empty byte array when no path is given."""
    with pytest.raises((OSError, RuntimeError, ValueError, TypeError)):
        mm._np_read_file(np, "/nonexistent_switcheroo_test_path_xyz.bin")
    assert bytes(np.asarray(mm._np_read_file(np)).flatten()[0]) == b""


def test__np_reduce_precision() -> None:
    """Test function test__np_reduce_precision."""
    with pytest.raises(TypeError):
        require_callable(mm._np_reduce_precision)(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_reduce_precision)(np)


def test__np_rem() -> None:
    """Test function test__np_rem."""
    mm._np_rem(np, np.ones((2, 2)))
    _out = mm._np_rem(np)
    assert _out is None


def test__np_resize_() -> None:
    """Test function test__np_resize_."""
    with pytest.raises(TypeError):
        mm._np_resize_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_resize_(np)


def test__np_result_type_() -> None:
    """Test function test__np_result_type_."""
    mm._np_result_type_(np, np.ones((2, 2)))
    with pytest.raises(ValueError):
        mm._np_result_type_(np)


def test__np_rfft() -> None:
    """Test function test__np_rfft."""
    mm._np_rfft(np, np.ones((2, 2)))
    _out = mm._np_rfft(np)
    assert _out is None


def test__np_rfft2() -> None:
    """Test function test__np_rfft2."""
    mm._np_rfft2(np, np.ones((2, 2)))
    _out = mm._np_rfft2(np)
    assert _out is None


def test__np_rfftfreq() -> None:
    """Test function test__np_rfftfreq."""
    mm._np_rfftfreq(np, np.ones((2, 2)))
    _out = mm._np_rfftfreq(np)
    assert _out is None


def test__np_rfftn() -> None:
    """Test function test__np_rfftn."""
    mm._np_rfftn(np, np.ones((2, 2)))
    _out = mm._np_rfftn(np)
    assert _out is None


def test__np_rfftnd() -> None:
    """Test function test__np_rfftnd."""
    mm._np_rfftnd(np, np.ones((2, 2)))
    _out = mm._np_rfftnd(np)
    assert _out is None


def test__np_rrelu() -> None:
    """Test function test__np_rrelu."""
    mm._np_rrelu(np, np.ones((2, 2)))
    _out = mm._np_rrelu(np)
    assert _out is None


def test__np_rsqrt() -> None:
    """Test function test__np_rsqrt."""
    mm._np_rsqrt(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_rsqrt)(np)


def test__np_scanop() -> None:
    """Test function test__np_scanop."""
    mm._np_scanop(np, np.ones((2, 2)))
    _out = mm._np_scanop(np)
    assert _out is None


def test__np_segment_sum() -> None:
    """Test function test__np_segment_sum."""
    with pytest.raises(TypeError):
        require_callable(mm._np_segment_sum)(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_segment_sum)(np)


def test__np_serialize_tensor() -> None:
    """Test function test__np_serialize_tensor."""
    mm._np_serialize_tensor(np, np.ones((2, 2)))
    _out = mm._np_serialize_tensor(np)
    assert np.asarray(_out).shape == ()


def test__np_shifted_chebyshev_polynomial_t() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_t."""
    mm._np_shifted_chebyshev_polynomial_t(np, np.ones((2, 2)))
    _out = mm._np_shifted_chebyshev_polynomial_t(np)
    assert _out is None


def test__np_shifted_chebyshev_polynomial_u() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_u."""
    mm._np_shifted_chebyshev_polynomial_u(np, np.ones((2, 2)))
    _out = mm._np_shifted_chebyshev_polynomial_u(np)
    assert _out is None


def test__np_shifted_chebyshev_polynomial_v() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_v."""
    mm._np_shifted_chebyshev_polynomial_v(np, np.ones((2, 2)))
    _out = mm._np_shifted_chebyshev_polynomial_v(np)
    assert _out is None


def test__np_shifted_chebyshev_polynomial_w() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_w."""
    mm._np_shifted_chebyshev_polynomial_w(np, np.ones((2, 2)))
    _out = mm._np_shifted_chebyshev_polynomial_w(np)
    assert _out is None


def test__np_sigmoid() -> None:
    """Test function test__np_sigmoid."""
    mm._np_sigmoid(np, np.ones((2, 2)))
    _out = mm._np_sigmoid(np)
    assert _out is None


def test__np_sobolsample() -> None:
    """Test function test__np_sobolsample."""
    with pytest.raises(TypeError):
        require_callable(mm._np_sobolsample)(np, coerce_int(np.ones((2, 2))))
    with pytest.raises(TypeError):
        require_callable(mm._np_sobolsample)(np)


def test__np_softmax() -> None:
    """Test function test__np_softmax."""
    mm._np_softmax(np, np.ones((2, 2)))
    _out = mm._np_softmax(np)
    assert _out is None


def test__np_sparsedensematmul() -> None:
    """Test function test__np_sparsedensematmul."""
    with pytest.raises(IndexError):
        mm._np_sparsedensematmul(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_sparsedensematmul(np)


def test__np_sparsemapvalues() -> None:
    """Test function test__np_sparsemapvalues."""
    with pytest.raises(IndexError):
        mm._np_sparsemapvalues(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_sparsemapvalues(np)


def test__np_sparsereducemax() -> None:
    """Test function test__np_sparsereducemax."""
    mm._np_sparsereducemax(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_sparsereducemax(np)


def test__np_sparsereshape() -> None:
    """Test function test__np_sparsereshape."""
    with pytest.raises(IndexError):
        mm._np_sparsereshape(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_sparsereshape(np)


def test__np_sparsesampledadd() -> None:
    """Test function test__np_sparsesampledadd."""
    with pytest.raises(IndexError):
        mm._np_sparsesampledadd(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_sparsesampledadd(np)


def test__np_sparsesegmentsum() -> None:
    """Test function test__np_sparsesegmentsum."""
    mm._np_sparsesegmentsum(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_sparsesegmentsum(np)


def test__np_sparsetranspose() -> None:
    """Test function test__np_sparsetranspose."""
    mm._np_sparsetranspose(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_sparsetranspose(np)


def test__np_stridedslice() -> None:
    """Test function test__np_stridedslice."""
    with pytest.raises(TypeError):
        require_callable(mm._np_stridedslice)(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        require_callable(mm._np_stridedslice)(np)


def test__np_switchop() -> None:
    """Test function test__np_switchop."""
    mm._np_switchop(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_switchop(np)


def test__np_tensor() -> None:
    """Test function test__np_tensor."""
    mm._np_tensor(np, np.ones((2, 2)))
    _out = mm._np_tensor(np)
    assert np.asarray(_out).shape == (0,)


def test__np_tensorarrayread() -> None:
    """Test function test__np_tensorarrayread."""
    with pytest.raises(IndexError):
        mm._np_tensorarrayread(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_tensorarrayread(np)


def test__np_tensorarraystack() -> None:
    """Test function test__np_tensorarraystack."""
    mm._np_tensorarraystack(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_tensorarraystack(np)


def test__np_tensorarraywrite() -> None:
    """Test function test__np_tensorarraywrite."""
    with pytest.raises(IndexError):
        mm._np_tensorarraywrite(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_tensorarraywrite(np)


def test__np_tensorconfig() -> None:
    """Test function test__np_tensorconfig."""
    try:
        cfg_first = mm._np_tensorconfig(np, np.ones((2, 2)))
        assert cfg_first is not None
    except ValueError as err:
        assert str(err) != ""
    cfg_default = mm._np_tensorconfig(np)
    assert cfg_default is not None


def test__np_trace() -> None:
    """Test function test__np_trace."""
    mm._np_trace(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_trace(np)


def test__np_trapz() -> None:
    """Test function test__np_trapz."""
    mm._np_trapz(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_trapz(np)


def test__np_trapz_() -> None:
    """Test function test__np_trapz_."""
    mm._np_trapz_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_trapz_(np)


def test__np_tri() -> None:
    """Test function test__np_tri."""
    with pytest.raises(ValueError):
        mm._np_tri(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_tri(np)


def test__np_trilindices() -> None:
    """Test function test__np_trilindices."""
    with pytest.raises(ValueError):
        mm._np_trilindices(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_trilindices(np)


def test__np_trilindicesfrom() -> None:
    """Test function test__np_trilindicesfrom."""
    mm._np_trilindicesfrom(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_trilindicesfrom(np)


def test__np_trimzeros() -> None:
    """Test function test__np_trimzeros."""
    mm._np_trimzeros(np, np.array([0, 1, 2, 0]))
    with pytest.raises(IndexError):
        mm._np_trimzeros(np)


def test__np_triuindices() -> None:
    """Test function test__np_triuindices."""
    with pytest.raises(ValueError):
        mm._np_triuindices(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_triuindices(np)


def test__np_triuindicesfrom() -> None:
    """Test function test__np_triuindicesfrom."""
    mm._np_triuindicesfrom(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_triuindicesfrom(np)


def test__np_truncate_div() -> None:
    """Test function test__np_truncate_div."""
    with pytest.raises(IndexError):
        mm._np_truncate_div(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_truncate_div(np)


def test__np_truncate_mod() -> None:
    """Test function test__np_truncate_mod."""
    with pytest.raises(IndexError):
        mm._np_truncate_mod(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_truncate_mod(np)


def test__np_uint() -> None:
    """Test function test__np_uint."""
    mm._np_uint(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_uint(np)


def test__np_uint8() -> None:
    """Test function test__np_uint8."""
    mm._np_uint8(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_uint8(np)


def test__np_union1d() -> None:
    """Test function test__np_union1d."""
    with pytest.raises(IndexError):
        mm._np_union1d(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_union1d(np)


def test__np_unpackbits() -> None:
    """Test function test__np_unpackbits."""
    with pytest.raises(TypeError):
        mm._np_unpackbits(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_unpackbits(np)


def test__np_unravelindex() -> None:
    """Test function test__np_unravelindex."""
    with pytest.raises(TypeError):
        mm._np_unravelindex(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_unravelindex(np)


def test__np_unwrap() -> None:
    """Test function test__np_unwrap."""
    mm._np_unwrap(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_unwrap(np)


def test__np_vander() -> None:
    """Test function test__np_vander."""
    with pytest.raises(ValueError):
        mm._np_vander(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_vander(np)


def test__np_vecdot() -> None:
    """Test function test__np_vecdot."""
    with pytest.raises(IndexError):
        mm._np_vecdot(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_vecdot(np)


def test__np_vectorize() -> None:
    """Test function test__np_vectorize."""
    with pytest.raises(ValueError):
        mm._np_vectorize(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_vectorize(np)


def test__np_write_file(tmp_path: os.PathLike[str]) -> None:
    """Verify `write_file` writes bytes to disk via `numpy_eager_registry`.

    Args:
        tmp_path (os.PathLike[str]): Temporary filesystem directory fixture.
    """
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    target = os.path.join(str(tmp_path), "sample.bin")
    write_fn = numpy_eager_registry.get("write_file")
    require_callable(write_fn)(np, target, b"hello_compiler")
    with open(target, "rb") as fh:
        assert fh.read() == b"hello_compiler"


def test__np_xlogy() -> None:
    """Test function test__np_xlogy."""
    with pytest.raises(TypeError):
        mm._np_xlogy(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_xlogy(np)


def test__np_zeros_() -> None:
    """Test function test__np_zeros_."""
    with pytest.raises(TypeError):
        mm._np_zeros_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_zeros_(np)


def test__np_zeros_like_() -> None:
    """Test function test__np_zeros_like_."""
    mm._np_zeros_like_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_zeros_like_(np)


def test_pmean_segment_clz() -> None:
    """Test function test_pmean_segment_clz."""
    import tests.backends.numpy.eager._math_advanced_test_ns as mm

    pmean_res = mm._np_pmean(np, np.ones((2, 2)), "x")
    seg_res = mm._np_segment_sum(np, np.ones((4,)), np.array([0, 0, 1, 1]))
    clz_res = mm._np_clz(np, np.array([1, 2, -1], dtype=np.int32))
    assert np.array_equal(np.asarray(pmean_res), np.ones((2, 2)))
    assert np.array_equal(seg_res, [2.0, 2.0])
    assert np.asarray(clz_res)[0] == 31
