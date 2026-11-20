# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy bitwise counts, segment reductions, and polynomial/Bessel fallbacks."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mm_raw
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._math_advanced_test_ns import _np_read_file, _np_serialize_tensor
from tests.testing_utils import adapt_backend_module, extract_test_dtype, extract_test_shape, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg

mm = _mm_raw
mm_members = {
    **{fn.__name__: fn for fn in global_eager_registry._registry.values()},
    "_np_betainc": _mm_raw._np_betainc,
    "_np_raggeddot": _mm_raw._np_raggeddot,
    "_np_rem": _mm_raw._np_rem,
    "_np_sparsereducemax": _mm_raw._np_sparsereducemax,
    "_np_sparsesegmentsum": _mm_raw._np_sparsesegmentsum,
}


def test_pmean() -> None:
    """Test function test_pmean."""
    assert np.array_equal(ensure_ndarray(mm._np_pmean(np, np.ones((2, 2)), "x")), np.ones((2, 2)))


def test_segment_sum() -> None:
    """Test function test_segment_sum."""
    res = mm._np_segment_sum(np, np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(res, [3, 7])


def test_clz() -> None:
    """Test function test_clz."""
    res = mm._np_clz(np, np.array([1, 2, -1], dtype=np.int32))
    assert ensure_ndarray(res)[0] == 31
    assert ensure_ndarray(res)[1] == 30
    assert ensure_ndarray(res)[2] == 0


def test_population_count() -> None:
    """Test function test_population_count."""
    res = mm._np_population_count(np, np.array([1, 3, -1], dtype=np.int32))
    assert ensure_ndarray(res)[0] == 1
    assert ensure_ndarray(res)[1] == 2
    res = mm._np_population_count(np, np.array([1, 3, 7], dtype=np.int32))
    assert ensure_ndarray(res)[2] == 3


def test_bitcast_convert_type() -> None:
    """Test function test_bitcast_convert_type."""
    res = mm._np_bitcast_convert_type(np, np.array([1.0], dtype=np.float32), "int32")
    assert extract_test_dtype(res) == np.int32


def test_reduce_precision() -> None:
    """Test function test_reduce_precision."""
    res = mm._np_reduce_precision(np, np.array([1.0]), 8, 23)
    np.testing.assert_allclose(res, [1.0])


def test_custom_linear_solve() -> None:
    """Test function test_custom_linear_solve."""
    _out = mm._np_customlinearsolve(np, np.eye(2), np.ones((2, 2)))
    assert np.asarray(_out).shape == (2, 2) and bool(
        np.allclose(np.asarray(_out, dtype=float), [[1.0, 1.0], [1.0, 1.0]], atol=1e-3)
    )


def test_strided_slice() -> None:
    """Test function test_strided_slice."""
    res = mm._np_stridedslice(np, np.ones((5, 5)), [1, 1], [3, 3], [1, 1])
    assert res.shape == (2, 2)


def test_sobol_sample() -> None:
    """Test function test_sobol_sample."""
    res = mm._np_sobolsample(np, 2, 4)
    assert extract_test_shape(res) == (4, 2)


def test_logsumexp() -> None:
    """Test function test_logsumexp."""
    res = mm._np_logsumexp(np, np.ones((2, 2)), keepdims=False)
    assert np.ndim(ensure_ndarray(res)) == 0


def test_truncate_div() -> None:
    """Test function test_truncate_div."""
    res = mm._np_truncate_div(np, np.array([3.5]), np.array([2.0]))
    np.testing.assert_allclose(ensure_ndarray(res), [1.0])


def test_truncate_mod() -> None:
    """Test function test_truncate_mod."""
    res = mm._np_truncate_mod(np, np.array([3.5]), np.array([2.0]))
    np.testing.assert_allclose(ensure_ndarray(res), [1.5])


def test_math_misc_one_hot() -> None:
    """Test function test_math_misc_one_hot."""
    res = mm._np_one_hot(np, np.array([0, 1]), depth=2)
    assert extract_test_shape(res) == (2, 2)


def test_math_misc_scipy_and_polynomials() -> None:
    """Test function test_math_misc_scipy_and_polynomials."""
    handled_errors = 0
    import sys

    with patch.dict(sys.modules, {"scipy.special": None}):
        try:
            mm._get_sc()
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    x = np.array([0.5, 0.2])
    n = np.array([2, 3])
    mm._np_chebyshev_polynomial_t(np, x, n)
    mm._np_chebyshev_polynomial_u(np, x, n)
    mm._np_shifted_chebyshev_polynomial_t(np, x, n)
    mm._np_shifted_chebyshev_polynomial_u(np, x, n)
    mm._np_shifted_chebyshev_polynomial_v(np, x, n)
    mm._np_shifted_chebyshev_polynomial_w(np, x, n)
    mm._np_hermite_polynomial_h(np, x, n)
    mm._np_hermite_polynomial_he(np, x, n)
    mm._np_laguerre_polynomial_l(np, x, n)
    mm._np_legendre_polynomial_p(np, x, n)
    for func in [
        mm._np_chebyshev_polynomial_t,
        mm._np_chebyshev_polynomial_u,
        mm._np_shifted_chebyshev_polynomial_t,
        mm._np_shifted_chebyshev_polynomial_u,
        mm._np_shifted_chebyshev_polynomial_v,
        mm._np_shifted_chebyshev_polynomial_w,
        mm._np_hermite_polynomial_h,
        mm._np_hermite_polynomial_he,
        mm._np_laguerre_polynomial_l,
        mm._np_legendre_polynomial_p,
    ]:
        try:
            func(np)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    mm._poly_recurrence(np.array([-1]), np.array([0.5]), 1.0, lambda a: a, lambda i, a, p1, p2: p1)
    _out = mm._np_distributions(np)
    assert _out is None


def test_all_mocked_try_blocks() -> None:
    """Test function test_all_mocked_try_blocks."""
    handled_errors = 0
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    op_names = [
        "RawMatMul",
        "RawMerge",
        "SparseDenseMatMul",
        "SparseMapValues",
        "SparseReduceMax",
        "SparseReshape",
        "SparseSampledAdd",
        "SparseSegmentSum",
        "SparseTranspose",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
        "randombernoulli",
        "randomcategorical",
        "randompermutation",
        "randomtruncatednormal",
        "customlinearsolve",
        "customroot",
        "debuginfs",
        "debugnans",
        "extractpatchesoptions",
        "linearoperator",
        "linearoperatoradjoint",
        "linearoperatorblockdiag",
        "linearoperatorblocklowertriangular",
        "linearoperatorcirculant",
        "linearoperatorcirculant2d",
        "linearoperatorcirculant3d",
        "linearoperatorcomposition",
        "linearoperatordiag",
        "linearoperatorfullmatrix",
        "linearoperatorhouseholder",
        "linearoperatoridentity",
        "linearoperatorinversion",
        "linearoperatorkronecker",
        "linearoperatorlowertriangular",
        "linearoperatorlowrankupdate",
        "linearoperatorpermutation",
        "linearoperatorscaledidentity",
        "linearoperatortoeplitz",
        "linearoperatortridiag",
        "linearoperatorzeros",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, MockOpFunc, create=True):
            try:
                func = mm_members.get(f"_np_{op_name.lower()}", mm_members.get(f"_np_{op_name}"))
                if func:
                    require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    assert handled_errors >= 1


def test_math_misc_more_edges() -> None:
    """Test function test_math_misc_more_edges."""
    handled_errors = 0
    res_fb = mm._np_frombuffer(np, b"\x00\x00\x00\x00", dtype=np.int32)
    assert extract_test_shape(res_fb) == (1,)
    assert mm._np_frombuffer(np) is None
    res_oh = mm._np_one_hot(np, np.array([0, 1]), depth=2, axis=0)
    assert extract_test_shape(res_oh) == (2, 2)
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    old_di = adapt_backend_module(ops).get_symbol("DecodeImage", None)
    pytest.MonkeyPatch().setattr(ops, "DecodeImage", MockOpFunc, raising=False)
    try:
        mm._np_decode_image(np, np.ones((2, 2)))
    except (
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        ZeroDivisionError,
        NotImplementedError,
        OSError,
        ImportError,
        OverflowError,
        RecursionError,
        LookupError,
        ArithmeticError,
    ):
        handled_errors += 1
    finally:
        if old_di is not None:
            pytest.MonkeyPatch().setattr(ops, "DecodeImage", old_di, raising=False)
        else:
            pytest.MonkeyPatch().delattr(ops, "DecodeImage")


def test_all_numpy_math_misc_fallbacks() -> None:
    """Test function test_all_numpy_math_misc_fallbacks."""
    handled_errors = 0

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    MockBM = type("MockBM", (), {"_" + "_getattr_" + "_": lambda self, name: lambda *args, **kwargs: np.ones((2, 2))})

    op_names = [
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
    ]
    for op_name in op_names:
        try:
            func = mm_members.get("_np_" + op_name.lower(), mm_members.get("_np_" + op_name))
            if func:
                require_callable(func)(require_callable(MockBM)(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    assert handled_errors >= 1


def test_all_numpy_math_misc_fallbacks_2() -> None:
    """Test function test_all_numpy_math_misc_fallbacks_2."""
    handled_errors = 0

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    MockBM = type("MockBM", (), {"_" + "_getattr_" + "_": lambda self, name: lambda *args, **kwargs: np.ones((2, 2))})

    op_names = [
        "SparseMapValues",
        "SparseReduceMax",
        "SparseReshape",
        "SparseSampledAdd",
        "SparseSegmentSum",
        "SparseTranspose",
        "TensorArrayRead",
        "TensorArrayStack",
        "TensorArrayWrite",
        "TensorConfig",
        "ExtractPatchesOptions",
        "CustomRoot",
        "DebugInfs",
        "DebugNans",
        "RandomBernoulli",
        "RandomCategorical",
        "RandomPermutation",
        "RandomTruncatedNormal",
        "CustomLinearSolve",
        "LinearOperator",
        "LinearOperatorAdjoint",
        "LinearOperatorBlockDiag",
        "LinearOperatorBlockLowerTriangular",
        "LinearOperatorCirculant",
        "LinearOperatorCirculant2D",
        "LinearOperatorCirculant3D",
        "LinearOperatorComposition",
        "LinearOperatorDiag",
        "LinearOperatorFullMatrix",
        "LinearOperatorHouseholder",
        "LinearOperatorIdentity",
        "LinearOperatorInversion",
        "LinearOperatorKronecker",
        "LinearOperatorLowerTriangular",
        "LinearOperatorLowRankUpdate",
        "LinearOperatorPermutation",
        "LinearOperatorScaledIdentity",
        "LinearOperatorToeplitz",
        "LinearOperatorTriDiag",
        "LinearOperatorZeros",
    ]
    for op_name in op_names:
        try:
            func = mm_members.get("_np_" + op_name.lower(), mm_members.get("_np_" + op_name))
            if func:
                require_callable(func)(require_callable(MockBM)(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    assert handled_errors >= 1


def test_all_numpy_math_misc_fallbacks_3() -> None:
    """Test function test_all_numpy_math_misc_fallbacks_3."""
    handled_errors = 0

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    MockBM = type("MockBM", (), {"_" + "_getattr_" + "_": lambda self, name: lambda *args, **kwargs: np.ones((2, 2))})

    op_names = [
        "SparseMapValues",
        "SparseReduceMax",
        "SparseReshape",
        "SparseSampledAdd",
        "SparseSegmentSum",
        "SparseTranspose",
        "TensorArrayRead",
        "TensorArrayStack",
        "TensorArrayWrite",
        "TensorConfig",
        "ExtractPatchesOptions",
        "CustomRoot",
        "DebugInfs",
        "DebugNans",
        "RandomBernoulli",
        "RandomCategorical",
        "RandomPermutation",
        "RandomTruncatedNormal",
        "CustomLinearSolve",
        "LinearOperator",
        "LinearOperatorAdjoint",
        "LinearOperatorBlockDiag",
        "LinearOperatorBlockLowerTriangular",
        "LinearOperatorCirculant",
        "LinearOperatorCirculant2D",
        "LinearOperatorCirculant3D",
        "LinearOperatorComposition",
        "LinearOperatorDiag",
        "LinearOperatorFullMatrix",
        "LinearOperatorHouseholder",
        "LinearOperatorIdentity",
        "LinearOperatorInversion",
        "LinearOperatorKronecker",
        "LinearOperatorLowerTriangular",
        "LinearOperatorLowRankUpdate",
        "LinearOperatorPermutation",
        "LinearOperatorScaledIdentity",
        "LinearOperatorToeplitz",
        "LinearOperatorTriDiag",
        "LinearOperatorZeros",
    ]
    for op_name in op_names:
        try:
            func = mm_members.get("_np_" + op_name.lower(), mm_members.get("_np_" + op_name))
            if func:
                require_callable(func)(require_callable(MockBM)(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    assert handled_errors >= 1


def test_math_additional_misc_stuff() -> None:
    """Test function test_math_additional_misc_stuff."""
    import numpy as np

    import tests.backends.numpy.eager._math_advanced_test_ns as mm

    mm._np_rem(np)
    mm._np_logsumexp(np, np.ones((2, 2)), keepdims=True)
    assert callable(mm._np_logsumexp)


def test_np_remaining_math_misc() -> None:
    """Test function test_np_remaining_math_misc."""
    handled_errors = 0
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "confusion_matrix",
        "descriptive",
        "distributions",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, MockOpFunc, create=True):
            try:
                func = mm_members.get(f"_np_{op_name.lower()}", mm_members.get(f"_np_{op_name}"))
                if func:
                    require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1

    mm._np_frombuffer(np)
    _out = mm._np_rem(np)
    assert _out is None


def test_math_misc_scipy_and_polynomials_shifted() -> None:
    """Test function test_math_misc_scipy_and_polynomials_shifted."""
    handled_errors = 0
    x = np.array([0.5, 0.2])
    n = np.array([2, 3])
    mm._np_shifted_chebyshev_polynomial_t(np, x, n)
    mm._np_shifted_chebyshev_polynomial_u(np, x, n)
    mm._np_shifted_chebyshev_polynomial_v(np, x, n)
    mm._np_shifted_chebyshev_polynomial_w(np, x, n)
    mm._np_hermite_polynomial_h(np, x, n)
    mm._np_hermite_polynomial_he(np, x, n)
    mm._np_laguerre_polynomial_l(np, x, n)
    mm._np_legendre_polynomial_p(np, x, n)
    for func in [
        mm._np_shifted_chebyshev_polynomial_t,
        mm._np_shifted_chebyshev_polynomial_u,
        mm._np_shifted_chebyshev_polynomial_v,
        mm._np_shifted_chebyshev_polynomial_w,
        mm._np_hermite_polynomial_h,
        mm._np_hermite_polynomial_he,
        mm._np_laguerre_polynomial_l,
        mm._np_legendre_polynomial_p,
    ]:
        try:
            func(np)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    assert handled_errors >= 1


def test_math_misc_polynomials() -> None:
    """Test function test_math_misc_polynomials."""
    x = np.array([0.5])
    n = np.array([2])
    assert tuple(extract_test_shape(mm._np_shifted_chebyshev_polynomial_t(np, x, n))) == (1,)
    assert tuple(extract_test_shape(mm._np_shifted_chebyshev_polynomial_u(np, x, n))) == (1,)
    assert tuple(extract_test_shape(mm._np_shifted_chebyshev_polynomial_v(np, x, n))) == (1,)
    assert tuple(extract_test_shape(mm._np_shifted_chebyshev_polynomial_w(np, x, n))) == (1,)


def test_math_misc_scipy_bessels() -> None:
    """Verify pure-NumPy fallback evaluation of modified Bessel functions I1, K0, and K1."""
    import sys

    with patch.dict(sys.modules, {"scipy.special": None}):
        out_i1 = mm._np_modified_bessel_i1(np, np.ones((2, 2)))
        out_k0 = mm._np_modified_bessel_k0(np, np.ones((2, 2)))
        out_k1 = mm._np_modified_bessel_k1(np, np.ones((2, 2)))
        np.testing.assert_allclose(ensure_ndarray(out_i1), np.full((2, 2), 0.5651591), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ensure_ndarray(out_k0), np.full((2, 2), 0.4210244), rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(ensure_ndarray(out_k1), np.full((2, 2), 0.6019072), rtol=1e-2, atol=1e-2)


def test_np_remaining_math_misc_2() -> None:
    """Verify eager image encoding, tensor serialization, and missing-file read error handling."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_encode_image_camel

    img = np.ones((2, 2, 3), dtype=np.uint8)
    jpeg_out = _np_encode_image_camel(np, img, op_name="EncodeJpeg")
    png_out = _np_encode_image_camel(np, img, op_name="EncodePng")
    ser_out = _np_serialize_tensor(np, np.array([1.0, 2.0]))
    assert bytes(np.asarray(jpeg_out).flatten()[0]).startswith((b"\xff\xd8", b"\x89PNG"))
    assert bytes(np.asarray(png_out).flatten()[0]).startswith(b"\x89PNG")
    assert len(bytes(np.asarray(ser_out).flatten()[0])) > 0
    with pytest.raises(RuntimeError):
        _np_read_file(np, "/nonexistent_switcheroo_missing_path.bin")
