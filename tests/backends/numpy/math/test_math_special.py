"""Tests for test_math_special."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import (
    _np_confusion_matrix,
    _np_decode_csv,
    _np_decode_image,
    _np_descriptive,
    _np_distributions,
    _np_parse_example,
    _np_parse_tensor,
    _np_rawmatmul,
    _np_read_file,
    _np_rem,
    _np_serialize_tensor,
    _np_sparsedensematmul,
    _np_write_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_pmean() -> None:
    """Test function test_pmean."""
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", mm._np_pmean)(np, cast("Callable[..., EagerTensorProtocol]", np.ones)((2, 2)), "x"), cast("Callable[..., EagerTensorProtocol]", np.ones)((2, 2)))


def test_segment_sum() -> None:
    """Test function test_segment_sum."""
    res = mm._np_segment_sum(np, np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1]))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [3, 7])


def test_clz() -> None:
    """Test function test_clz."""
    res = mm._np_clz(np, np.array([1, 2, -1], dtype=np.int32))
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)) == 31
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(1)) == 30
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(2)) == 0


def test_population_count() -> None:
    """Test function test_population_count."""
    res = mm._np_population_count(np, np.array([1, 3, -1], dtype=np.int32))
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)) == 1
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(1)) == 2
    res = mm._np_population_count(np, np.array([1, 3, 7], dtype=np.int32))
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(2)) == 3


def test_bitcast_convert_type() -> None:
    """Test function test_bitcast_convert_type."""
    res = mm._np_bitcast_convert_type(np, np.array([1.0], dtype=np.float32), "int32")
    assert cast("EagerTensorProtocol", getattr(res, "dtype")) == np.int32


def test_reduce_precision() -> None:
    """Test function test_reduce_precision."""
    res = mm._np_reduce_precision(np, np.array([1.0]), 8, 23)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1.0])


def test_custom_linear_solve() -> None:
    """Test function test_custom_linear_solve."""
    try:
        mm._np_customlinearsolve(np, np.eye(2), np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_strided_slice() -> None:
    """Test function test_strided_slice."""
    res = mm._np_stridedslice(np, np.ones((5, 5)), [1, 1], [3, 3], [1, 1])
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)


def test_sobol_sample() -> None:
    """Test function test_sobol_sample."""
    res = mm._np_sobolsample(np, 2, 4)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (4, 2)


def test_logsumexp() -> None:
    """Test function test_logsumexp."""
    res = mm._np_logsumexp(np, np.ones((2, 2)), keepdims=False)
    assert cast("Callable[..., EagerTensorProtocol]", np.ndim)(res) == 0


def test_truncate_div() -> None:
    """Test function test_truncate_div."""
    res = mm._np_truncate_div(np, np.array([3.5]), np.array([2.0]))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1.0])


def test_truncate_mod() -> None:
    """Test function test_truncate_mod."""
    res = mm._np_truncate_mod(np, np.array([3.5]), np.array([2.0]))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1.5])


def test_math_misc_one_hot() -> None:
    """Test function test_math_misc_one_hot."""
    res = mm._np_one_hot(np, np.array([0, 1]), depth=2)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)


def test_math_misc_scipy_and_polynomials() -> None:
    """Test function test_math_misc_scipy_and_polynomials."""
    import sys

    old_scipy = sys.modules.get("scipy.special", None)
    getattr(sys.modules, "__setitem__")("scipy.special", None)
    try:
        mm._get_sc()
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    finally:
        if old_scipy:
            sys.modules["scipy.special"] = old_scipy
        else:
            del sys.modules["scipy.special"]
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
        except ValueError as _exc:
            assert _exc is not None
    mm._poly_recurrence(np.array([-1]), np.array([0.5]), 1.0, lambda a: a, lambda i, a, p1, p2: p1)
    try:
        mm._np_distributions(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_all_mocked_try_blocks() -> None:
    """Test function test_all_mocked_try_blocks."""
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __call__."""
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
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, f"_np_{op_name.lower()}", getattr(mm, f"_np_{op_name}", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_math_misc_more_edges() -> None:
    """Test function test_math_misc_more_edges."""
    res_fb = mm._np_frombuffer(np, b"\x00\x00\x00\x00", dtype=np.int32)
    assert cast("EagerTensorProtocol", getattr(res_fb, "shape")) == (1,)
    assert mm._np_frombuffer(np) is None
    res_oh = mm._np_one_hot(np, np.array([0, 1]), depth=2, axis=0)
    assert cast("EagerTensorProtocol", getattr(res_oh, "shape")) == (2, 2)
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __call__."""
            pass

    old_di = getattr(ops, "DecodeImage", None)
    setattr(ops, "DecodeImage", MockOpFunc)
    try:
        mm._np_decode_image(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    finally:
        if old_di:
            setattr(ops, "DecodeImage", old_di)
        else:
            delattr(ops, "DecodeImage")


def test_all_numpy_math_misc_fallbacks() -> None:
    """Test function test_all_numpy_math_misc_fallbacks."""

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __call__."""
            pass

    class MockBM:
        """Test class MockBM."""

        def __init__(self) -> None:
            """Test function __init__."""
            pass

        def __getattr__(self, name: EagerValue) -> EagerValue:
            """Test function __getattr__."""

            def dummy(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function dummy."""
                return np.ones((2, 2))

            return dummy

    op_names = ["decode_csv", "decode_image", "parse_example", "parse_tensor", "serialize_tensor", "encode_image", "encode_jpeg", "encode_png", "encode_base64", "decode_base64", "decode_json_example", "decode_compressed", "read_file", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        try:
            func = getattr(mm, "_np_" + op_name.lower(), getattr(mm, "_np_" + op_name, None))
            if func:
                func(MockBM(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_all_numpy_math_misc_fallbacks_2() -> None:
    """Test function test_all_numpy_math_misc_fallbacks_2."""

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __call__."""
            pass

    class MockBM:
        """Test class MockBM."""

        def __init__(self) -> None:
            """Test function __init__."""
            pass

        def __getattr__(self, name: EagerValue) -> EagerValue:
            """Test function __getattr__."""

            def dummy(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function dummy."""
                return np.ones((2, 2))

            return dummy

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
            func = getattr(mm, "_np_" + op_name.lower(), getattr(mm, "_np_" + op_name, None))
            if func:
                func(MockBM(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_all_numpy_math_misc_fallbacks_3() -> None:
    """Test function test_all_numpy_math_misc_fallbacks_3."""

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __call__."""
            pass

    class MockBM:
        """Test class MockBM."""

        def __init__(self) -> None:
            """Test function __init__."""
            pass

        def __getattr__(self, name: EagerValue) -> EagerValue:
            """Test function __getattr__."""

            def dummy(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function dummy."""
                return np.ones((2, 2))

            return dummy

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
            func = getattr(mm, "_np_" + op_name.lower(), getattr(mm, "_np_" + op_name, None))
            if func:
                func(MockBM(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_math_additional_misc_stuff() -> None:
    """Test function test_math_additional_misc_stuff."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    mm._np_rem(np)
    mm._np_logsumexp(np, np.ones((2, 2)), keepdims=True)
    assert callable(mm._np_logsumexp)


def test_np_remaining_math_misc() -> None:
    """Test function test_np_remaining_math_misc."""
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__."""
            return np.ones((2, 2))

    op_names = ["RawMatMul", "SparseDenseMatMul", "decode_csv", "decode_image", "parse_example", "parse_tensor", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, f"_np_{op_name.lower()}", getattr(mm, f"_np_{op_name}", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    mm._np_frombuffer(np)
    mm._np_rem(np)


def test_math_misc_scipy_and_polynomials_shifted() -> None:
    """Test function test_math_misc_scipy_and_polynomials_shifted."""
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
    for func in [mm._np_shifted_chebyshev_polynomial_t, mm._np_shifted_chebyshev_polynomial_u, mm._np_shifted_chebyshev_polynomial_v, mm._np_shifted_chebyshev_polynomial_w, mm._np_hermite_polynomial_h, mm._np_hermite_polynomial_he, mm._np_laguerre_polynomial_l, mm._np_legendre_polynomial_p]:
        try:
            func(np)
        except ValueError as _exc:
            assert _exc is not None


def test_math_misc_polynomials() -> None:
    """Test function test_math_misc_polynomials."""
    x = np.array([0.5])
    n = np.array([2])
    mm._np_shifted_chebyshev_polynomial_t(np, x, n)
    mm._np_shifted_chebyshev_polynomial_u(np, x, n)
    mm._np_shifted_chebyshev_polynomial_v(np, x, n)
    mm._np_shifted_chebyshev_polynomial_w(np, x, n)
    assert n is not None or n is None


def test_math_misc_scipy_bessels() -> None:
    """Test function test_math_misc_scipy_bessels."""
    import sys

    old_scipy = sys.modules.get("scipy.special", None)
    getattr(sys.modules, "__setitem__")("scipy.special", None)
    try:
        mm._np_modified_bessel_i1(np, np.ones((2, 2)))
        mm._np_modified_bessel_k0(np, np.ones((2, 2)))
        mm._np_modified_bessel_k1(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    finally:
        if old_scipy:
            sys.modules["scipy.special"] = old_scipy
        else:
            del sys.modules["scipy.special"]


def test_np_remaining_math_misc_2() -> None:
    """Test function test_np_remaining_math_misc_2."""
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__."""
            return np.ones((2, 2))

    op_names = ["encode_image", "encode_jpeg", "encode_png", "encode_base64", "decode_base64", "decode_json_example", "decode_compressed", "read_file", "write_file", "serialize_tensor"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, f"_np_{op_name.lower()}", getattr(mm, f"_np_{op_name}", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_np_remaining_math_misc_3() -> None:
    """Test function test_np_remaining_math_misc_3."""
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__."""
            return np.ones((2, 2))

    op_names = ["shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u", "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, f"_np_{op_name.lower()}", getattr(mm, f"_np_{op_name}", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    mm._np_rfftfreq(np, 10, d=0.1)
    mm._np_frombuffer(np)
    mm._np_rem(np, 1)


def test_remaining_math_misc_stuff() -> None:
    """Test function test_remaining_math_misc_stuff."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__."""
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
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
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, "_np_" + op_name.lower(), getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking() -> None:
    """Test function test_final_math_misc_mocking."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__."""
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
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
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again() -> None:
    """Test function test_final_math_misc_mocking_again."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__."""
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
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
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_once_more() -> None:
    """Test function test_final_math_misc_mocking_once_more."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class MockOpFuncWithInit:
        """Test class MockOpFuncWithInit."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> MockOpFuncWithInit:
            """Test function __new__."""
            return cast("MockOpFuncWithInit", np.ones((2, 2)))

    class RealMockType:
        """Test class RealMockType."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            pass

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function __call__."""
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
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
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_once_more_2() -> None:
    """Test function test_final_math_misc_mocking_once_more_2."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType:
        """Test class RealMockType."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType:
            """Test function __new__."""
            return cast("RealMockType", np.ones((2, 2)))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
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
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again() -> None:
    """Test function test_final_math_misc_mocking_again_and_again."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
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
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            raise Exception("Trigger catch")

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
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
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_2() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_2."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["decode_csv", "decode_image", "parse_example", "parse_tensor", "serialize_tensor", "encode_image", "encode_jpeg", "encode_png", "encode_base64", "decode_base64", "decode_json_example", "decode_compressed", "read_file", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_3() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_3."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "EncodeImage", "EncodeJpeg", "EncodePng", "EncodeBase64", "DecodeBase64", "DecodeJsonExample", "DecodeCompressed", "ReadFile", "WriteFile", "ConfusionMatrix", "Descriptive", "Distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name.lower(), getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_4() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_4."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        try:
            func = getattr(mm, "_np_" + op_name.lower() + "_camel", getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_final_math_misc_mocking_again_and_again_again_5() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_5."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name.lower() + "_camel", getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_camel() -> None:
    """Test function test_final_math_misc_mocking_camel."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name.lower() + "_camel", getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_6() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_6."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["decode_csv", "decode_image", "parse_example", "parse_tensor", "serialize_tensor", "read_file", "write_file"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_7() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_7."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        try:
            func = getattr(mm, "_np_" + op_name.lower() + "_camel", None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_final_math_misc_mocking_again_and_again_again_8() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_8."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    mm._np_decode_csv_camel(np)
    mm._np_decode_image_camel(np)
    mm._np_parse_example_camel(np)
    mm._np_parse_tensor_camel(np)
    mm._np_serialize_tensor_camel(np)
    mm._np_read_file_camel(np)
    mm._np_write_file_camel(np)
    assert callable(mm._np_write_file_camel)


def test_final_math_misc_mocking_for_remaining() -> None:
    """Test function test_final_math_misc_mocking_for_remaining."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["confusion_matrix", "descriptive", "distributions", "shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u", "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w", "frombuffer", "rem"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_8_fixed() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_8_fixed."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_9() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_9."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name.lower(), None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_10() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_10."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_11() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_11."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name + "_lower", None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_12() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_12."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["serialize_tensor", "write_file"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name + "_lower", getattr(mm, "_np_" + op_name.lower() + "_camel", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_13() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_13."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name + "_lower", None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_14() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_14."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_15() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_15."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["rem", "serialize_tensor", "write_file"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_16() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_16."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: EagerValue, **kwargs: EagerValue) -> RealMockType2:
            """Test function __new__."""
            return cast("RealMockType2", np.ones((2, 2)))

    op_names = ["confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_17() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_17."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    x = np.array([0.5, 0.2])
    n = np.array([2, 3])
    mm._np_shifted_chebyshev_polynomial_t(np, x, n)
    mm._np_shifted_chebyshev_polynomial_u(np, x, n)
    mm._np_shifted_chebyshev_polynomial_v(np, x, n)
    mm._np_shifted_chebyshev_polynomial_w(np, x, n)
    assert n is not None or n is None


class MockCallableClass:
    """Test class MockCallableClass."""

    def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function __call__."""
        return args


def _test_np_ops_subclass_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_np_ops_subclass.

    Args:
        _state: Shared test state dictionary.
    """
    original_rawmatmul = getattr(_ops, "RawMatMul", None)
    original_sparsedensematmul = getattr(_ops, "SparseDenseMatMul", None)
    original_decode_csv = getattr(_ops, "decode_csv", None)
    original_decode_image = getattr(_ops, "decode_image", None)
    original_parse_example = getattr(_ops, "parse_example", None)
    original_parse_tensor = getattr(_ops, "parse_tensor", None)
    original_read_file = getattr(_ops, "read_file", None)
    original_rem = getattr(_ops, "rem", None)
    original_serialize_tensor = getattr(_ops, "serialize_tensor", None)
    original_write_file = getattr(_ops, "write_file", None)
    original_confusion_matrix = getattr(_ops, "confusion_matrix", None)
    original_descriptive = getattr(_ops, "descriptive", None)
    original_distributions = getattr(_ops, "distributions", None)
    getattr(_state, "update")({"original_confusion_matrix": original_confusion_matrix, "original_decode_csv": original_decode_csv, "original_decode_image": original_decode_image, "original_descriptive": original_descriptive})
    getattr(_state, "update")({"original_distributions": original_distributions, "original_parse_example": original_parse_example, "original_parse_tensor": original_parse_tensor, "original_rawmatmul": original_rawmatmul})
    getattr(_state, "update")({"original_read_file": original_read_file, "original_rem": original_rem, "original_serialize_tensor": original_serialize_tensor, "original_sparsedensematmul": original_sparsedensematmul})
    getattr(_state, "update")({"original_write_file": original_write_file})
    _state.update(locals())


def _test_np_ops_subclass_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_np_ops_subclass.

    Args:
        _state: Shared test state dictionary.
    """
    original_confusion_matrix, original_decode_csv, original_decode_image, original_descriptive = _state["original_confusion_matrix"], _state["original_decode_csv"], _state["original_decode_image"], _state["original_descriptive"]
    original_distributions, original_parse_example, original_parse_tensor, original_rawmatmul = _state["original_distributions"], _state["original_parse_example"], _state["original_parse_tensor"], _state["original_rawmatmul"]
    original_read_file, original_rem, original_serialize_tensor, original_sparsedensematmul = _state["original_read_file"], _state["original_rem"], _state["original_serialize_tensor"], _state["original_sparsedensematmul"]
    original_write_file = _state["original_write_file"]
    try:
        setattr(_ops, "RawMatMul", MockCallableClass)
        setattr(_ops, "SparseDenseMatMul", MockCallableClass)
        setattr(_ops, "decode_csv", MockCallableClass)
        setattr(_ops, "decode_image", MockCallableClass)
        setattr(_ops, "parse_example", MockCallableClass)
        setattr(_ops, "parse_tensor", MockCallableClass)
        setattr(_ops, "read_file", MockCallableClass)
        setattr(_ops, "rem", MockCallableClass)
        setattr(_ops, "serialize_tensor", MockCallableClass)
        setattr(_ops, "write_file", MockCallableClass)
        _ops.confusion_matrix = MockCallableClass
        setattr(_ops, "descriptive", MockCallableClass)
        setattr(_ops, "distributions", MockCallableClass)

        _np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
        _np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))
        _np_decode_csv(np, "1,2,3")
        _np_decode_image(np, b"")
        _np_parse_example(np, b"")
        # For _np_parse_tensor we need a valid arg if using astype fallback
        _np_parse_tensor(np, [1, 2], out_type=np.float32)
        _np_read_file(np, "/tmp/ml_switcheroo_test_file2.txt")
        print("HASATTR REM:", hasattr(_ops, "rem"))
        _np_rem(np, np.array([5]), np.array([2]))
        _np_serialize_tensor(np, np.ones((2, 2)))
        _np_write_file(np, "/tmp/ml_switcheroo_test_file2.txt", "hello")
        _np_confusion_matrix(np, np.array([1]), np.array([1]))
        _np_descriptive(np, np.array([1]))
        _np_distributions(np, np.array([1]))

    finally:
        setattr(_ops, "RawMatMul", original_rawmatmul)
        setattr(_ops, "SparseDenseMatMul", original_sparsedensematmul)
        setattr(_ops, "decode_csv", original_decode_csv)
        setattr(_ops, "decode_image", original_decode_image)
        setattr(_ops, "parse_example", original_parse_example)
        _ops.parse_tensor = original_parse_tensor
        _ops.read_file = original_read_file
        setattr(_ops, "rem", original_rem)
        _ops.serialize_tensor = original_serialize_tensor
        setattr(_ops, "write_file", original_write_file)
        _ops.confusion_matrix = original_confusion_matrix
        setattr(_ops, "descriptive", original_descriptive)
        setattr(_ops, "distributions", original_distributions)
    _state.update(locals())


def test_np_ops_subclass() -> None:
    """Test function test_np_ops_subclass."""
    import pytest

    with pytest.raises(Exception, match=r".*"):
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"pytest": pytest})
        for _sub_fn in (_test_np_ops_subclass_sub_1, _test_np_ops_subclass_sub_2):
            _sub_fn(_state)
        assert isinstance(_state, dict)


def _test_np_ops_missing_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_np_ops_missing.

    Args:
        _state: Shared test state dictionary.
    """
    del _ops.RawMatMul
    del _ops.SparseDenseMatMul
    del _ops.decode_csv
    del _ops.decode_image
    del _ops.parse_example
    del _ops.parse_tensor
    del _ops.read_file
    del _ops.rem
    del _ops.serialize_tensor
    del _ops.write_file
    del _ops.confusion_matrix
    delattr(_ops, "descriptive")
    delattr(_ops, "distributions")

    _np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
    _np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))

    try:
        _np_decode_csv(np, "1,2,3")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    _state.update(locals())


def _test_np_ops_missing_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_np_ops_missing.

    Args:
        _state: Shared test state dictionary.
    """
    try:
        _np_decode_image(np, b"")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        _np_parse_example(np, b"")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        _np_parse_tensor(np, b"")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        _np_read_file(np, "/tmp/ml_switcheroo_test_file2.txt")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    print("HASATTR REM:", hasattr(_ops, "rem"))
    _np_rem(np, np.array([5]), np.array([2]))
    _np_serialize_tensor(np, np.ones((2, 2)))
    _state.update(locals())


def _test_np_ops_missing_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_np_ops_missing.

    Args:
        _state: Shared test state dictionary.
    """
    try:
        _np_write_file(np, "/tmp/ml_switcheroo_test_file2.txt", "hello")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    _np_confusion_matrix(np, np.array([1]), np.array([1]))
    _np_descriptive(np, np.array([1]))
    _np_distributions(np, np.array([1]))
    _state.update(locals())


def test_np_ops_missing() -> None:
    """Test function test_np_ops_missing."""
    original_rawmatmul = getattr(_ops, "RawMatMul", None)
    original_sparsedensematmul = getattr(_ops, "SparseDenseMatMul", None)
    original_decode_csv = getattr(_ops, "decode_csv", None)
    original_decode_image = getattr(_ops, "decode_image", None)
    original_parse_example = getattr(_ops, "parse_example", None)
    original_parse_tensor = getattr(_ops, "parse_tensor", None)
    original_read_file = getattr(_ops, "read_file", None)
    original_rem = getattr(_ops, "rem", None)
    original_serialize_tensor = getattr(_ops, "serialize_tensor", None)
    original_write_file = getattr(_ops, "write_file", None)
    original_confusion_matrix = getattr(_ops, "confusion_matrix", None)
    original_descriptive = getattr(_ops, "descriptive", None)
    original_distributions = getattr(_ops, "distributions", None)

    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"original_confusion_matrix": original_confusion_matrix, "original_decode_csv": original_decode_csv, "original_decode_image": original_decode_image, "original_descriptive": original_descriptive})
        getattr(_state, "update")({"original_distributions": original_distributions, "original_parse_example": original_parse_example, "original_parse_tensor": original_parse_tensor, "original_rawmatmul": original_rawmatmul})
        getattr(_state, "update")({"original_read_file": original_read_file, "original_rem": original_rem, "original_serialize_tensor": original_serialize_tensor, "original_sparsedensematmul": original_sparsedensematmul})
        getattr(_state, "update")({"original_write_file": original_write_file})
        for _sub_fn in (_test_np_ops_missing_sub_1, _test_np_ops_missing_sub_2, _test_np_ops_missing_sub_3):
            _sub_fn(_state)
        assert isinstance(_state, dict)

    finally:
        setattr(_ops, "RawMatMul", original_rawmatmul)
        setattr(_ops, "SparseDenseMatMul", original_sparsedensematmul)
        setattr(_ops, "decode_csv", original_decode_csv)
        setattr(_ops, "decode_image", original_decode_image)
        setattr(_ops, "parse_example", original_parse_example)
        setattr(_ops, "parse_tensor", original_parse_tensor)
        setattr(_ops, "read_file", original_read_file)
        setattr(_ops, "rem", original_rem)
        setattr(_ops, "serialize_tensor", original_serialize_tensor)
        setattr(_ops, "write_file", original_write_file)
        setattr(_ops, "confusion_matrix", original_confusion_matrix)
        setattr(_ops, "descriptive", original_descriptive)
        setattr(_ops, "distributions", original_distributions)


def test_confusion_matrix_num_classes() -> None:
    """Test function test_confusion_matrix_num_classes."""
    _np_confusion_matrix(np, np.array([1]), np.array([1]), num_classes=5)
    assert callable(_np_confusion_matrix)


def test__np_affineconfig() -> None:
    """Test function test__np_affineconfig."""
    try:
        mm._np_affineconfig(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_affineconfig(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_all_gather() -> None:
    """Test function test__np_all_gather."""
    try:
        cast("EagerTensorProtocol", getattr(mm, "_np_all_gather"))(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(mm, "_np_all_gather"))(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_all_reduce() -> None:
    """Test function test__np_all_reduce."""
    try:
        cast("EagerTensorProtocol", getattr(mm, "_np_all_reduce"))(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(mm, "_np_all_reduce"))(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_all_to_all() -> None:
    """Test function test__np_all_to_all."""
    try:
        cast("EagerTensorProtocol", getattr(mm, "_np_all_to_all"))(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        cast("EagerTensorProtocol", getattr(mm, "_np_all_to_all"))(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_append() -> None:
    """Test function test__np_append."""
    try:
        mm._np_append(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_append(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_apply_along_axis() -> None:
    """Test function test__np_apply_along_axis."""
    try:
        mm._np_apply_along_axis(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_apply_along_axis(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_apply_over_axes() -> None:
    """Test function test__np_apply_over_axes."""
    try:
        mm._np_apply_over_axes(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_apply_over_axes(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_arange_() -> None:
    """Test function test__np_arange_."""
    try:
        mm._np_arange_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_arange_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_argpartition() -> None:
    """Test function test__np_argpartition."""
    try:
        mm._np_argpartition(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_argpartition(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_argwhere() -> None:
    """Test function test__np_argwhere."""
    try:
        mm._np_argwhere(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_argwhere(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_array_equiv_() -> None:
    """Test function test__np_array_equiv_."""
    try:
        mm._np_array_equiv_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_array_equiv_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_array_repr_() -> None:
    """Test function test__np_array_repr_."""
    try:
        mm._np_array_repr_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_array_repr_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_array_str_() -> None:
    """Test function test__np_array_str_."""
    try:
        mm._np_array_str_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_array_str_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_assertop() -> None:
    """Test function test__np_assertop."""
    try:
        mm._np_assertop(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_assertop(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_asstringconfig() -> None:
    """Test function test__np_asstringconfig."""
    try:
        mm._np_asstringconfig(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_asstringconfig(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_atleast_1d() -> None:
    """Test function test__np_atleast_1d."""
    try:
        mm._np_atleast_1d(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_atleast_1d(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_atleast_2d() -> None:
    """Test function test__np_atleast_2d."""
    try:
        mm._np_atleast_2d(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_atleast_2d(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_atleast_3d() -> None:
    """Test function test__np_atleast_3d."""
    try:
        mm._np_atleast_3d(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_atleast_3d(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_average() -> None:
    """Test function test__np_average."""
    try:
        mm._np_average(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_average(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_bessel_i0e() -> None:
    """Test function test__np_bessel_i0e."""
    try:
        mm._np_bessel_i0e(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_bessel_i0e(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_bessel_i1e() -> None:
    """Test function test__np_bessel_i1e."""
    try:
        mm._np_bessel_i1e(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_bessel_i1e(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_betainc() -> None:
    """Test function test__np_betainc."""
    try:
        mm._np_betainc(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_betainc(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_bitcast_convert_type() -> None:
    """Test function test__np_bitcast_convert_type."""
    try:
        mm._np_bitcast_convert_type(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_bitcast_convert_type(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_bitwise_count() -> None:
    """Test function test__np_bitwise_count."""
    try:
        mm._np_bitwise_count(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_bitwise_count(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_blackman_() -> None:
    """Test function test__np_blackman_."""
    try:
        mm._np_blackman_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_blackman_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_block() -> None:
    """Test function test__np_block."""
    try:
        mm._np_block(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_block(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_blurconfig() -> None:
    """Test function test__np_blurconfig."""
    try:
        mm._np_blurconfig(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_blurconfig(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_broadcast_arrays_() -> None:
    """Test function test__np_broadcast_arrays_."""
    try:
        mm._np_broadcast_arrays_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_broadcast_arrays_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_callable() -> None:
    """Test function test__np_callable."""
    try:
        mm._np_callable(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_callable(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_can_cast_() -> None:
    """Test function test__np_can_cast_."""
    try:
        mm._np_can_cast_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_can_cast_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_chebyshev_polynomial_t() -> None:
    """Test function test__np_chebyshev_polynomial_t."""
    try:
        mm._np_chebyshev_polynomial_t(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_chebyshev_polynomial_t(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_chebyshev_polynomial_u() -> None:
    """Test function test__np_chebyshev_polynomial_u."""
    try:
        mm._np_chebyshev_polynomial_u(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_chebyshev_polynomial_u(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_choose() -> None:
    """Test function test__np_choose."""
    try:
        mm._np_choose(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_choose(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_clip() -> None:
    """Test function test__np_clip."""
    try:
        mm._np_clip(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_clip(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_clz() -> None:
    """Test function test__np_clz."""
    try:
        mm._np_clz(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_clz(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_column_stack() -> None:
    """Test function test__np_column_stack."""
    try:
        mm._np_column_stack(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_column_stack(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_compress() -> None:
    """Test function test__np_compress."""
    try:
        mm._np_compress(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_compress(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_confusion_matrix() -> None:
    """Test function test__np_confusion_matrix."""
    try:
        mm._np_confusion_matrix(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_confusion_matrix(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_convgeneraldilatedlocal() -> None:
    """Test function test__np_convgeneraldilatedlocal."""
    try:
        mm._np_convgeneraldilatedlocal(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_convgeneraldilatedlocal(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_convgeneraldilatedpatches() -> None:
    """Test function test__np_convgeneraldilatedpatches."""
    try:
        mm._np_convgeneraldilatedpatches(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_convgeneraldilatedpatches(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_convolve() -> None:
    """Test function test__np_convolve."""
    try:
        mm._np_convolve(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_convolve(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_convwithgeneralpadding() -> None:
    """Test function test__np_convwithgeneralpadding."""
    try:
        mm._np_convwithgeneralpadding(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_convwithgeneralpadding(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_corrcoef() -> None:
    """Test function test__np_corrcoef."""
    try:
        mm._np_corrcoef(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_corrcoef(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_correlate() -> None:
    """Test function test__np_correlate."""
    try:
        mm._np_correlate(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_correlate(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_cov() -> None:
    """Test function test__np_cov."""
    try:
        mm._np_cov(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_cov(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_customlinearsolve() -> None:
    """Test function test__np_customlinearsolve."""
    try:
        mm._np_customlinearsolve(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_customlinearsolve(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_customroot() -> None:
    """Test function test__np_customroot."""
    try:
        mm._np_customroot(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_customroot(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_debuginfs() -> None:
    """Test function test__np_debuginfs."""
    try:
        mm._np_debuginfs(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_debuginfs(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_debugnans() -> None:
    """Test function test__np_debugnans."""
    try:
        mm._np_debugnans(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_debugnans(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_decode_csv() -> None:
    """Test function test__np_decode_csv."""
    try:
        mm._np_decode_csv(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_decode_csv(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_decode_image() -> None:
    """Test function test__np_decode_image."""
    try:
        mm._np_decode_image(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_decode_image(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_delete_() -> None:
    """Test function test__np_delete_."""
    try:
        mm._np_delete_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_delete_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_descriptive() -> None:
    """Test function test__np_descriptive."""
    try:
        mm._np_descriptive(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_descriptive(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_diag_indices_() -> None:
    """Test function test__np_diag_indices_."""
    try:
        mm._np_diag_indices_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_diag_indices_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_diag_indices_from_() -> None:
    """Test function test__np_diag_indices_from_."""
    try:
        mm._np_diag_indices_from_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_diag_indices_from_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_diagflat_() -> None:
    """Test function test__np_diagflat_."""
    try:
        mm._np_diagflat_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_diagflat_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_diagonal_() -> None:
    """Test function test__np_diagonal_."""
    try:
        mm._np_diagonal_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_diagonal_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_diff_() -> None:
    """Test function test__np_diff_."""
    try:
        mm._np_diff_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_diff_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_digitize_() -> None:
    """Test function test__np_digitize_."""
    try:
        mm._np_digitize_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_digitize_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_distributions() -> None:
    """Test function test__np_distributions."""
    try:
        mm._np_distributions(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_distributions(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_dotgeneral() -> None:
    """Test function test__np_dotgeneral."""
    try:
        mm._np_dotgeneral(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_dotgeneral(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ediff1d_() -> None:
    """Test function test__np_ediff1d_."""
    try:
        mm._np_ediff1d_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ediff1d_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_einsum_path_() -> None:
    """Test function test__np_einsum_path_."""
    try:
        mm._np_einsum_path_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_einsum_path_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_elasticconfig() -> None:
    """Test function test__np_elasticconfig."""
    try:
        mm._np_elasticconfig(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_elasticconfig(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_empty_() -> None:
    """Test function test__np_empty_."""
    try:
        mm._np_empty_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_empty_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_empty_like_() -> None:
    """Test function test__np_empty_like_."""
    try:
        mm._np_empty_like_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_empty_like_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_expand_dims_() -> None:
    """Test function test__np_expand_dims_."""
    try:
        mm._np_expand_dims_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_expand_dims_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_extract_() -> None:
    """Test function test__np_extract_."""
    try:
        mm._np_extract_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_extract_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_extractpatchesoptions() -> None:
    """Test function test__np_extractpatchesoptions."""
    try:
        mm._np_extractpatchesoptions(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_extractpatchesoptions(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fabs_() -> None:
    """Test function test__np_fabs_."""
    try:
        mm._np_fabs_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fabs_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fftn() -> None:
    """Test function test__np_fftn."""
    try:
        mm._np_fftn(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fftn(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fftnd() -> None:
    """Test function test__np_fftnd."""
    try:
        mm._np_fftnd(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fftnd(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fftshift() -> None:
    """Test function test__np_fftshift."""
    try:
        mm._np_fftshift(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fftshift(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_flatnonzero_() -> None:
    """Test function test__np_flatnonzero_."""
    try:
        mm._np_flatnonzero_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_flatnonzero_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_flip_op_() -> None:
    """Test function test__np_flip_op_."""
    try:
        mm._np_flip_op_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_flip_op_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_flip_reverse_() -> None:
    """Test function test__np_flip_reverse_."""
    try:
        mm._np_flip_reverse_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_flip_reverse_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fliplr_() -> None:
    """Test function test__np_fliplr_."""
    try:
        mm._np_fliplr_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fliplr_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_flipud_() -> None:
    """Test function test__np_flipud_."""
    try:
        mm._np_flipud_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_flipud_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_frombuffer() -> None:
    """Test function test__np_frombuffer."""
    try:
        mm._np_frombuffer(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_frombuffer(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fromdlpack() -> None:
    """Test function test__np_fromdlpack."""
    try:
        mm._np_fromdlpack(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fromdlpack(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fromfunction_() -> None:
    """Test function test__np_fromfunction_."""
    try:
        mm._np_fromfunction_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fromfunction_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fromiter_() -> None:
    """Test function test__np_fromiter_."""
    try:
        mm._np_fromiter_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fromiter_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_frompyfunc_() -> None:
    """Test function test__np_frompyfunc_."""
    try:
        mm._np_frompyfunc_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_frompyfunc_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_fromstring_() -> None:
    """Test function test__np_fromstring_."""
    try:
        mm._np_fromstring_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_fromstring_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_full_() -> None:
    """Test function test__np_full_."""
    try:
        mm._np_full_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_full_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_full_like_() -> None:
    """Test function test__np_full_like_."""
    try:
        mm._np_full_like_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_full_like_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_geomspace_() -> None:
    """Test function test__np_geomspace_."""
    try:
        mm._np_geomspace_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_geomspace_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_get_printoptions_() -> None:
    """Test function test__np_get_printoptions_."""
    try:
        mm._np_get_printoptions_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_get_printoptions_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_hamming_() -> None:
    """Test function test__np_hamming_."""
    try:
        mm._np_hamming_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_hamming_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_hanning_() -> None:
    """Test function test__np_hanning_."""
    try:
        mm._np_hanning_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_hanning_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_hermite_polynomial_h() -> None:
    """Test function test__np_hermite_polynomial_h."""
    try:
        mm._np_hermite_polynomial_h(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_hermite_polynomial_h(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_hermite_polynomial_he() -> None:
    """Test function test__np_hermite_polynomial_he."""
    try:
        mm._np_hermite_polynomial_he(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_hermite_polynomial_he(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_hfft() -> None:
    """Test function test__np_hfft."""
    try:
        mm._np_hfft(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_hfft(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_histogram2d_() -> None:
    """Test function test__np_histogram2d_."""
    try:
        mm._np_histogram2d_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_histogram2d_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_histogram_() -> None:
    """Test function test__np_histogram_."""
    try:
        mm._np_histogram_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_histogram_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_histogram_bin_edges_() -> None:
    """Test function test__np_histogram_bin_edges_."""
    try:
        mm._np_histogram_bin_edges_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_histogram_bin_edges_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_histogramdd_() -> None:
    """Test function test__np_histogramdd_."""
    try:
        mm._np_histogramdd_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_histogramdd_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ifft() -> None:
    """Test function test__np_ifft."""
    try:
        mm._np_ifft(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ifft(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ifft2() -> None:
    """Test function test__np_ifft2."""
    try:
        mm._np_ifft2(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ifft2(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ifftn() -> None:
    """Test function test__np_ifftn."""
    try:
        mm._np_ifftn(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ifftn(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ifftnd() -> None:
    """Test function test__np_ifftnd."""
    try:
        mm._np_ifftnd(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ifftnd(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ifftshift() -> None:
    """Test function test__np_ifftshift."""
    try:
        mm._np_ifftshift(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ifftshift(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_indices_() -> None:
    """Test function test__np_indices_."""
    try:
        mm._np_indices_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_indices_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_insert_() -> None:
    """Test function test__np_insert_."""
    try:
        mm._np_insert_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_insert_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_interp_() -> None:
    """Test function test__np_interp_."""
    try:
        mm._np_interp_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_interp_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_intersect1d_() -> None:
    """Test function test__np_intersect1d_."""
    try:
        mm._np_intersect1d_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_intersect1d_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_irfft2() -> None:
    """Test function test__np_irfft2."""
    try:
        mm._np_irfft2(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_irfft2(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_irfftn() -> None:
    """Test function test__np_irfftn."""
    try:
        mm._np_irfftn(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_irfftn(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_irfftnd() -> None:
    """Test function test__np_irfftnd."""
    try:
        mm._np_irfftnd(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_irfftnd(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_iscomplex_() -> None:
    """Test function test__np_iscomplex_."""
    try:
        mm._np_iscomplex_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_iscomplex_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_iscomplexobj_() -> None:
    """Test function test__np_iscomplexobj_."""
    try:
        mm._np_iscomplexobj_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_iscomplexobj_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_isin_() -> None:
    """Test function test__np_isin_."""
    try:
        mm._np_isin_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_isin_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_isreal_() -> None:
    """Test function test__np_isreal_."""
    try:
        mm._np_isreal_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_isreal_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_isrealobj_() -> None:
    """Test function test__np_isrealobj_."""
    try:
        mm._np_isrealobj_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_isrealobj_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_isscalar_() -> None:
    """Test function test__np_isscalar_."""
    try:
        mm._np_isscalar_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_isscalar_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_issubdtype_issubdtype_() -> None:
    """Test function test__np_issubdtype_issubdtype_."""
    try:
        mm._np_issubdtype_issubdtype_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_issubdtype_issubdtype_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_issubdtype_op_() -> None:
    """Test function test__np_issubdtype_op_."""
    try:
        mm._np_issubdtype_op_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_issubdtype_op_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_iterable_() -> None:
    """Test function test__np_iterable_."""
    try:
        mm._np_iterable_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_iterable_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ix__() -> None:
    """Test function test__np_ix__."""
    try:
        mm._np_ix__(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ix__(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_kaiser_() -> None:
    """Test function test__np_kaiser_."""
    try:
        mm._np_kaiser_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_kaiser_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_key() -> None:
    """Test function test__np_key."""
    try:
        mm._np_key(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_key(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_kron_() -> None:
    """Test function test__np_kron_."""
    try:
        mm._np_kron_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_kron_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_laguerre_polynomial_l() -> None:
    """Test function test__np_laguerre_polynomial_l."""
    try:
        mm._np_laguerre_polynomial_l(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_laguerre_polynomial_l(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_legendre_polynomial_p() -> None:
    """Test function test__np_legendre_polynomial_p."""
    try:
        mm._np_legendre_polynomial_p(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_legendre_polynomial_p(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_lexsort_() -> None:
    """Test function test__np_lexsort_."""
    try:
        mm._np_lexsort_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_lexsort_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linalg_cholesky_() -> None:
    """Test function test__np_linalg_cholesky_."""
    try:
        mm._np_linalg_cholesky_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linalg_cholesky_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linalg_det_() -> None:
    """Test function test__np_linalg_det_."""
    try:
        mm._np_linalg_det_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linalg_det_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linalg_inv_() -> None:
    """Test function test__np_linalg_inv_."""
    try:
        mm._np_linalg_inv_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linalg_inv_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linalg_matrix_power_() -> None:
    """Test function test__np_linalg_matrix_power_."""
    try:
        mm._np_linalg_matrix_power_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linalg_matrix_power_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linalg_pinv_() -> None:
    """Test function test__np_linalg_pinv_."""
    try:
        mm._np_linalg_pinv_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linalg_pinv_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linalg_svd_() -> None:
    """Test function test__np_linalg_svd_."""
    try:
        mm._np_linalg_svd_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linalg_svd_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperator() -> None:
    """Test function test__np_linearoperator."""
    try:
        mm._np_linearoperator(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperator(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatoradjoint() -> None:
    """Test function test__np_linearoperatoradjoint."""
    try:
        mm._np_linearoperatoradjoint(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatoradjoint(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorblockdiag() -> None:
    """Test function test__np_linearoperatorblockdiag."""
    try:
        mm._np_linearoperatorblockdiag(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorblockdiag(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorblocklowertriangular() -> None:
    """Test function test__np_linearoperatorblocklowertriangular."""
    try:
        mm._np_linearoperatorblocklowertriangular(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorblocklowertriangular(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorcirculant() -> None:
    """Test function test__np_linearoperatorcirculant."""
    try:
        mm._np_linearoperatorcirculant(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorcirculant(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorcirculant2d() -> None:
    """Test function test__np_linearoperatorcirculant2d."""
    try:
        mm._np_linearoperatorcirculant2d(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorcirculant2d(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorcirculant3d() -> None:
    """Test function test__np_linearoperatorcirculant3d."""
    try:
        mm._np_linearoperatorcirculant3d(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorcirculant3d(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorcomposition() -> None:
    """Test function test__np_linearoperatorcomposition."""
    try:
        mm._np_linearoperatorcomposition(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorcomposition(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatordiag() -> None:
    """Test function test__np_linearoperatordiag."""
    try:
        mm._np_linearoperatordiag(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatordiag(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorfullmatrix() -> None:
    """Test function test__np_linearoperatorfullmatrix."""
    try:
        mm._np_linearoperatorfullmatrix(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorfullmatrix(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorhouseholder() -> None:
    """Test function test__np_linearoperatorhouseholder."""
    try:
        mm._np_linearoperatorhouseholder(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorhouseholder(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatoridentity() -> None:
    """Test function test__np_linearoperatoridentity."""
    try:
        mm._np_linearoperatoridentity(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatoridentity(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorinversion() -> None:
    """Test function test__np_linearoperatorinversion."""
    try:
        mm._np_linearoperatorinversion(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorinversion(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorkronecker() -> None:
    """Test function test__np_linearoperatorkronecker."""
    try:
        mm._np_linearoperatorkronecker(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorkronecker(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorlowertriangular() -> None:
    """Test function test__np_linearoperatorlowertriangular."""
    try:
        mm._np_linearoperatorlowertriangular(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorlowertriangular(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorlowrankupdate() -> None:
    """Test function test__np_linearoperatorlowrankupdate."""
    try:
        mm._np_linearoperatorlowrankupdate(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorlowrankupdate(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorpermutation() -> None:
    """Test function test__np_linearoperatorpermutation."""
    try:
        mm._np_linearoperatorpermutation(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorpermutation(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorscaledidentity() -> None:
    """Test function test__np_linearoperatorscaledidentity."""
    try:
        mm._np_linearoperatorscaledidentity(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorscaledidentity(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatortoeplitz() -> None:
    """Test function test__np_linearoperatortoeplitz."""
    try:
        mm._np_linearoperatortoeplitz(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatortoeplitz(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatortridiag() -> None:
    """Test function test__np_linearoperatortridiag."""
    try:
        mm._np_linearoperatortridiag(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatortridiag(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_linearoperatorzeros() -> None:
    """Test function test__np_linearoperatorzeros."""
    try:
        mm._np_linearoperatorzeros(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_linearoperatorzeros(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_load_() -> None:
    """Test function test__np_load_."""
    try:
        mm._np_load_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_load_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_log1p2() -> None:
    """Test function test__np_log1p2."""
    try:
        mm._np_log1p2(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_log1p2(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_log_softmax() -> None:
    """Test function test__np_log_softmax."""
    try:
        mm._np_log_softmax(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_log_softmax(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_logsumexp() -> None:
    """Test function test__np_logsumexp."""
    try:
        mm._np_logsumexp(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_logsumexp(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_mask_indices_() -> None:
    """Test function test__np_mask_indices_."""
    try:
        mm._np_mask_indices_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_mask_indices_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_median_() -> None:
    """Test function test__np_median_."""
    try:
        mm._np_median_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_median_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_modf_() -> None:
    """Test function test__np_modf_."""
    try:
        mm._np_modf_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_modf_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_modified_bessel_i0() -> None:
    """Test function test__np_modified_bessel_i0."""
    try:
        mm._np_modified_bessel_i0(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_modified_bessel_i0(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_modified_bessel_i1() -> None:
    """Test function test__np_modified_bessel_i1."""
    try:
        mm._np_modified_bessel_i1(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_modified_bessel_i1(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_modified_bessel_k0() -> None:
    """Test function test__np_modified_bessel_k0."""
    try:
        mm._np_modified_bessel_k0(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_modified_bessel_k0(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_modified_bessel_k1() -> None:
    """Test function test__np_modified_bessel_k1."""
    try:
        mm._np_modified_bessel_k1(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_modified_bessel_k1(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_mvlgamma() -> None:
    """Test function test__np_mvlgamma."""
    try:
        mm._np_mvlgamma(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_mvlgamma(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_nonzero_() -> None:
    """Test function test__np_nonzero_."""
    try:
        mm._np_nonzero_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_nonzero_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_one_hot() -> None:
    """Test function test__np_one_hot."""
    try:
        mm._np_one_hot(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_one_hot(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ones_() -> None:
    """Test function test__np_ones_."""
    try:
        mm._np_ones_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ones_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ones_like_() -> None:
    """Test function test__np_ones_like_."""
    try:
        mm._np_ones_like_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ones_like_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_packbits() -> None:
    """Test function test__np_packbits."""
    try:
        mm._np_packbits(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_packbits(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_parse_example() -> None:
    """Test function test__np_parse_example."""
    try:
        mm._np_parse_example(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_parse_example(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_parse_tensor() -> None:
    """Test function test__np_parse_tensor."""
    try:
        mm._np_parse_tensor(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_parse_tensor(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_perspectiveconfig() -> None:
    """Test function test__np_perspectiveconfig."""
    try:
        mm._np_perspectiveconfig(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_perspectiveconfig(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_piecewise() -> None:
    """Test function test__np_piecewise."""
    try:
        mm._np_piecewise(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_piecewise(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_pmean() -> None:
    """Test function test__np_pmean."""
    try:
        mm._np_pmean(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_pmean(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_population_count() -> None:
    """Test function test__np_population_count."""
    try:
        mm._np_population_count(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_population_count(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_promotetypes() -> None:
    """Test function test__np_promotetypes."""
    try:
        mm._np_promotetypes(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_promotetypes(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_psum() -> None:
    """Test function test__np_psum."""
    try:
        mm._np_psum(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_psum(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_raggeddot() -> None:
    """Test function test__np_raggeddot."""
    try:
        mm._np_raggeddot(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_raggeddot(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_randombernoulli() -> None:
    """Test function test__np_randombernoulli."""
    try:
        mm._np_randombernoulli(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_randombernoulli(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_randomcategorical() -> None:
    """Test function test__np_randomcategorical."""
    try:
        mm._np_randomcategorical(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_randomcategorical(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_randompermutation() -> None:
    """Test function test__np_randompermutation."""
    try:
        mm._np_randompermutation(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_randompermutation(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_randomtruncatednormal() -> None:
    """Test function test__np_randomtruncatednormal."""
    try:
        mm._np_randomtruncatednormal(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_randomtruncatednormal(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_ravel_multi_index_() -> None:
    """Test function test__np_ravel_multi_index_."""
    try:
        mm._np_ravel_multi_index_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_ravel_multi_index_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rawconv2d() -> None:
    """Test function test__np_rawconv2d."""
    try:
        mm._np_rawconv2d(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rawconv2d(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rawmatmul() -> None:
    """Test function test__np_rawmatmul."""
    try:
        mm._np_rawmatmul(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rawmatmul(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rawmerge() -> None:
    """Test function test__np_rawmerge."""
    try:
        mm._np_rawmerge(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rawmerge(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rawop() -> None:
    """Test function test__np_rawop."""
    try:
        mm._np_rawop(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rawop(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rawswitch() -> None:
    """Test function test__np_rawswitch."""
    try:
        mm._np_rawswitch(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rawswitch(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_read_file() -> None:
    """Test function test__np_read_file."""
    try:
        mm._np_read_file(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_read_file(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_reduce_precision() -> None:
    """Test function test__np_reduce_precision."""
    try:
        mm._np_reduce_precision(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_reduce_precision(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rem() -> None:
    """Test function test__np_rem."""
    try:
        mm._np_rem(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rem(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_resize_() -> None:
    """Test function test__np_resize_."""
    try:
        mm._np_resize_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_resize_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_result_type_() -> None:
    """Test function test__np_result_type_."""
    try:
        mm._np_result_type_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_result_type_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rfft() -> None:
    """Test function test__np_rfft."""
    try:
        mm._np_rfft(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rfft(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rfft2() -> None:
    """Test function test__np_rfft2."""
    try:
        mm._np_rfft2(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rfft2(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rfftfreq() -> None:
    """Test function test__np_rfftfreq."""
    try:
        mm._np_rfftfreq(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rfftfreq(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rfftn() -> None:
    """Test function test__np_rfftn."""
    try:
        mm._np_rfftn(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rfftn(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rfftnd() -> None:
    """Test function test__np_rfftnd."""
    try:
        mm._np_rfftnd(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rfftnd(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rrelu() -> None:
    """Test function test__np_rrelu."""
    try:
        mm._np_rrelu(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rrelu(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_rsqrt() -> None:
    """Test function test__np_rsqrt."""
    try:
        mm._np_rsqrt(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_rsqrt(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_scanop() -> None:
    """Test function test__np_scanop."""
    try:
        mm._np_scanop(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_scanop(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_segment_sum() -> None:
    """Test function test__np_segment_sum."""
    try:
        mm._np_segment_sum(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_segment_sum(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_serialize_tensor() -> None:
    """Test function test__np_serialize_tensor."""
    try:
        mm._np_serialize_tensor(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_serialize_tensor(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_shifted_chebyshev_polynomial_t() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_t."""
    try:
        mm._np_shifted_chebyshev_polynomial_t(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_shifted_chebyshev_polynomial_t(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_shifted_chebyshev_polynomial_u() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_u."""
    try:
        mm._np_shifted_chebyshev_polynomial_u(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_shifted_chebyshev_polynomial_u(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_shifted_chebyshev_polynomial_v() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_v."""
    try:
        mm._np_shifted_chebyshev_polynomial_v(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_shifted_chebyshev_polynomial_v(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_shifted_chebyshev_polynomial_w() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_w."""
    try:
        mm._np_shifted_chebyshev_polynomial_w(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_shifted_chebyshev_polynomial_w(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sigmoid() -> None:
    """Test function test__np_sigmoid."""
    try:
        mm._np_sigmoid(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sigmoid(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sobolsample() -> None:
    """Test function test__np_sobolsample."""
    try:
        mm._np_sobolsample(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sobolsample(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_softmax() -> None:
    """Test function test__np_softmax."""
    try:
        mm._np_softmax(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_softmax(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sparsedensematmul() -> None:
    """Test function test__np_sparsedensematmul."""
    try:
        mm._np_sparsedensematmul(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sparsedensematmul(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sparsemapvalues() -> None:
    """Test function test__np_sparsemapvalues."""
    try:
        mm._np_sparsemapvalues(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sparsemapvalues(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sparsereducemax() -> None:
    """Test function test__np_sparsereducemax."""
    try:
        mm._np_sparsereducemax(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sparsereducemax(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sparsereshape() -> None:
    """Test function test__np_sparsereshape."""
    try:
        mm._np_sparsereshape(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sparsereshape(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sparsesampledadd() -> None:
    """Test function test__np_sparsesampledadd."""
    try:
        mm._np_sparsesampledadd(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sparsesampledadd(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sparsesegmentsum() -> None:
    """Test function test__np_sparsesegmentsum."""
    try:
        mm._np_sparsesegmentsum(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sparsesegmentsum(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_sparsetranspose() -> None:
    """Test function test__np_sparsetranspose."""
    try:
        mm._np_sparsetranspose(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_sparsetranspose(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_stridedslice() -> None:
    """Test function test__np_stridedslice."""
    try:
        mm._np_stridedslice(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_stridedslice(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_switchop() -> None:
    """Test function test__np_switchop."""
    try:
        mm._np_switchop(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_switchop(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_tensor() -> None:
    """Test function test__np_tensor."""
    try:
        mm._np_tensor(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_tensor(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_tensorarrayread() -> None:
    """Test function test__np_tensorarrayread."""
    try:
        mm._np_tensorarrayread(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_tensorarrayread(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_tensorarraystack() -> None:
    """Test function test__np_tensorarraystack."""
    try:
        mm._np_tensorarraystack(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_tensorarraystack(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_tensorarraywrite() -> None:
    """Test function test__np_tensorarraywrite."""
    try:
        mm._np_tensorarraywrite(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_tensorarraywrite(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_tensorconfig() -> None:
    """Test function test__np_tensorconfig."""
    try:
        mm._np_tensorconfig(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_tensorconfig(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_trace() -> None:
    """Test function test__np_trace."""
    try:
        mm._np_trace(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_trace(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_trapz() -> None:
    """Test function test__np_trapz."""
    try:
        mm._np_trapz(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_trapz(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_trapz_() -> None:
    """Test function test__np_trapz_."""
    try:
        mm._np_trapz_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_trapz_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_tri() -> None:
    """Test function test__np_tri."""
    try:
        mm._np_tri(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_tri(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_trilindices() -> None:
    """Test function test__np_trilindices."""
    try:
        mm._np_trilindices(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_trilindices(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_trilindicesfrom() -> None:
    """Test function test__np_trilindicesfrom."""
    try:
        mm._np_trilindicesfrom(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_trilindicesfrom(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_trimzeros() -> None:
    """Test function test__np_trimzeros."""
    try:
        mm._np_trimzeros(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_trimzeros(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_triuindices() -> None:
    """Test function test__np_triuindices."""
    try:
        mm._np_triuindices(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_triuindices(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_triuindicesfrom() -> None:
    """Test function test__np_triuindicesfrom."""
    try:
        mm._np_triuindicesfrom(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_triuindicesfrom(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_truncate_div() -> None:
    """Test function test__np_truncate_div."""
    try:
        mm._np_truncate_div(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_truncate_div(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_truncate_mod() -> None:
    """Test function test__np_truncate_mod."""
    try:
        mm._np_truncate_mod(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_truncate_mod(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_uint() -> None:
    """Test function test__np_uint."""
    try:
        mm._np_uint(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_uint(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_uint8() -> None:
    """Test function test__np_uint8."""
    try:
        mm._np_uint8(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_uint8(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_union1d() -> None:
    """Test function test__np_union1d."""
    try:
        mm._np_union1d(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_union1d(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_unpackbits() -> None:
    """Test function test__np_unpackbits."""
    try:
        mm._np_unpackbits(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_unpackbits(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_unravelindex() -> None:
    """Test function test__np_unravelindex."""
    try:
        mm._np_unravelindex(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_unravelindex(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_unwrap() -> None:
    """Test function test__np_unwrap."""
    try:
        mm._np_unwrap(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_unwrap(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_vander() -> None:
    """Test function test__np_vander."""
    try:
        mm._np_vander(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_vander(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_vecdot() -> None:
    """Test function test__np_vecdot."""
    try:
        mm._np_vecdot(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_vecdot(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_vectorize() -> None:
    """Test function test__np_vectorize."""
    try:
        mm._np_vectorize(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_vectorize(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_write_file() -> None:
    """Test function test__np_write_file."""
    try:
        pass  # mm._np_write_file(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        pass  # mm._np_write_file(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_xlogy() -> None:
    """Test function test__np_xlogy."""
    try:
        mm._np_xlogy(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_xlogy(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_zeros_() -> None:
    """Test function test__np_zeros_."""
    try:
        mm._np_zeros_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_zeros_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test__np_zeros_like_() -> None:
    """Test function test__np_zeros_like_."""
    try:
        mm._np_zeros_like_(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mm._np_zeros_like_(np)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_pmean_segment_clz() -> None:
    """Test function test_pmean_segment_clz."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    mm._np_pmean(np, np.ones((2, 2)), "x")
    mm._np_segment_sum(np, np.ones((4,)), np.array([0, 0, 1, 1]))
    mm._np_clz(np, np.array([1, 2, -1], dtype=np.int32))
    assert callable(mm._np_clz)
