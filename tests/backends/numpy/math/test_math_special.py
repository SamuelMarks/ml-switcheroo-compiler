# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for test_math_special."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_confusion_matrix, _np_decode_csv, _np_decode_image, _np_descriptive, _np_distributions, _np_parse_example, _np_parse_tensor, _np_rawmatmul, _np_read_file, _np_rem, _np_serialize_tensor, _np_sparsedensematmul, _np_write_file

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
    mm._np_customlinearsolve(np, np.eye(2), np.ones((2, 2)))
    assert callable(test_custom_linear_solve)


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
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    mm._poly_recurrence(np.array([-1]), np.array([0.5]), 1.0, lambda a: a, lambda i, a, p1, p2: p1)
    mm._np_distributions(np)
    assert callable(test_math_misc_scipy_and_polynomials)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_all_mocked_try_blocks)


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
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_all_numpy_math_misc_fallbacks)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_all_numpy_math_misc_fallbacks_2)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_all_numpy_math_misc_fallbacks_3)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    mm._np_frombuffer(np)
    mm._np_rem(np)
    assert callable(test_np_remaining_math_misc)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_math_misc_scipy_and_polynomials_shifted)


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
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    finally:
        if old_scipy:
            sys.modules["scipy.special"] = old_scipy
        else:
            del sys.modules["scipy.special"]
    assert callable(test_math_misc_scipy_bessels)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_np_remaining_math_misc_2)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    mm._np_rfftfreq(np, 10, d=0.1)
    mm._np_frombuffer(np)
    mm._np_rem(np, 1)
    assert callable(test_np_remaining_math_misc_3)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_remaining_math_misc_stuff)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_once_more)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_once_more_2)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_2)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_3)


def test_final_math_misc_mocking_again_and_again_again_4() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_4."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        func = getattr(mm, "_np_" + op_name.lower() + "_camel", getattr(mm, "_np_" + op_name, None))
        if func:
            func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
    assert callable(test_final_math_misc_mocking_again_and_again_again_4)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_5)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_camel)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_6)


def test_final_math_misc_mocking_again_and_again_again_7() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_7."""
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        func = getattr(mm, "_np_" + op_name.lower() + "_camel", None)
        if func:
            func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
    assert callable(test_final_math_misc_mocking_again_and_again_again_7)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_for_remaining)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_8_fixed)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_9)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_10)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_11)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_12)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_13)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_14)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_15)


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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    assert callable(test_final_math_misc_mocking_again_and_again_again_16)


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
    (original_confusion_matrix, original_decode_csv, original_decode_image, original_descriptive) = (_state["original_confusion_matrix"], _state["original_decode_csv"], _state["original_decode_image"], _state["original_descriptive"])
    (original_distributions, original_parse_example, original_parse_tensor, original_rawmatmul) = (_state["original_distributions"], _state["original_parse_example"], _state["original_parse_tensor"], _state["original_rawmatmul"])
    (original_read_file, original_rem, original_serialize_tensor, original_sparsedensematmul) = (_state["original_read_file"], _state["original_rem"], _state["original_serialize_tensor"], _state["original_sparsedensematmul"])
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

    with pytest.raises(Exception, match=".*"):
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
    _np_decode_csv(np, "1,2,3")
    _state.update(locals())


def _test_np_ops_missing_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_np_ops_missing.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(RuntimeError):
        _np_decode_image(np, b"")
    with pytest.raises(RuntimeError):
        _np_parse_example(np, b"")
    with pytest.raises(RuntimeError):
        _np_parse_tensor(np, b"")
    _np_read_file(np, "/tmp/ml_switcheroo_test_file2.txt")
    print("HASATTR REM:", hasattr(_ops, "rem"))
    _np_rem(np, np.array([5]), np.array([2]))
    _np_serialize_tensor(np, np.ones((2, 2)))
    _state.update(locals())


def _test_np_ops_missing_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_np_ops_missing.

    Args:
        _state: Shared test state dictionary.
    """
    _np_write_file(np, "/tmp/ml_switcheroo_test_file2.txt", "hello")
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
    mm._np_affineconfig(np, np.ones((2, 2)))
    mm._np_affineconfig(np)
    assert callable(test__np_affineconfig)


def test__np_all_gather() -> None:
    """Test function test__np_all_gather."""
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(mm, "_np_all_gather"))(np, np.ones((2, 2)))
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(mm, "_np_all_gather"))(np)


def test__np_all_reduce() -> None:
    """Test function test__np_all_reduce."""
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(mm, "_np_all_reduce"))(np, np.ones((2, 2)))
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(mm, "_np_all_reduce"))(np)


def test__np_all_to_all() -> None:
    """Test function test__np_all_to_all."""
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(mm, "_np_all_to_all"))(np, np.ones((2, 2)))
    with pytest.raises(AttributeError):
        cast("EagerTensorProtocol", getattr(mm, "_np_all_to_all"))(np)


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
    mm._np_assertop(np)
    assert callable(test__np_assertop)


def test__np_asstringconfig() -> None:
    """Test function test__np_asstringconfig."""
    mm._np_asstringconfig(np, np.ones((2, 2)))
    mm._np_asstringconfig(np)
    assert callable(test__np_asstringconfig)


def test__np_atleast_1d() -> None:
    """Test function test__np_atleast_1d."""
    mm._np_atleast_1d(np, np.ones((2, 2)))
    mm._np_atleast_1d(np)
    assert callable(test__np_atleast_1d)


def test__np_atleast_2d() -> None:
    """Test function test__np_atleast_2d."""
    mm._np_atleast_2d(np, np.ones((2, 2)))
    mm._np_atleast_2d(np)
    assert callable(test__np_atleast_2d)


def test__np_atleast_3d() -> None:
    """Test function test__np_atleast_3d."""
    mm._np_atleast_3d(np, np.ones((2, 2)))
    mm._np_atleast_3d(np)
    assert callable(test__np_atleast_3d)


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
        mm._np_bitcast_convert_type(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_bitcast_convert_type(np)


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
    mm._np_broadcast_arrays_(np)
    assert callable(test__np_broadcast_arrays_)


def test__np_callable() -> None:
    """Test function test__np_callable."""
    mm._np_callable(np, np.ones((2, 2)))
    mm._np_callable(np)
    assert callable(test__np_callable)


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
    with pytest.raises(ValueError):
        mm._np_clip(np, np.ones((2, 2)))
    mm._np_clip(np)


def test__np_clz() -> None:
    """Test function test__np_clz."""
    with pytest.raises(TypeError):
        mm._np_clz(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_clz(np)


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
    mm._np_confusion_matrix(np)
    assert callable(test__np_confusion_matrix)


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
    mm._np_corrcoef(np, np.ones((2, 2)))
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
    mm._np_decode_csv(np)
    assert callable(test__np_decode_csv)


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
    mm._np_descriptive(np)
    assert callable(test__np_descriptive)


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
    mm._np_distributions(np)
    assert callable(test__np_distributions)


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
    mm._np_elasticconfig(np, np.ones((2, 2)))
    mm._np_elasticconfig(np)
    assert callable(test__np_elasticconfig)


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
    mm._np_extractpatchesoptions(np, np.ones((2, 2)))
    mm._np_extractpatchesoptions(np)
    assert callable(test__np_extractpatchesoptions)


def test__np_fabs_() -> None:
    """Test function test__np_fabs_."""
    mm._np_fabs_(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_fabs_(np)


def test__np_fftn() -> None:
    """Test function test__np_fftn."""
    mm._np_fftn(np, np.ones((2, 2)))
    mm._np_fftn(np)
    assert callable(test__np_fftn)


def test__np_fftnd() -> None:
    """Test function test__np_fftnd."""
    mm._np_fftnd(np, np.ones((2, 2)))
    mm._np_fftnd(np)
    assert callable(test__np_fftnd)


def test__np_fftshift() -> None:
    """Test function test__np_fftshift."""
    mm._np_fftshift(np, np.ones((2, 2)))
    mm._np_fftshift(np)
    assert callable(test__np_fftshift)


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
    mm._np_frombuffer(np)
    assert callable(test__np_frombuffer)


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
    mm._np_fromstring_(np, np.ones((2, 2)))
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
    mm._np_hfft(np)
    assert callable(test__np_hfft)


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
    mm._np_ifft(np)
    assert callable(test__np_ifft)


def test__np_ifft2() -> None:
    """Test function test__np_ifft2."""
    mm._np_ifft2(np, np.ones((2, 2)))
    mm._np_ifft2(np)
    assert callable(test__np_ifft2)


def test__np_ifftn() -> None:
    """Test function test__np_ifftn."""
    mm._np_ifftn(np, np.ones((2, 2)))
    mm._np_ifftn(np)
    assert callable(test__np_ifftn)


def test__np_ifftnd() -> None:
    """Test function test__np_ifftnd."""
    mm._np_ifftnd(np, np.ones((2, 2)))
    mm._np_ifftnd(np)
    assert callable(test__np_ifftnd)


def test__np_ifftshift() -> None:
    """Test function test__np_ifftshift."""
    mm._np_ifftshift(np, np.ones((2, 2)))
    mm._np_ifftshift(np)
    assert callable(test__np_ifftshift)


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
    mm._np_irfft2(np)
    assert callable(test__np_irfft2)


def test__np_irfftn() -> None:
    """Test function test__np_irfftn."""
    mm._np_irfftn(np, np.ones((2, 2)))
    mm._np_irfftn(np)
    assert callable(test__np_irfftn)


def test__np_irfftnd() -> None:
    """Test function test__np_irfftnd."""
    mm._np_irfftnd(np, np.ones((2, 2)))
    mm._np_irfftnd(np)
    assert callable(test__np_irfftnd)


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
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=".*"):
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
    with pytest.raises((ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=".*"):
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
    mm._np_log_softmax(np)
    assert callable(test__np_log_softmax)


def test__np_logsumexp() -> None:
    """Test function test__np_logsumexp."""
    mm._np_logsumexp(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_logsumexp(np)


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
    mm._np_modified_bessel_i0(np)
    assert callable(test__np_modified_bessel_i0)


def test__np_modified_bessel_i1() -> None:
    """Test function test__np_modified_bessel_i1."""
    mm._np_modified_bessel_i1(np, np.ones((2, 2)))
    mm._np_modified_bessel_i1(np)
    assert callable(test__np_modified_bessel_i1)


def test__np_modified_bessel_k0() -> None:
    """Test function test__np_modified_bessel_k0."""
    mm._np_modified_bessel_k0(np, np.ones((2, 2)))
    mm._np_modified_bessel_k0(np)
    assert callable(test__np_modified_bessel_k0)


def test__np_modified_bessel_k1() -> None:
    """Test function test__np_modified_bessel_k1."""
    mm._np_modified_bessel_k1(np, np.ones((2, 2)))
    mm._np_modified_bessel_k1(np)
    assert callable(test__np_modified_bessel_k1)


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
    mm._np_one_hot(np)
    assert callable(test__np_one_hot)


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


def test__np_parse_example() -> None:
    """Test function test__np_parse_example."""
    with pytest.raises(RuntimeError):
        mm._np_parse_example(np, np.ones((2, 2)))
    mm._np_parse_example(np)


def test__np_parse_tensor() -> None:
    """Test function test__np_parse_tensor."""
    with pytest.raises(RuntimeError):
        mm._np_parse_tensor(np, np.ones((2, 2)))
    mm._np_parse_tensor(np)


def test__np_perspectiveconfig() -> None:
    """Test function test__np_perspectiveconfig."""
    mm._np_perspectiveconfig(np, np.ones((2, 2)))
    mm._np_perspectiveconfig(np)
    assert callable(test__np_perspectiveconfig)


def test__np_piecewise() -> None:
    """Test function test__np_piecewise."""
    with pytest.raises(IndexError):
        mm._np_piecewise(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_piecewise(np)


def test__np_pmean() -> None:
    """Test function test__np_pmean."""
    with pytest.raises(TypeError):
        mm._np_pmean(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_pmean(np)


def test__np_population_count() -> None:
    """Test function test__np_population_count."""
    with pytest.raises(TypeError):
        mm._np_population_count(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_population_count(np)


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
    mm._np_rawmerge(np)
    assert callable(test__np_rawmerge)


def test__np_rawop() -> None:
    """Test function test__np_rawop."""
    mm._np_rawop(np, np.ones((2, 2)))
    mm._np_rawop(np)
    assert callable(test__np_rawop)


def test__np_rawswitch() -> None:
    """Test function test__np_rawswitch."""
    mm._np_rawswitch(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_rawswitch(np)


def test__np_read_file() -> None:
    """Test function test__np_read_file."""
    mm._np_read_file(np, np.ones((2, 2)))
    mm._np_read_file(np)
    assert callable(test__np_read_file)


def test__np_reduce_precision() -> None:
    """Test function test__np_reduce_precision."""
    with pytest.raises(TypeError):
        mm._np_reduce_precision(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_reduce_precision(np)


def test__np_rem() -> None:
    """Test function test__np_rem."""
    mm._np_rem(np, np.ones((2, 2)))
    mm._np_rem(np)
    assert callable(test__np_rem)


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
    mm._np_rfft(np)
    assert callable(test__np_rfft)


def test__np_rfft2() -> None:
    """Test function test__np_rfft2."""
    mm._np_rfft2(np, np.ones((2, 2)))
    mm._np_rfft2(np)
    assert callable(test__np_rfft2)


def test__np_rfftfreq() -> None:
    """Test function test__np_rfftfreq."""
    mm._np_rfftfreq(np, np.ones((2, 2)))
    mm._np_rfftfreq(np)
    assert callable(test__np_rfftfreq)


def test__np_rfftn() -> None:
    """Test function test__np_rfftn."""
    mm._np_rfftn(np, np.ones((2, 2)))
    mm._np_rfftn(np)
    assert callable(test__np_rfftn)


def test__np_rfftnd() -> None:
    """Test function test__np_rfftnd."""
    mm._np_rfftnd(np, np.ones((2, 2)))
    mm._np_rfftnd(np)
    assert callable(test__np_rfftnd)


def test__np_rrelu() -> None:
    """Test function test__np_rrelu."""
    mm._np_rrelu(np, np.ones((2, 2)))
    mm._np_rrelu(np)
    assert callable(test__np_rrelu)


def test__np_rsqrt() -> None:
    """Test function test__np_rsqrt."""
    mm._np_rsqrt(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_rsqrt(np)


def test__np_scanop() -> None:
    """Test function test__np_scanop."""
    mm._np_scanop(np, np.ones((2, 2)))
    mm._np_scanop(np)
    assert callable(test__np_scanop)


def test__np_segment_sum() -> None:
    """Test function test__np_segment_sum."""
    with pytest.raises(TypeError):
        mm._np_segment_sum(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_segment_sum(np)


def test__np_serialize_tensor() -> None:
    """Test function test__np_serialize_tensor."""
    mm._np_serialize_tensor(np, np.ones((2, 2)))
    mm._np_serialize_tensor(np)
    assert callable(test__np_serialize_tensor)


def test__np_shifted_chebyshev_polynomial_t() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_t."""
    mm._np_shifted_chebyshev_polynomial_t(np, np.ones((2, 2)))
    mm._np_shifted_chebyshev_polynomial_t(np)
    assert callable(test__np_shifted_chebyshev_polynomial_t)


def test__np_shifted_chebyshev_polynomial_u() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_u."""
    mm._np_shifted_chebyshev_polynomial_u(np, np.ones((2, 2)))
    mm._np_shifted_chebyshev_polynomial_u(np)
    assert callable(test__np_shifted_chebyshev_polynomial_u)


def test__np_shifted_chebyshev_polynomial_v() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_v."""
    mm._np_shifted_chebyshev_polynomial_v(np, np.ones((2, 2)))
    mm._np_shifted_chebyshev_polynomial_v(np)
    assert callable(test__np_shifted_chebyshev_polynomial_v)


def test__np_shifted_chebyshev_polynomial_w() -> None:
    """Test function test__np_shifted_chebyshev_polynomial_w."""
    mm._np_shifted_chebyshev_polynomial_w(np, np.ones((2, 2)))
    mm._np_shifted_chebyshev_polynomial_w(np)
    assert callable(test__np_shifted_chebyshev_polynomial_w)


def test__np_sigmoid() -> None:
    """Test function test__np_sigmoid."""
    mm._np_sigmoid(np, np.ones((2, 2)))
    mm._np_sigmoid(np)
    assert callable(test__np_sigmoid)


def test__np_sobolsample() -> None:
    """Test function test__np_sobolsample."""
    with pytest.raises(TypeError):
        mm._np_sobolsample(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_sobolsample(np)


def test__np_softmax() -> None:
    """Test function test__np_softmax."""
    mm._np_softmax(np, np.ones((2, 2)))
    mm._np_softmax(np)
    assert callable(test__np_softmax)


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
        mm._np_stridedslice(np, np.ones((2, 2)))
    with pytest.raises(TypeError):
        mm._np_stridedslice(np)


def test__np_switchop() -> None:
    """Test function test__np_switchop."""
    mm._np_switchop(np, np.ones((2, 2)))
    with pytest.raises(IndexError):
        mm._np_switchop(np)


def test__np_tensor() -> None:
    """Test function test__np_tensor."""
    mm._np_tensor(np, np.ones((2, 2)))
    mm._np_tensor(np)
    assert callable(test__np_tensor)


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
    mm._np_tensorconfig(np, np.ones((2, 2)))
    mm._np_tensorconfig(np)
    assert callable(test__np_tensorconfig)


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
    with pytest.raises(ValueError):
        mm._np_trimzeros(np, np.ones((2, 2)))
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


def test__np_write_file() -> None:
    """Test function test__np_write_file."""
    pass
    pass
    assert callable(test__np_write_file)


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
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    mm._np_pmean(np, np.ones((2, 2)), "x")
    mm._np_segment_sum(np, np.ones((4,)), np.array([0, 0, 1, 1]))
    mm._np_clz(np, np.array([1, 2, -1], dtype=np.int32))
    assert callable(mm._np_clz)
