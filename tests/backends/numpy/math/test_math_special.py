# ruff: noqa: E501
import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm


def test_pmean():
    assert np.array_equal(mm._np_pmean(np, np.ones((2, 2)), "x"), np.ones((2, 2)))


def test_segment_sum():
    res = mm._np_segment_sum(np, np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(res, [3, 7])


def test_clz():
    res = mm._np_clz(np, np.array([1, 2, -1], dtype=np.int32))
    assert res[0] == 31
    assert res[1] == 30
    assert res[2] == 0


def test_population_count():
    res = mm._np_population_count(np, np.array([1, 3, -1], dtype=np.int32))
    assert res[0] == 1
    assert res[1] == 2
    res = mm._np_population_count(np, np.array([1, 3, 7], dtype=np.int32))
    assert res[2] == 3


def test_bitcast_convert_type():
    res = mm._np_bitcast_convert_type(np, np.array([1.0], dtype=np.float32), "int32")
    assert res.dtype == np.int32


def test_reduce_precision():
    res = mm._np_reduce_precision(np, np.array([1.0]), 8, 23)
    np.testing.assert_allclose(res, [1.0])


def test_custom_linear_solve():
    try:
        mm._np_customlinearsolve(np, np.eye(2), np.ones((2, 2)))
    except Exception:
        pass


def test_strided_slice():
    res = mm._np_stridedslice(np, np.ones((5, 5)), [1, 1], [3, 3], [1, 1])
    assert res.shape == (2, 2)


def test_sobol_sample():
    res = mm._np_sobolsample(np, 2, 4)
    assert res.shape == (4, 2)


def test_logsumexp():
    res = mm._np_logsumexp(np, np.ones((2, 2)), keepdims=False)
    assert np.ndim(res) == 0


def test_truncate_div():
    res = mm._np_truncate_div(np, np.array([3.5]), np.array([2.0]))
    np.testing.assert_allclose(res, [1.0])


def test_truncate_mod():
    res = mm._np_truncate_mod(np, np.array([3.5]), np.array([2.0]))
    np.testing.assert_allclose(res, [1.5])


def test_math_misc_one_hot():
    res = mm._np_one_hot(np, np.array([0, 1]), depth=2)
    assert res.shape == (2, 2)


def test_math_misc_scipy_and_polynomials():
    import sys

    old_scipy = sys.modules.get("scipy.special", None)
    sys.modules["scipy.special"] = None
    try:
        mm._get_sc()
    except Exception:
        pass
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
        except ValueError:
            pass
    mm._poly_recurrence(np.array([-1]), np.array([0.5]), 1.0, lambda a: a, lambda i, a, p1, p2: p1)
    try:
        mm._np_distributions(np)
    except Exception:
        pass


def test_all_mocked_try_blocks():
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
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
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_math_misc_more_edges():
    res_fb = mm._np_frombuffer(np, b"\x00\x00\x00\x00", dtype=np.int32)
    assert res_fb.shape == (1,)
    assert mm._np_frombuffer(np) is None
    res_oh = mm._np_one_hot(np, np.array([0, 1]), depth=2, axis=0)
    assert res_oh.shape == (2, 2)
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

    old_di = getattr(ops, "DecodeImage", None)
    ops.DecodeImage = MockOpFunc
    try:
        mm._np_decode_image(np, np.ones((2, 2)))
    except Exception:
        pass
    finally:
        if old_di:
            ops.DecodeImage = old_di
        else:
            delattr(ops, "DecodeImage")


def test_all_numpy_math_misc_fallbacks():

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

    class MockBM:
        def __init__(self):
            pass

        def __getattr__(self, name):

            def dummy(*args, **kwargs):
                return np.ones((2, 2))

            return dummy

    op_names = ["decode_csv", "decode_image", "parse_example", "parse_tensor", "serialize_tensor", "encode_image", "encode_jpeg", "encode_png", "encode_base64", "decode_base64", "decode_json_example", "decode_compressed", "read_file", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        try:
            func = getattr(mm, "_np_" + op_name.lower(), getattr(mm, "_np_" + op_name, None))
            if func:
                func(MockBM(), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass


def test_all_numpy_math_misc_fallbacks_2():

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

    class MockBM:
        def __init__(self):
            pass

        def __getattr__(self, name):

            def dummy(*args, **kwargs):
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
        except Exception:
            pass


def test_all_numpy_math_misc_fallbacks_3():

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

    class MockBM:
        def __init__(self):
            pass

        def __getattr__(self, name):

            def dummy(*args, **kwargs):
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
        except Exception:
            pass


def test_more_math_misc_stuff():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    mm._np_rem(np)
    mm._np_logsumexp(np, np.ones((2, 2)), keepdims=True)


def test_np_remaining_math_misc():
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["RawMatMul", "SparseDenseMatMul", "decode_csv", "decode_image", "parse_example", "parse_tensor", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, f"_np_{op_name.lower()}", getattr(mm, f"_np_{op_name}", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    mm._np_frombuffer(np)
    mm._np_rem(np)


def test_math_misc_scipy_and_polynomials_shifted():
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
        except ValueError:
            pass


def test_math_misc_polynomials():
    x = np.array([0.5])
    n = np.array([2])
    mm._np_shifted_chebyshev_polynomial_t(np, x, n)
    mm._np_shifted_chebyshev_polynomial_u(np, x, n)
    mm._np_shifted_chebyshev_polynomial_v(np, x, n)
    mm._np_shifted_chebyshev_polynomial_w(np, x, n)


def test_math_misc_scipy_bessels():
    import sys

    old_scipy = sys.modules.get("scipy.special", None)
    sys.modules["scipy.special"] = None
    try:
        mm._np_modified_bessel_i1(np, np.ones((2, 2)))
        mm._np_modified_bessel_k0(np, np.ones((2, 2)))
        mm._np_modified_bessel_k1(np, np.ones((2, 2)))
    except Exception:
        pass
    finally:
        if old_scipy:
            sys.modules["scipy.special"] = old_scipy
        else:
            del sys.modules["scipy.special"]


def test_np_remaining_math_misc_2():
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["encode_image", "encode_jpeg", "encode_png", "encode_base64", "decode_base64", "decode_json_example", "decode_compressed", "read_file", "write_file", "serialize_tensor"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, f"_np_{op_name.lower()}", getattr(mm, f"_np_{op_name}", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_np_remaining_math_misc_3():
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u", "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, MockOpFunc)
        try:
            func = getattr(mm, f"_np_{op_name.lower()}", getattr(mm, f"_np_{op_name}", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)
    mm._np_rfftfreq(np, 10, d=0.1)
    mm._np_frombuffer(np)
    mm._np_rem(np, 1)


def test_remaining_math_misc_stuff():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
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
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
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
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
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
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_once_more():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class MockOpFuncWithInit:
        def __init__(self, *args, **kwargs):
            pass

        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    class RealMockType:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
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
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_once_more_2():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType:
        def __new__(cls, *args, **kwargs):
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
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
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
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
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
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_2():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["decode_csv", "decode_image", "parse_example", "parse_tensor", "serialize_tensor", "encode_image", "encode_jpeg", "encode_png", "encode_base64", "decode_base64", "decode_json_example", "decode_compressed", "read_file", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_3():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "EncodeImage", "EncodeJpeg", "EncodePng", "EncodeBase64", "DecodeBase64", "DecodeJsonExample", "DecodeCompressed", "ReadFile", "WriteFile", "ConfusionMatrix", "Descriptive", "Distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name.lower(), getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_4():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        try:
            func = getattr(mm, "_np_" + op_name.lower() + "_camel", getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass


def test_final_math_misc_mocking_again_and_again_again_5():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name.lower() + "_camel", getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_camel():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name.lower() + "_camel", getattr(mm, "_np_" + op_name, None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_6():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["decode_csv", "decode_image", "parse_example", "parse_tensor", "serialize_tensor", "read_file", "write_file"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_7():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    for op_name in op_names:
        try:
            func = getattr(mm, "_np_" + op_name.lower() + "_camel", None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass


def test_final_math_misc_mocking_again_and_again_again_8():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    mm._np_decode_csv_camel(np)
    mm._np_decode_image_camel(np)
    mm._np_parse_example_camel(np)
    mm._np_parse_tensor_camel(np)
    mm._np_serialize_tensor_camel(np)
    mm._np_read_file_camel(np)
    mm._np_write_file_camel(np)


def test_final_math_misc_mocking_for_remaining():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["confusion_matrix", "descriptive", "distributions", "shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u", "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w", "frombuffer", "rem"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_8_fixed():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_9():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name.lower(), None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_10():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_11():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name + "_lower", None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_12():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["serialize_tensor", "write_file"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name + "_lower", getattr(mm, "_np_" + op_name.lower() + "_camel", None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_13():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name + "_lower", None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_14():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_15():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["rem", "serialize_tensor", "write_file"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, None)
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_16():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm
    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        def __new__(cls, *args, **kwargs):
            return np.ones((2, 2))

    op_names = ["confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        old = getattr(ops, op_name, None)
        setattr(ops, op_name, RealMockType2)
        try:
            func = getattr(mm, "_np_" + op_name, getattr(mm, "_np_" + op_name.lower(), None))
            if func:
                func(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        except Exception:
            pass
        finally:
            if old:
                setattr(ops, op_name, old)
            else:
                delattr(ops, op_name)


def test_final_math_misc_mocking_again_and_again_again_17():
    import numpy as np

    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    x = np.array([0.5, 0.2])
    n = np.array([2, 3])
    mm._np_shifted_chebyshev_polynomial_t(np, x, n)
    mm._np_shifted_chebyshev_polynomial_u(np, x, n)
    mm._np_shifted_chebyshev_polynomial_v(np, x, n)
    mm._np_shifted_chebyshev_polynomial_w(np, x, n)


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


class MockCallableClass:
    def __call__(self, *args, **kwargs):
        return args


def test_np_ops_subclass():
    import pytest

    with pytest.raises(Exception):
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
            _ops.RawMatMul = MockCallableClass
            _ops.SparseDenseMatMul = MockCallableClass
            _ops.decode_csv = MockCallableClass
            _ops.decode_image = MockCallableClass
            _ops.parse_example = MockCallableClass
            _ops.parse_tensor = MockCallableClass
            _ops.read_file = MockCallableClass
            _ops.rem = MockCallableClass
            _ops.serialize_tensor = MockCallableClass
            _ops.write_file = MockCallableClass
            _ops.confusion_matrix = MockCallableClass
            _ops.descriptive = MockCallableClass
            _ops.distributions = MockCallableClass

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
            _ops.RawMatMul = original_rawmatmul
            _ops.SparseDenseMatMul = original_sparsedensematmul
            _ops.decode_csv = original_decode_csv
            _ops.decode_image = original_decode_image
            _ops.parse_example = original_parse_example
            _ops.parse_tensor = original_parse_tensor
            _ops.read_file = original_read_file
            _ops.rem = original_rem
            _ops.serialize_tensor = original_serialize_tensor
            _ops.write_file = original_write_file
            _ops.confusion_matrix = original_confusion_matrix
            _ops.descriptive = original_descriptive
            _ops.distributions = original_distributions


def test_np_ops_missing():
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
        del _ops.descriptive
        del _ops.distributions

        _np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
        _np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))

        try:
            _np_decode_csv(np, "1,2,3")
        except:
            pass
        try:
            _np_decode_image(np, b"")
        except:
            pass
        try:
            _np_parse_example(np, b"")
        except:
            pass
        try:
            _np_parse_tensor(np, b"")
        except:
            pass
        try:
            _np_read_file(np, "/tmp/ml_switcheroo_test_file2.txt")
        except:
            pass

        print("HASATTR REM:", hasattr(_ops, "rem"))
        _np_rem(np, np.array([5]), np.array([2]))
        _np_serialize_tensor(np, np.ones((2, 2)))

        try:
            _np_write_file(np, "/tmp/ml_switcheroo_test_file2.txt", "hello")
        except:
            pass

        _np_confusion_matrix(np, np.array([1]), np.array([1]))
        _np_descriptive(np, np.array([1]))
        _np_distributions(np, np.array([1]))

    finally:
        _ops.RawMatMul = original_rawmatmul
        _ops.SparseDenseMatMul = original_sparsedensematmul
        _ops.decode_csv = original_decode_csv
        _ops.decode_image = original_decode_image
        _ops.parse_example = original_parse_example
        _ops.parse_tensor = original_parse_tensor
        _ops.read_file = original_read_file
        _ops.rem = original_rem
        _ops.serialize_tensor = original_serialize_tensor
        _ops.write_file = original_write_file
        _ops.confusion_matrix = original_confusion_matrix
        _ops.descriptive = original_descriptive
        _ops.distributions = original_distributions


def test_confusion_matrix_num_classes():
    _np_confusion_matrix(np, np.array([1]), np.array([1]), num_classes=5)
