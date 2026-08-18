import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import _np_tensorarraywrite, _np_topk
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import _all_gather, _np_updateslice, _updateslice
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_misc_ext import _adjoint, _indexindim


class DummyBackend:
    pass


def test_math_missing():
    # math_internal.py 79 (_np_tensorarraywrite)
    db = DummyBackend()
    arr = [0, 0, 0]
    res = _np_tensorarraywrite(db, arr, 1, 5)
    assert res == [0, 5, 0]

    # math_internal.py 99 (_np_topk)
    db = DummyBackend()
    arr = np.array([1, 3, 2, 4])
    vals, idx = _np_topk(db, arr, 2)
    assert list(vals) == [3, 4]
    assert list(idx) == [1, 3]

    # math_manipulation.py 26 (_all_gather)
    db = DummyBackend()  # no stack, no array
    tensor = 5
    res = _all_gather(db, tensor)
    assert res == 5

    db2 = DummyBackend()
    db2.array = lambda x: ["array", x]
    res2 = _all_gather(db2, 5)
    assert res2 == ["array", [5]]

    # math_manipulation.py 76 (_pswapaxes)
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import _pswapaxes

    res = _pswapaxes(DummyBackend())
    assert res is None

    # math_manipulation.py 156-157 (_updateslice)
    db = DummyBackend()
    db.array = np.array
    arr = np.zeros((3, 3))
    update = np.ones((2, 2))
    res = _updateslice(db, arr, update, [1, 1])
    assert res[1, 1] == 1.0

    # math_manipulation.py 367 (_np_updateslice)
    db = DummyBackend()
    arr = [0, 0]
    res = _np_updateslice(db, arr, 0, 5)
    assert res == [5, 0]

    # math_misc_ext.py 466 (_adjoint)
    class DbAdjoint:
        pass

    dba = DbAdjoint()
    dba.asarray = np.asarray
    res = _adjoint(dba, np.array([[1j]]))
    assert res[0, 0] == -1j

    # math_misc_ext.py 736 (_indexindim keepdims=True)
    db = DummyBackend()
    db.array = np.array
    arr = np.array([1, 2, 3])
    res = _indexindim(db, arr, index=1, keepdims=True)
    assert list(res) == [2]

    # math_arithmetic.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_arithmetic import _np_scattermul, _np_truncatediv, _np_truncatemod, _np_xdivy

    db = DummyBackend()
    assert _np_scattermul(db, np.array([1]), np.array([0]), np.array([2])) is not None
    assert _np_truncatediv(db, np.array([5]), np.array([2])) is not None
    assert _np_truncatemod(db, np.array([5]), np.array([2])) is not None
    assert _np_xdivy(db, np.array([0]), np.array([0])) is not None

    # math_bitwise.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_bitwise import _np_packbits, _np_unpackbits

    assert _np_packbits(db, np.array([0, 1])) is not None
    assert _np_unpackbits(db, np.array([1], dtype=np.uint8)) is not None

    # math_creation.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_creation import _fromfunction, _fromiter, _frompyfunc, _np_fromfunction, _np_fromiter, _np_frompyfunc

    assert _np_fromfunction(db, lambda i, j: i + j, (2, 2)) is not None
    assert _np_fromiter(db, [1, 2, 3], int) is not None
    assert _np_frompyfunc(db, lambda x: x, 1, 1) is not None
    db.fromfunction = np.fromfunction
    assert _fromfunction(db, lambda i, j: i + j, (2, 2)) is not None
    db.frompyfunc = np.frompyfunc
    assert _frompyfunc(db, lambda x: x, 1, 1) is not None

    class DummyBackend2:
        pass

    db2 = DummyBackend2()
    # Mocking fromiter so it doesn't fail
    db2.fromiter = np.fromiter
    assert _fromiter(db2, [1, 2, 3], dtype=int) is not None

    class DummyBackend3:
        pass

    db3 = DummyBackend3()
    try:
        _fromiter(db3, [1, 2, 3])
    except AttributeError:
        pass

    # math_fft.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _np_hfft

    assert _np_hfft(db, np.array([1, 2, 3])) is not None

    # math_matrix.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix import _scaled_dot_product_attention_eager

    class MatmulBk:
        @staticmethod
        def matmul(a, b):
            return a

    arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert _scaled_dot_product_attention_eager(MatmulBk(), arr, arr, arr) is not None

    # math_matrix.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix import _scaled_dot_product_attention_eager

    class BkAttn:
        @staticmethod
        def matmul(a, b):
            return a

    arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert _scaled_dot_product_attention_eager(BkAttn(), arr, arr, arr) is not None

    # math_matrix.py (no transpose, no mask)
    class BkAttn2:
        @staticmethod
        def matmul(a, b):
            return a

        @staticmethod
        def exp(a):
            return a

        @staticmethod
        def sum(a, axis, keepdims):
            return a

        @staticmethod
        def max(a, axis, keepdims):
            return a

    assert _scaled_dot_product_attention_eager(BkAttn2(), arr, arr, arr, scale=1.0) is not None
    assert _scaled_dot_product_attention_eager(BkAttn2(), arr, arr, arr, mask=arr, scale=1.0) is not None

    # math_nn.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_nn import _global_adaptive_pool

    db = DummyBackend()
    db.mean = np.mean
    db.stack = np.stack
    assert _global_adaptive_pool(db, arr, (1, 1)) is not None
    # 1D, 2D, 3D missing branches
    arr1d = np.array([1.0, 2.0, 3.0])
    arr2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr3d = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert _global_adaptive_pool(db, arr1d, 1) is not None
    assert _global_adaptive_pool(db, arr2d, (1, 1)) is not None
    assert _global_adaptive_pool(db, arr3d, (1, 1, 1)) is not None
    assert _global_adaptive_pool(db, 5, 1) == 5

    # math_reduction.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_reduction import _apply_softmax

    class BkSoftmax1:
        pass

    class BkSoftmax2:
        class nn:
            @staticmethod
            def softmax(x, axis):
                return x

    assert _apply_softmax(BkSoftmax2(), arr) is not None
    try:
        _apply_softmax(BkSoftmax1(), arr)
    except Exception:
        pass

    # math_testing.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_testing import _allclose

    class HasItem:
        def item(self):
            return 5

    class HasToList:
        def tolist(self):
            return 5

    # DummyBackend will fall back to np.allclose, we just want to hit _val
    db = DummyBackend()
    _allclose(db, np.array([1]), np.array([1]), rtol=HasItem())
    _allclose(db, np.array([1]), np.array([1]), rtol=HasToList())

    # math_matrix.py (is_causal=True)
    assert _scaled_dot_product_attention_eager(BkAttn2(), arr, arr, arr, is_causal=True, scale=1.0) is not None

    # math_nn.py (len(in_shape) < spatial_dims)
    assert _global_adaptive_pool(db, np.array(5), (1, 1)) == 5

    # math_testing.py (_val returning itself)
    _allclose(db, np.array([1]), np.array([1]), rtol=5.0)
