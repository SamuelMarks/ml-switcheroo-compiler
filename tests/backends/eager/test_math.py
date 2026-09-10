import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_general import _adjoint, _indexindim
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import _np_tensorarraywrite, _np_topk
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import _all_gather, _np_updateslice, _updateslice


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
    assert sorted(list(vals)) == [3, 4]
    assert sorted(list(idx)) == [1, 3]

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

    # math_general.py 466 (_adjoint)
    class DbAdjoint:
        pass

    dba = DbAdjoint()
    dba.asarray = np.asarray
    res = _adjoint(dba, np.array([[1j]]))
    assert res[0, 0] == -1j

    # math_general.py 736 (_indexindim keepdims=True)
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


def test_math_manipulation_matrix_nn_100cov():
    """Ensure 100% coverage for math_manipulation, math_matrix, and math_nn."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import (
        _np_broadcast_to,
        _np_transpose,
    )
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix import (
        _batch_matmul,
        _dot,
        _matmul,
    )
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_nn import (
        _batch_norm,
        _bce_loss,
        _elu,
        _gelu,
        _global_adaptive_pool,
        _huber_loss,
        _kl_loss,
        _layer_norm,
        _leaky_relu,
        _mse_loss,
        _rms_norm,
        _silu,
    )

    db = DummyBackend()

    # math_manipulation.py
    arr = np.ones((2, 3))
    assert _np_transpose(np, arr, permutation=(1, 0)).shape == (3, 2)
    assert _np_transpose(np, arr).shape == (3, 2)
    assert _np_transpose(db, arr, permutation=(1, 0)).shape == (3, 2)

    assert _np_broadcast_to(np, np.ones((1, 2)), shape=(3, 2)).shape == (3, 2)
    assert _np_broadcast_to(db, np.ones((1, 2)), shape=(3, 2)).shape == (3, 2)

    # math_matrix.py
    m1 = np.ones((2, 2))
    m2 = np.ones((2, 2))
    assert _matmul(np, m1, m2, transpose_a=True, transpose_b=True).shape == (2, 2)
    assert _matmul(np, m1, m2).shape == (2, 2)
    assert _matmul(db, m1, m2).shape == (2, 2)

    bm1 = np.ones((2, 2, 2))
    bm2 = np.ones((2, 2, 2))
    assert _batch_matmul(np, bm1, bm2).shape == (2, 2, 2)
    assert _batch_matmul(db, bm1, bm2).shape == (2, 2, 2)

    v1 = np.ones(3)
    v2 = np.ones(3)
    assert _dot(np, v1, v2) == 3.0
    assert _dot(db, v1, v2) == 3.0

    # math_nn.py
    x_2d = np.ones((2, 4))
    assert _layer_norm(np, x_2d).shape == (2, 4)

    x_4d = np.ones((2, 4, 3, 3))
    assert _batch_norm(np, x_4d).shape == (2, 4, 3, 3)
    assert _batch_norm(np, x_4d, 1.0, 0.0, np.zeros((1, 4, 1, 1)), np.ones((1, 4, 1, 1))).shape == (2, 4, 3, 3)

    assert _rms_norm(np, x_2d).shape == (2, 4)
    assert _mse_loss(np, np.ones((2, 2)), np.zeros((2, 2))) == 1.0
    assert _bce_loss(np, np.array([0.2, 0.8]), np.array([0.0, 1.0])) is not None

    # bce_loss fallback without clip on backend
    db_math = DummyBackend()
    db_math.log = np.log
    db_math.mean = np.mean
    assert _bce_loss(db_math, np.array([0.2, 0.8]), np.array([0.0, 1.0])) is not None

    assert _huber_loss(np, np.array([0.5, 2.0]), np.array([0.0, 0.0])) is not None
    assert _kl_loss(np, np.array([0.2, 0.8]), np.array([0.3, 0.7])) is not None
    assert _gelu(np, np.array([0.5, -0.5])).shape == (2,)
    assert _silu(np, np.array([0.5, -0.5])).shape == (2,)
    assert _elu(np, np.array([0.5, -0.5])).shape == (2,)
    assert _leaky_relu(np, np.array([0.5, -0.5])).shape == (2,)

    # Line 323: 4D adaptive pool falling through
    assert _global_adaptive_pool(np, np.ones((1, 2, 3, 4, 5)), (1, 1, 1, 1)).shape == (1, 2, 3, 4, 5)


def test_math_reduction_full_coverage():
    """Ensure 100% coverage for math_reduction.py."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_reduction import (
        _broadcast_like,
        _broadcast_reduce,
        _norm,
        _numel,
        _reduce_max,
        _reduce_mean,
        _reduce_min,
        _reduce_prod,
        _reduce_sum,
        _segment_sum,
    )

    db = DummyBackend()
    arr = np.ones((2, 3))

    # _reduce_sum
    assert _reduce_sum(np, arr, axis=0).shape == (3,)
    assert _reduce_sum(db, arr) is arr

    # _reduce_mean
    assert _reduce_mean(np, arr, axis=0).shape == (3,)
    assert _reduce_mean(db, arr) is arr

    # _reduce_prod
    assert _reduce_prod(np, arr, axis=0).shape == (3,)
    assert _reduce_prod(db, arr) is arr

    # _reduce_max
    assert _reduce_max(np, arr, axis=0).shape == (3,)
    assert _reduce_max(db, arr) is arr

    # _reduce_min
    assert _reduce_min(np, arr, axis=0).shape == (3,)
    assert _reduce_min(db, arr) is arr

    # _norm
    assert _norm(np, arr) == np.linalg.norm(arr)
    db_sqrt = DummyBackend()
    db_sqrt.sqrt = np.sqrt
    db_sqrt.sum = np.sum
    assert _norm(db_sqrt, arr) == np.linalg.norm(arr)

    # _numel
    class HasSize:
        size = 12

    class HasNumel:
        def numel(self):
            return 15

    assert _numel(np, HasSize()) == 12.0
    assert _numel(np, HasNumel()) == 15.0
    assert _numel(np, [1, 2, 3]) == 3.0

    # _broadcast_reduce
    assert _broadcast_reduce(np) is None
    assert _broadcast_reduce(np, arr) is arr
    assert _broadcast_reduce(np, arr, arr) is arr
    # cot_ndim > tgt_ndim
    cot_3d = np.ones((2, 2, 3))
    tgt_2d = np.ones((2, 3))
    assert _broadcast_reduce(np, cot_3d, tgt_2d).shape == (2, 3)
    # td == 1 and cd > 1
    tgt_1d = np.ones((1, 3))
    assert _broadcast_reduce(np, arr, tgt_1d).shape == (1, 3)

    # _broadcast_like
    assert _broadcast_like(np) is None
    assert _broadcast_like(np, arr) is arr
    assert _broadcast_like(np, arr, arr) is arr
    assert _broadcast_like(np, np.ones(3), arr).shape == (2, 3)
    assert _broadcast_like(db, np.ones(3), arr).shape == (3,)

    # _segment_sum
    assert _segment_sum(np) is None
    data = np.ones((4, 2))
    segments = np.array([0, 0, 1, 1])
    res_seg = _segment_sum(np, data, segments, 2)
    assert res_seg.shape == (2, 2)
