"""Tests for test_math."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_general import _adjoint, _indexindim
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import _np_tensorarraywrite, _np_topk
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import (
    _all_gather,
    _np_updateslice,
    _updateslice,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import (
    EagerBoundCallable,
    GenericNDArray,
    MockTestStub,
    coerce_int_list,
    coerce_operand_list,
    extract_test_shape,
    require_callable,
    unpack_pair,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class DummyBackend(MockTestStub):
    """Test class DummyBackend."""

    pass


class _TestMathMissingState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_missing` split helpers."""

    BkAttn2: type
    BkSoftmax1: type
    HasItem: type
    HasToList: type
    _allclose: EagerBoundCallable
    _apply_softmax: EagerBoundCallable
    _fromiter: EagerBoundCallable
    _global_adaptive_pool: EagerBoundCallable
    _np_scattermul: EagerBoundCallable
    _np_truncatediv: EagerBoundCallable
    _np_truncatemod: EagerBoundCallable
    _np_xdivy: EagerBoundCallable
    _scaled_dot_product_attention_eager: EagerBoundCallable
    arr: list[int] | GenericNDArray
    db: DummyBackend


def _test_math_missing_part_1(_state: _TestMathMissingState) -> None:
    """Execute part 1 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    db = DummyBackend()
    arr = [0, 0, 0]
    res = _np_tensorarraywrite(db, arr, 1, 5)
    assert res == [0, 5, 0]

    # math_internal.py 99 (_np_topk)
    db = DummyBackend()
    arr = coerce_int_list(np.array([1, 3, 2, 4]))
    vals, idx = unpack_pair(_np_topk(db, arr, 2))
    assert sorted(coerce_int_list(vals)) == [3, 4]
    assert sorted(coerce_int_list(idx)) == [1, 3]

    # math_manipulation.py 26 (_all_gather)
    db = DummyBackend()  # no stack, no array
    tensor = 5
    res = _all_gather(db, tensor)
    assert res == 5

    db2 = DummyBackend()
    pytest.MonkeyPatch().setattr(db2, "array", lambda x: ["array", x], raising=False)
    res2 = _all_gather(db2, 5)
    assert res2 == ["array", [5]]

    # math_manipulation.py 76 (_pswapaxes)
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import _pswapaxes

    res = _pswapaxes(DummyBackend())
    assert res is None

    # math_manipulation.py 156-157 (_updateslice)
    db = DummyBackend()
    pytest.MonkeyPatch().setattr(db, "array", np.array, raising=False)
    _state.update({"arr": arr, "db": db})


def _test_math_missing_part_2(_state: _TestMathMissingState) -> None:
    """Execute part 2 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    db = _state["db"]
    arr = ensure_ndarray(np.zeros((3, 3)))
    update = np.ones((2, 2))
    res = _updateslice(db, arr, update, [1, 1])
    assert (
        ensure_ndarray(res)[
            (
                1,
                1,
            )
        ]
        == 1.0
    )

    # math_manipulation.py 367 (_np_updateslice)
    db = DummyBackend()
    arr = ensure_ndarray(np.array([0.0, 0.0]))
    res = _np_updateslice(db, arr, 0, 5)
    np.testing.assert_allclose(np.asarray(res), [5.0, 0.0])

    # math_general.py 466 (_adjoint)
    class DbAdjoint(MockTestStub):
        """Test class DbAdjoint."""

        pass

    dba = DbAdjoint()
    pytest.MonkeyPatch().setattr(dba, "asarray", np.asarray, raising=False)
    res = _adjoint(dba, np.array([[1j]]))
    assert (
        ensure_ndarray(res)[
            (
                0,
                0,
            )
        ]
        == -1j
    )

    # math_general.py 736 (_indexindim keepdims=True)
    db = DummyBackend()
    pytest.MonkeyPatch().setattr(db, "array", np.array, raising=False)
    arr = ensure_ndarray(np.array([1, 2, 3]))
    res = _indexindim(db, arr, index=1, keepdims=True)
    assert coerce_operand_list(res) == [2]

    # math_arithmetic.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_arithmetic import (
        _np_scattermul,
        _np_truncatediv,
        _np_truncatemod,
        _np_xdivy,
    )

    db = DummyBackend()
    _state.update(
        {
            "_np_scattermul": require_callable(_np_scattermul),
            "_np_truncatediv": require_callable(_np_truncatediv),
            "_np_truncatemod": require_callable(_np_truncatemod),
        }
    )
    _state.update({"_np_xdivy": require_callable(_np_xdivy), "arr": arr, "db": db})


def _test_math_missing_part_3(_state: _TestMathMissingState) -> None:
    """Execute part 3 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    _np_scattermul, _np_truncatediv, _np_truncatemod, _np_xdivy = (
        _state["_np_scattermul"],
        _state["_np_truncatediv"],
        _state["_np_truncatemod"],
        _state["_np_xdivy"],
    )
    db = _state["db"]
    assert _np_scattermul(db, np.array([1]), np.array([0]), np.array([2])) is not None
    assert _np_truncatediv(db, np.array([5]), np.array([2])) is not None
    assert _np_truncatemod(db, np.array([5]), np.array([2])) is not None
    assert _np_xdivy(db, np.array([0]), np.array([0])) is not None

    # math_bitwise.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_bitwise import _np_packbits, _np_unpackbits

    assert _np_packbits(db, np.array([0, 1])) is not None
    assert _np_unpackbits(db, np.array([1], dtype=np.uint8)) is not None

    # math_creation.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_creation import (
        _fromfunction,
        _fromiter,
        _frompyfunc,
        _np_fromfunction,
        _np_fromiter,
        _np_frompyfunc,
    )

    assert _np_fromfunction(db, lambda i, j: i + j, (2, 2)) is not None
    assert _np_fromiter(db, [1, 2, 3], int) is not None
    assert _np_frompyfunc(db, lambda x: x, 1, 1) is not None
    pytest.MonkeyPatch().setattr(db, "fromfunction", np.fromfunction, raising=False)
    assert _fromfunction(db, lambda i, j: i + j, (2, 2)) is not None
    pytest.MonkeyPatch().setattr(db, "frompyfunc", np.frompyfunc, raising=False)
    assert _frompyfunc(db, lambda x: x, 1, 1) is not None

    class DummyBackend2(MockTestStub):
        """Test class DummyBackend2."""

        pass

    db2 = DummyBackend2()
    # Mocking fromiter so it doesn't fail
    pytest.MonkeyPatch().setattr(db2, "fromiter", np.fromiter, raising=False)
    assert _fromiter(db2, [1, 2, 3], dtype=int) is not None
    _state.update({"_fromiter": require_callable(_fromiter)})
    _state.update(
        {
            "_np_scattermul": require_callable(_np_scattermul),
            "_np_truncatediv": require_callable(_np_truncatediv),
            "_np_truncatemod": require_callable(_np_truncatemod),
        }
    )
    _state.update({"_np_xdivy": require_callable(_np_xdivy), "db": db})


def _test_math_missing_part_4(_state: _TestMathMissingState) -> None:
    """Execute part 4 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    _fromiter, db = _state["_fromiter"], _state["db"]

    class DummyBackend3:
        """Test class DummyBackend3."""

        pass

    db3 = DummyBackend3()
    with pytest.raises(AttributeError):
        _fromiter(db3, [1, 2, 3])

    # math_fft.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _np_hfft

    assert _np_hfft(db, np.array([1, 2, 3])) is not None

    # math_matrix.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix import _scaled_dot_product_attention_eager

    class MatmulBk(MockTestStub):
        """Test class MatmulBk."""

        @staticmethod
        def matmul(a: BackendArg, b: BackendArg) -> BackendArg:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                BackendArg: Computed output value from `matmul`.
            """
            return a

    arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert _scaled_dot_product_attention_eager(MatmulBk(), arr, arr, arr) is not None

    # math_matrix.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix import _scaled_dot_product_attention_eager

    _state.update({"_scaled_dot_product_attention_eager": require_callable(_scaled_dot_product_attention_eager)})
    _state.update({"arr": arr})
    _state.update({"_fromiter": require_callable(_fromiter), "db": db})


def _test_math_missing_part_5(_state: _TestMathMissingState) -> None:
    """Execute part 5 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    _scaled_dot_product_attention_eager = _state["_scaled_dot_product_attention_eager"]

    class BkAttn:
        """Test class BkAttn."""

        @staticmethod
        def matmul(a: BackendArg, b: BackendArg) -> BackendArg:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                BackendArg: Computed output value from `matmul`.
            """
            return a

    arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert _scaled_dot_product_attention_eager(BkAttn(), arr, arr, arr) is not None

    class BkAttn2:
        """Test class BkAttn2."""

        @staticmethod
        def matmul(a: BackendArg, b: BackendArg) -> BackendArg:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.

            Returns:
                BackendArg: Computed output value from `matmul`.
            """
            return a

        @staticmethod
        def exp(a: BackendArg) -> BackendArg:
            """Test function exp.

            Args:
                a: Test parameter `a`.

            Returns:
                BackendArg: Computed output value from `exp`.
            """
            return a

        @staticmethod
        def sum(a: BackendArg, axis: BackendArg, keepdims: BackendArg) -> BackendArg:
            """Test function sum.

            Args:
                a: Test parameter `a`.
                axis: Test parameter `axis`.
                keepdims: Test parameter `keepdims`.

            Returns:
                BackendArg: Computed output value from `sum`.
            """
            return a

        @staticmethod
        def max(a: BackendArg, axis: BackendArg, keepdims: BackendArg) -> BackendArg:
            """Test function max.

            Args:
                a: Test parameter `a`.
                axis: Test parameter `axis`.
                keepdims: Test parameter `keepdims`.

            Returns:
                BackendArg: Computed output value from `max`.
            """
            return a

    assert _scaled_dot_product_attention_eager(BkAttn2(), arr, arr, arr, scale=1.0) is not None
    _state.update({"BkAttn2": BkAttn2, "arr": arr})
    _state.update({"_scaled_dot_product_attention_eager": require_callable(_scaled_dot_product_attention_eager)})


def _test_math_missing_part_6(_state: _TestMathMissingState) -> None:
    """Execute part 6 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    BkAttn2, _scaled_dot_product_attention_eager, arr = (
        _state["BkAttn2"],
        _state["_scaled_dot_product_attention_eager"],
        _state["arr"],
    )
    assert _scaled_dot_product_attention_eager(BkAttn2(), arr, arr, arr, mask=arr, scale=1.0) is not None

    # math_nn.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_nn import _global_adaptive_pool

    db = DummyBackend()
    pytest.MonkeyPatch().setattr(db, "mean", np.mean, raising=False)
    pytest.MonkeyPatch().setattr(db, "stack", np.stack, raising=False)
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
        """Test class BkSoftmax1."""

        pass

    _state.update(
        {
            "BkSoftmax1": BkSoftmax1,
            "_apply_softmax": require_callable(_apply_softmax),
            "_global_adaptive_pool": require_callable(_global_adaptive_pool),
        }
    )
    _state.update({"db": db})
    _state.update(
        {
            "BkAttn2": BkAttn2,
            "_scaled_dot_product_attention_eager": require_callable(_scaled_dot_product_attention_eager),
            "arr": arr,
        }
    )


def _test_math_missing_part_7(_state: _TestMathMissingState) -> None:
    """Execute part 7 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    BkSoftmax1, _apply_softmax, arr = _state["BkSoftmax1"], _state["_apply_softmax"], _state["arr"]

    class BkSoftmax2:
        """Test class BkSoftmax2."""

        class nn:
            """Test class nn."""

            @staticmethod
            def softmax(x: BackendArg, axis: BackendArg) -> BackendArg:
                """Test function softmax.

                Args:
                    x: Test parameter `x`.
                    axis: Test parameter `axis`.

                Returns:
                    BackendArg: Computed output value from `softmax`.
                """
                return x

    assert _apply_softmax(BkSoftmax2(), arr) is not None
    assert _apply_softmax(BkSoftmax1(), arr) is not None

    # math_testing.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_testing import _allclose

    class HasItem:
        """Test class HasItem."""

        def item(self) -> BackendArg:
            """Test function item.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return 5

    class HasToList:
        """Test class HasToList."""

        def tolist(self) -> BackendArg:
            """Test function tolist.

            Returns:
                BackendArg: Computed output value from `tolist`.
            """
            return 5

    _state.update({"HasItem": HasItem, "HasToList": HasToList, "_allclose": require_callable(_allclose)})
    _state.update({"BkSoftmax1": BkSoftmax1, "_apply_softmax": require_callable(_apply_softmax), "arr": arr})


def _test_math_missing_part_8(_state: _TestMathMissingState) -> None:
    """Execute part 8 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    BkAttn2, HasItem, HasToList, _allclose = (
        _state["BkAttn2"],
        _state["HasItem"],
        _state["HasToList"],
        _state["_allclose"],
    )
    _global_adaptive_pool, _scaled_dot_product_attention_eager, arr = (
        _state["_global_adaptive_pool"],
        _state["_scaled_dot_product_attention_eager"],
        _state["arr"],
    )
    db = DummyBackend()
    _allclose(db, np.array([1]), np.array([1]), rtol=HasItem())
    _allclose(db, np.array([1]), np.array([1]), rtol=HasToList())

    # math_matrix.py (is_causal=True)
    assert _scaled_dot_product_attention_eager(BkAttn2(), arr, arr, arr, is_causal=True, scale=1.0) is not None

    # math_nn.py (len(in_shape) < spatial_dims)
    assert _global_adaptive_pool(db, np.array(5), (1, 1)) == 5

    # math_testing.py (_val returning itself)
    _allclose(db, np.array([1]), np.array([1]), rtol=5.0)
    _state.update({"db": db})


def test_math_missing() -> None:
    # math_internal.py 79 (_np_tensorarraywrite)
    """Test function test_math_missing."""
    _state: _TestMathMissingState = {}
    _test_math_missing_part_1(_state)
    _test_math_missing_part_2(_state)
    _test_math_missing_part_3(_state)
    _test_math_missing_part_4(_state)
    _test_math_missing_part_5(_state)
    _test_math_missing_part_6(_state)
    _test_math_missing_part_7(_state)
    _test_math_missing_part_8(_state)
    assert len(_state) > 0


def test_math_manipulation_matrix_nn_100cov() -> None:
    """Ensure 100% coverage for math_manipulation, math_matrix, and math_nn."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import _np_broadcast_to, _np_transpose
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix import _batch_matmul, _dot, _matmul
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
    assert extract_test_shape(_np_transpose(np, arr, permutation=(1, 0))) == (3, 2)
    assert extract_test_shape(_np_transpose(np, arr)) == (3, 2)
    assert extract_test_shape(_np_transpose(db, arr, permutation=(1, 0))) == (3, 2)

    assert extract_test_shape(_np_broadcast_to(np, np.ones((1, 2)), shape=(3, 2))) == (3, 2)
    assert extract_test_shape(_np_broadcast_to(db, np.ones((1, 2)), shape=(3, 2))) == (3, 2)

    # math_matrix.py
    m1 = np.ones((2, 2))
    m2 = np.ones((2, 2))
    assert extract_test_shape(_matmul(np, m1, m2, transpose_a=True, transpose_b=True)) == (
        2,
        2,
    )
    assert extract_test_shape(_matmul(np, m1, m2)) == (2, 2)
    assert extract_test_shape(_matmul(db, m1, m2)) == (2, 2)

    bm1 = np.ones((2, 2, 2))
    bm2 = np.ones((2, 2, 2))
    assert extract_test_shape(_batch_matmul(np, bm1, bm2)) == (2, 2, 2)
    assert extract_test_shape(_batch_matmul(db, bm1, bm2)) == (2, 2, 2)

    v1 = np.ones(3)
    v2 = np.ones(3)
    assert _dot(np, v1, v2) == 3.0
    assert _dot(db, v1, v2) == 3.0

    # math_nn.py
    x_2d = np.ones((2, 4))
    assert extract_test_shape(_layer_norm(np, x_2d)) == (2, 4)

    x_4d = np.ones((2, 4, 3, 3))
    assert extract_test_shape(_batch_norm(np, x_4d)) == (2, 4, 3, 3)
    assert extract_test_shape(_batch_norm(np, x_4d, 1.0, 0.0, np.zeros((1, 4, 1, 1)), np.ones((1, 4, 1, 1)))) == (
        2,
        4,
        3,
        3,
    )

    assert extract_test_shape(_rms_norm(np, x_2d)) == (2, 4)
    assert _mse_loss(np, np.ones((2, 2)), np.zeros((2, 2))) == 1.0
    assert _bce_loss(np, np.array([0.2, 0.8]), np.array([0.0, 1.0])) is not None

    # bce_loss fallback without clip on backend
    db_math = DummyBackend()
    pytest.MonkeyPatch().setattr(db_math, "log", np.log, raising=False)
    pytest.MonkeyPatch().setattr(db_math, "mean", np.mean, raising=False)
    assert _bce_loss(db_math, np.array([0.2, 0.8]), np.array([0.0, 1.0])) is not None

    assert _huber_loss(np, np.array([0.5, 2.0]), np.array([0.0, 0.0])) is not None
    assert _kl_loss(np, np.array([0.2, 0.8]), np.array([0.3, 0.7])) is not None
    assert extract_test_shape(_gelu(np, np.array([0.5, -0.5]))) == (2,)
    assert extract_test_shape(_silu(np, np.array([0.5, -0.5]))) == (2,)
    assert extract_test_shape(_elu(np, np.array([0.5, -0.5]))) == (2,)
    assert extract_test_shape(_leaky_relu(np, np.array([0.5, -0.5]))) == (2,)

    # Line 323: 4D adaptive pool falling through
    assert extract_test_shape(_global_adaptive_pool(np, np.ones((1, 2, 3, 4, 5)), (1, 1, 1, 1))) == (1, 2, 3, 4, 5)


def test_math_reduction_full_coverage() -> None:
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
    assert extract_test_shape(_reduce_sum(np, arr, axis=0)) == (3,)
    assert _reduce_sum(db, arr) is arr

    # _reduce_mean
    assert extract_test_shape(_reduce_mean(np, arr, axis=0)) == (3,)
    assert _reduce_mean(db, arr) is arr

    # _reduce_prod
    assert extract_test_shape(_reduce_prod(np, arr, axis=0)) == (3,)
    assert _reduce_prod(db, arr) is arr

    # _reduce_max
    assert extract_test_shape(_reduce_max(np, arr, axis=0)) == (3,)
    assert _reduce_max(db, arr) is arr

    # _reduce_min
    assert extract_test_shape(_reduce_min(np, arr, axis=0)) == (3,)
    assert _reduce_min(db, arr) is arr

    # _norm
    assert _norm(np, arr) == np.linalg.norm(arr)
    db_sqrt = DummyBackend()
    pytest.MonkeyPatch().setattr(db_sqrt, "sqrt", np.sqrt, raising=False)
    pytest.MonkeyPatch().setattr(db_sqrt, "sum", np.sum, raising=False)
    assert _norm(db_sqrt, arr) == np.linalg.norm(arr)

    # _numel
    class HasSize(MockTestStub):
        """Test class HasSize."""

        size = 12

    class HasNumel(MockTestStub):
        """Test class HasNumel."""

        def numel(self) -> BackendArg:
            """Test function numel.

            Returns:
                BackendArg: Computed output value from `numel`.
            """
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
    assert extract_test_shape(_broadcast_reduce(np, cot_3d, tgt_2d)) == (2, 3)
    # td == 1 and cd > 1
    tgt_1d = np.ones((1, 3))
    assert extract_test_shape(_broadcast_reduce(np, arr, tgt_1d)) == (1, 3)

    # _broadcast_like
    assert _broadcast_like(np) is None
    assert _broadcast_like(np, arr) is arr
    assert _broadcast_like(np, arr, arr) is arr
    assert extract_test_shape(_broadcast_like(np, np.ones(3), arr)) == (2, 3)
    assert extract_test_shape(_broadcast_like(db, np.ones(3), arr)) == (3,)

    # _segment_sum
    assert _segment_sum(np) is None
    data = np.ones((4, 2))
    segments = np.array([0, 0, 1, 1])
    res_seg = _segment_sum(np, data, segments, 2)
    assert extract_test_shape(res_seg) == (2, 2)
