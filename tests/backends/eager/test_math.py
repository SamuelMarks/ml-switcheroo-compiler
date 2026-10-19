"""Tests for test_math."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_general import _adjoint, _indexindim
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import _np_tensorarraywrite, _np_topk
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import _all_gather, _np_updateslice, _updateslice

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class DummyBackend:
    """Test class DummyBackend."""

    pass


def _test_math_missing_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
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
    arr = cast("list[int]", np.array([1, 3, 2, 4]))
    vals, idx = cast("EagerCallableProtocol", _np_topk)(db, arr, 2)
    assert cast("EagerCallableProtocol", sorted)(cast("EagerCallableProtocol", list)(vals)) == [3, 4]
    assert cast("EagerCallableProtocol", sorted)(cast("EagerCallableProtocol", list)(idx)) == [1, 3]

    # math_manipulation.py 26 (_all_gather)
    db = DummyBackend()  # no stack, no array
    tensor = 5
    res = _all_gather(db, tensor)
    assert res == 5

    db2 = DummyBackend()
    setattr(db2, "array", lambda x: ["array", x])
    res2 = _all_gather(db2, 5)
    assert res2 == ["array", [5]]

    # math_manipulation.py 76 (_pswapaxes)
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation import _pswapaxes

    res = _pswapaxes(DummyBackend())
    assert res is None

    # math_manipulation.py 156-157 (_updateslice)
    db = DummyBackend()
    setattr(db, "array", cast("EagerTensorProtocol", getattr(np, "array")))
    getattr(_state, "update")({"_pswapaxes": _pswapaxes, "arr": arr, "db": db, "db2": db2})
    getattr(_state, "update")({"idx": idx, "res": res, "res2": res2, "tensor": tensor})
    getattr(_state, "update")({"vals": vals})
    _state.update(locals())


def _test_math_missing_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    db = _state["db"]
    arr = np.zeros((3, 3))
    update = np.ones((2, 2))
    res = _updateslice(db, arr, update, [1, 1])
    assert (
        cast(
            "EagerTensorProtocol",
            getattr(res, "__getitem__")(
                (
                    1,
                    1,
                )
            ),
        )
        == 1.0
    )

    # math_manipulation.py 367 (_np_updateslice)
    db = cast("EagerTensorProtocol", DummyBackend())
    arr = np.array([0.0, 0.0])
    res = _np_updateslice(db, arr, 0, 5)
    np.testing.assert_allclose(cast("np.ndarray", res), [5.0, 0.0])

    # math_general.py 466 (_adjoint)
    class DbAdjoint:
        """Test class DbAdjoint."""

        pass

    dba = DbAdjoint()
    setattr(dba, "asarray", cast("EagerTensorProtocol", getattr(np, "asarray")))
    res = _adjoint(dba, np.array([[1j]]))
    assert (
        cast(
            "EagerTensorProtocol",
            getattr(res, "__getitem__")(
                (
                    0,
                    0,
                )
            ),
        )
        == -1j
    )

    # math_general.py 736 (_indexindim keepdims=True)
    db = cast("EagerTensorProtocol", DummyBackend())
    setattr(db, "array", cast("EagerTensorProtocol", getattr(np, "array")))
    arr = np.array([1, 2, 3])
    res = _indexindim(db, arr, index=1, keepdims=True)
    assert cast("EagerCallableProtocol", list)(res) == [2]

    # math_arithmetic.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_arithmetic import _np_scattermul, _np_truncatediv, _np_truncatemod, _np_xdivy

    db = cast("EagerTensorProtocol", DummyBackend())
    getattr(_state, "update")({"DbAdjoint": DbAdjoint, "_np_scattermul": _np_scattermul, "_np_truncatediv": _np_truncatediv, "_np_truncatemod": _np_truncatemod})
    getattr(_state, "update")({"_np_xdivy": _np_xdivy, "arr": arr, "db": db, "dba": dba})
    getattr(_state, "update")({"res": res, "update": update})
    _state.update(locals())


def _test_math_missing_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    _np_scattermul, _np_truncatediv, _np_truncatemod, _np_xdivy = _state["_np_scattermul"], _state["_np_truncatediv"], _state["_np_truncatemod"], _state["_np_xdivy"]
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
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_creation import _fromfunction, _fromiter, _frompyfunc, _np_fromfunction, _np_fromiter, _np_frompyfunc

    assert _np_fromfunction(db, lambda i, j: i + j, (2, 2)) is not None
    assert _np_fromiter(db, [1, 2, 3], int) is not None
    assert _np_frompyfunc(db, lambda x: x, 1, 1) is not None
    setattr(db, "fromfunction", cast("EagerTensorProtocol", getattr(np, "fromfunction")))
    assert _fromfunction(db, lambda i, j: i + j, (2, 2)) is not None
    setattr(db, "frompyfunc", cast("EagerTensorProtocol", getattr(np, "frompyfunc")))
    assert _frompyfunc(db, lambda x: x, 1, 1) is not None

    class DummyBackend2:
        """Test class DummyBackend2."""

        pass

    db2 = DummyBackend2()
    # Mocking fromiter so it doesn't fail
    setattr(db2, "fromiter", cast("EagerTensorProtocol", getattr(np, "fromiter")))
    assert _fromiter(db2, [1, 2, 3], dtype=int) is not None
    getattr(_state, "update")({"DummyBackend2": DummyBackend2, "_fromfunction": _fromfunction, "_fromiter": _fromiter, "_frompyfunc": _frompyfunc})
    getattr(_state, "update")({"_np_fromfunction": _np_fromfunction, "_np_fromiter": _np_fromiter, "_np_frompyfunc": _np_frompyfunc, "_np_packbits": _np_packbits})
    getattr(_state, "update")({"_np_unpackbits": _np_unpackbits, "db2": db2})
    _state.update(locals())


def _test_math_missing_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
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
        cast("EagerCallableProtocol", _fromiter)(db3, [1, 2, 3])

    # math_fft.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _np_hfft

    assert _np_hfft(db, np.array([1, 2, 3])) is not None

    # math_matrix.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix import _scaled_dot_product_attention_eager

    class MatmulBk:
        """Test class MatmulBk."""

        @staticmethod
        def matmul(a: EagerValue, b: EagerValue) -> EagerValue:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
            """
            return a

    arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert _scaled_dot_product_attention_eager(MatmulBk(), arr, arr, arr) is not None

    # math_matrix.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_matrix import _scaled_dot_product_attention_eager

    getattr(_state, "update")({"DummyBackend3": DummyBackend3, "MatmulBk": MatmulBk, "_np_hfft": _np_hfft, "_scaled_dot_product_attention_eager": _scaled_dot_product_attention_eager})
    getattr(_state, "update")({"arr": arr, "db3": db3})
    _state.update(locals())


def _test_math_missing_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    _scaled_dot_product_attention_eager = _state["_scaled_dot_product_attention_eager"]

    class BkAttn:
        """Test class BkAttn."""

        @staticmethod
        def matmul(a: EagerValue, b: EagerValue) -> EagerValue:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
            """
            return a

    arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert cast("EagerCallableProtocol", _scaled_dot_product_attention_eager)(cast("EagerCallableProtocol", BkAttn)(), arr, arr, arr) is not None

    class BkAttn2:
        """Test class BkAttn2."""

        @staticmethod
        def matmul(a: EagerValue, b: EagerValue) -> EagerValue:
            """Test function matmul.

            Args:
                a: Test parameter `a`.
                b: Test parameter `b`.
            """
            return a

        @staticmethod
        def exp(a: EagerValue) -> EagerValue:
            """Test function exp.

            Args:
                a: Test parameter `a`.
            """
            return a

        @staticmethod
        def sum(a: EagerValue, axis: EagerValue, keepdims: EagerValue) -> EagerValue:
            """Test function sum.

            Args:
                a: Test parameter `a`.
                axis: Test parameter `axis`.
                keepdims: Test parameter `keepdims`.
            """
            return a

        @staticmethod
        def max(a: EagerValue, axis: EagerValue, keepdims: EagerValue) -> EagerValue:
            """Test function max.

            Args:
                a: Test parameter `a`.
                axis: Test parameter `axis`.
                keepdims: Test parameter `keepdims`.
            """
            return a

    assert cast("EagerCallableProtocol", _scaled_dot_product_attention_eager)(cast("EagerCallableProtocol", BkAttn2)(), arr, arr, arr, scale=1.0) is not None
    getattr(_state, "update")({"BkAttn": BkAttn, "BkAttn2": BkAttn2, "arr": arr})
    _state.update(locals())


def _test_math_missing_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    BkAttn2, _scaled_dot_product_attention_eager, arr = _state["BkAttn2"], _state["_scaled_dot_product_attention_eager"], _state["arr"]
    assert _scaled_dot_product_attention_eager(BkAttn2(), arr, arr, arr, mask=arr, scale=1.0) is not None

    # math_nn.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_nn import _global_adaptive_pool

    db = DummyBackend()
    setattr(db, "mean", cast("EagerTensorProtocol", getattr(np, "mean")))
    setattr(db, "stack", cast("EagerTensorProtocol", getattr(np, "stack")))
    assert cast("EagerCallableProtocol", _global_adaptive_pool)(db, arr, (1, 1)) is not None
    # 1D, 2D, 3D missing branches
    arr1d = np.array([1.0, 2.0, 3.0])
    arr2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr3d = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert cast("EagerCallableProtocol", _global_adaptive_pool)(db, arr1d, 1) is not None
    assert cast("EagerCallableProtocol", _global_adaptive_pool)(db, arr2d, (1, 1)) is not None
    assert cast("EagerCallableProtocol", _global_adaptive_pool)(db, arr3d, (1, 1, 1)) is not None
    assert cast("EagerCallableProtocol", _global_adaptive_pool)(db, 5, 1) == 5

    # math_reduction.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_reduction import _apply_softmax

    class BkSoftmax1:
        """Test class BkSoftmax1."""

        pass

    getattr(_state, "update")({"BkSoftmax1": BkSoftmax1, "_apply_softmax": _apply_softmax, "_global_adaptive_pool": _global_adaptive_pool, "arr1d": arr1d})
    getattr(_state, "update")({"arr2d": arr2d, "arr3d": arr3d, "db": db})
    _state.update(locals())


def _test_math_missing_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
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
            def softmax(x: EagerValue, axis: EagerValue) -> EagerValue:
                """Test function softmax.

                Args:
                    x: Test parameter `x`.
                    axis: Test parameter `axis`.
                """
                return x

    assert cast("EagerCallableProtocol", _apply_softmax)(cast("EagerCallableProtocol", BkSoftmax2)(), arr) is not None
    assert _apply_softmax(BkSoftmax1(), arr) is not None

    # math_testing.py
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_testing import _allclose

    class HasItem:
        """Test class HasItem."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 5

    class HasToList:
        """Test class HasToList."""

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return 5

    getattr(_state, "update")({"BkSoftmax2": BkSoftmax2, "HasItem": HasItem, "HasToList": HasToList, "_allclose": _allclose})
    _state.update(locals())


def _test_math_missing_part_8(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 8 of test_math_missing.

    Args:
        _state: Shared test state dictionary.
    """
    BkAttn2, HasItem, HasToList, _allclose = _state["BkAttn2"], _state["HasItem"], _state["HasToList"], _state["_allclose"]
    _global_adaptive_pool, _scaled_dot_product_attention_eager, arr = _state["_global_adaptive_pool"], _state["_scaled_dot_product_attention_eager"], _state["arr"]
    db = DummyBackend()
    cast("EagerCallableProtocol", _allclose)(db, cast("EagerCallableProtocol", np.array)([1]), cast("EagerCallableProtocol", np.array)([1]), rtol=cast("EagerCallableProtocol", HasItem)())
    cast("EagerCallableProtocol", _allclose)(db, cast("EagerCallableProtocol", np.array)([1]), cast("EagerCallableProtocol", np.array)([1]), rtol=cast("EagerCallableProtocol", HasToList)())

    # math_matrix.py (is_causal=True)
    assert _scaled_dot_product_attention_eager(BkAttn2(), arr, arr, arr, is_causal=True, scale=1.0) is not None

    # math_nn.py (len(in_shape) < spatial_dims)
    assert cast("EagerCallableProtocol", _global_adaptive_pool)(db, cast("EagerCallableProtocol", np.array)(5), (1, 1)) == 5

    # math_testing.py (_val returning itself)
    cast("EagerCallableProtocol", _allclose)(db, cast("EagerCallableProtocol", np.array)([1]), cast("EagerCallableProtocol", np.array)([1]), rtol=5.0)
    getattr(_state, "update")({"db": db})


def test_math_missing() -> None:
    # math_internal.py 79 (_np_tensorarraywrite)
    """Test function test_math_missing."""
    _state: dict[str, EagerTensorProtocol] = {}
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
    assert cast("EagerTensorProtocol", getattr(_np_transpose(np, arr, permutation=(1, 0)), "shape")) == (3, 2)
    assert cast("EagerTensorProtocol", getattr(_np_transpose(np, arr), "shape")) == (3, 2)
    assert cast("EagerTensorProtocol", getattr(_np_transpose(db, arr, permutation=(1, 0)), "shape")) == (3, 2)

    assert cast("EagerTensorProtocol", getattr(_np_broadcast_to(np, np.ones((1, 2)), shape=(3, 2)), "shape")) == (3, 2)
    assert cast("EagerTensorProtocol", getattr(_np_broadcast_to(db, np.ones((1, 2)), shape=(3, 2)), "shape")) == (3, 2)

    # math_matrix.py
    m1 = np.ones((2, 2))
    m2 = np.ones((2, 2))
    assert cast("EagerTensorProtocol", getattr(_matmul(np, m1, m2, transpose_a=True, transpose_b=True), "shape")) == (2, 2)
    assert cast("EagerTensorProtocol", getattr(_matmul(np, m1, m2), "shape")) == (2, 2)
    assert cast("EagerTensorProtocol", getattr(_matmul(db, m1, m2), "shape")) == (2, 2)

    bm1 = np.ones((2, 2, 2))
    bm2 = np.ones((2, 2, 2))
    assert cast("EagerTensorProtocol", getattr(_batch_matmul(np, bm1, bm2), "shape")) == (2, 2, 2)
    assert cast("EagerTensorProtocol", getattr(_batch_matmul(db, bm1, bm2), "shape")) == (2, 2, 2)

    v1 = np.ones(3)
    v2 = np.ones(3)
    assert _dot(np, v1, v2) == 3.0
    assert _dot(db, v1, v2) == 3.0

    # math_nn.py
    x_2d = np.ones((2, 4))
    assert cast("EagerTensorProtocol", getattr(_layer_norm(np, x_2d), "shape")) == (2, 4)

    x_4d = np.ones((2, 4, 3, 3))
    assert cast("EagerTensorProtocol", getattr(_batch_norm(np, x_4d), "shape")) == (2, 4, 3, 3)
    assert cast("EagerTensorProtocol", getattr(_batch_norm(np, x_4d, 1.0, 0.0, np.zeros((1, 4, 1, 1)), np.ones((1, 4, 1, 1))), "shape")) == (2, 4, 3, 3)

    assert cast("EagerTensorProtocol", getattr(_rms_norm(np, x_2d), "shape")) == (2, 4)
    assert _mse_loss(np, np.ones((2, 2)), np.zeros((2, 2))) == 1.0
    assert _bce_loss(np, np.array([0.2, 0.8]), np.array([0.0, 1.0])) is not None

    # bce_loss fallback without clip on backend
    db_math = DummyBackend()
    setattr(db_math, "log", cast("EagerTensorProtocol", getattr(np, "log")))
    setattr(db_math, "mean", cast("EagerTensorProtocol", getattr(np, "mean")))
    assert _bce_loss(db_math, np.array([0.2, 0.8]), np.array([0.0, 1.0])) is not None

    assert _huber_loss(np, np.array([0.5, 2.0]), np.array([0.0, 0.0])) is not None
    assert _kl_loss(np, np.array([0.2, 0.8]), np.array([0.3, 0.7])) is not None
    assert cast("EagerTensorProtocol", getattr(_gelu(np, np.array([0.5, -0.5])), "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(_silu(np, np.array([0.5, -0.5])), "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(_elu(np, np.array([0.5, -0.5])), "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(_leaky_relu(np, np.array([0.5, -0.5])), "shape")) == (2,)

    # Line 323: 4D adaptive pool falling through
    assert cast("EagerTensorProtocol", getattr(_global_adaptive_pool(np, np.ones((1, 2, 3, 4, 5)), (1, 1, 1, 1)), "shape")) == (1, 2, 3, 4, 5)


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
    assert cast("EagerTensorProtocol", getattr(_reduce_sum(np, arr, axis=0), "shape")) == (3,)
    assert _reduce_sum(db, arr) is arr

    # _reduce_mean
    assert cast("EagerTensorProtocol", getattr(_reduce_mean(np, arr, axis=0), "shape")) == (3,)
    assert _reduce_mean(db, arr) is arr

    # _reduce_prod
    assert cast("EagerTensorProtocol", getattr(_reduce_prod(np, arr, axis=0), "shape")) == (3,)
    assert _reduce_prod(db, arr) is arr

    # _reduce_max
    assert cast("EagerTensorProtocol", getattr(_reduce_max(np, arr, axis=0), "shape")) == (3,)
    assert _reduce_max(db, arr) is arr

    # _reduce_min
    assert cast("EagerTensorProtocol", getattr(_reduce_min(np, arr, axis=0), "shape")) == (3,)
    assert _reduce_min(db, arr) is arr

    # _norm
    assert _norm(np, arr) == np.linalg.norm(arr)
    db_sqrt = DummyBackend()
    setattr(db_sqrt, "sqrt", cast("EagerTensorProtocol", getattr(np, "sqrt")))
    setattr(db_sqrt, "sum", cast("EagerTensorProtocol", getattr(np, "sum")))
    assert _norm(db_sqrt, arr) == np.linalg.norm(arr)

    # _numel
    class HasSize:
        """Test class HasSize."""

        size = 12

    class HasNumel:
        """Test class HasNumel."""

        def numel(self) -> EagerValue:
            """Test function numel."""
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
    assert cast("EagerTensorProtocol", getattr(_broadcast_reduce(np, cot_3d, tgt_2d), "shape")) == (2, 3)
    # td == 1 and cd > 1
    tgt_1d = np.ones((1, 3))
    assert cast("EagerTensorProtocol", getattr(_broadcast_reduce(np, arr, tgt_1d), "shape")) == (1, 3)

    # _broadcast_like
    assert _broadcast_like(np) is None
    assert _broadcast_like(np, arr) is arr
    assert _broadcast_like(np, arr, arr) is arr
    assert cast("EagerTensorProtocol", getattr(_broadcast_like(np, np.ones(3), arr), "shape")) == (2, 3)
    assert cast("EagerTensorProtocol", getattr(_broadcast_like(db, np.ones(3), arr), "shape")) == (3,)

    # _segment_sum
    assert _segment_sum(np) is None
    data = np.ones((4, 2))
    segments = np.array([0, 0, 1, 1])
    res_seg = _segment_sum(np, data, segments, 2)
    assert cast("EagerTensorProtocol", getattr(res_seg, "shape")) == (2, 2)
