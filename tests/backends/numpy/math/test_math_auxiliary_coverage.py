"""Tests for test_math_auxiliary_coverage."""

from __future__ import annotations

import types

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_auxiliary as aux_module
from ml_switcheroo_compiler.core.dtype import DType


def test_math_auxiliary_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test auxiliary math operations directly and through eager registry.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        None
    """
    # ConstantOfShape
    c_arr = aux_module._np_constant_of_shape(np, (2, 2), value=3.0)
    assert np.allclose(c_arr, np.full((2, 2), 3.0))
    reg_cos = numpy_eager_registry.get("ConstantOfShape")
    assert np.allclose(reg_cos(np, (2, 2), value=3.0), np.full((2, 2), 3.0))

    # ReduceWindow
    monkeypatch.setattr(aux_module, "_reduce_window", lambda *args, **kwargs: np.array([42.0]))
    rw_res = aux_module._np_reduce_window(np, np.ones((4, 4)))
    assert rw_res[0] == 42.0
    reg_rw = numpy_eager_registry.get("ReduceWindow")
    assert reg_rw is not None

    # TestEagerOp
    assert np.allclose(aux_module._np_test_eager_op(np), [1.0, 2.0, 3.0])
    assert np.allclose(numpy_eager_registry.get("TestEagerOp")(np), [1.0, 2.0, 3.0])

    # Unknown
    assert aux_module._np_unknown(np) == 0.0
    assert numpy_eager_registry.get("Unknown")(np) == 0.0

    # Rand with dtype branches
    r_bf = aux_module._np_rand(np, 2, 3, dtype="bfloat16")
    assert r_bf.dtype == np.float32

    r_f8 = aux_module._np_rand(np, 2, 3, dtype="float8_e4m3")
    assert r_f8.dtype == np.float32

    r_i4 = aux_module._np_rand(np, 2, 3, dtype="int4")
    assert r_i4.dtype == np.int8

    r_std = aux_module._np_rand(np, 2, 3, dtype=np.float64)
    assert r_std.dtype == np.float64

    mock_rand_mod = types.SimpleNamespace(
        random=types.SimpleNamespace(rand=lambda *a: np.array([0.5])),
        array=lambda a: a,
    )
    r_str = aux_module._np_rand(mock_rand_mod, 1, dtype="int64")
    assert r_str.dtype == np.int64

    reg_rand = numpy_eager_registry.get("Rand")
    assert reg_rand(np, 2, 2).shape == (2, 2)

    # IsNonDecreasing branches
    mock_diff_mod = types.SimpleNamespace(
        size=lambda a: np.size(a),
        diff=lambda a: a[1:] - a[:-1],
        all=lambda a: bool(all(x for x in a.flat)),
        array=lambda a: np.array(a),
    )
    assert aux_module._np_is_non_decreasing(mock_diff_mod, np.array([5]))
    assert aux_module._np_is_non_decreasing(mock_diff_mod, np.array([1, 2, 2, 3]))
    assert not aux_module._np_is_non_decreasing(mock_diff_mod, np.array([1, 3, 2]))
    reg_ind = numpy_eager_registry.get("IsNonDecreasing")
    assert reg_ind(mock_diff_mod, np.array([1, 2, 3]))

    # IsStrictlyIncreasing branches
    assert aux_module._np_is_strictly_increasing(mock_diff_mod, np.array([5]))
    assert aux_module._np_is_strictly_increasing(mock_diff_mod, np.array([1, 2, 3]))
    assert not aux_module._np_is_strictly_increasing(mock_diff_mod, np.array([1, 2, 2]))
    reg_isi = numpy_eager_registry.get("IsStrictlyIncreasing")
    assert reg_isi(mock_diff_mod, np.array([1, 2, 3]))

    # L2Normalize
    mat = np.array([[3.0, 4.0]])
    normed = aux_module._np_l2_normalize(np, mat, axis=-1)
    assert np.allclose(normed, [[0.6, 0.8]])
    reg_l2 = numpy_eager_registry.get("L2Normalize")
    assert np.allclose(reg_l2(np, mat, axis=-1), [[0.6, 0.8]])

    # ReduceEuclideanNorm
    assert np.isclose(aux_module._np_reduce_euclidean_norm(np, np.array([3.0, 4.0])), 5.0)
    reg_ren = numpy_eager_registry.get("ReduceEuclideanNorm")
    assert np.isclose(reg_ren(np, np.array([3.0, 4.0])), 5.0)

    # Clamp
    assert aux_module._clamp(np, 0.0, 5.0, 2.0) == 2.0
    reg_clamp = numpy_eager_registry.get("Clamp")
    assert reg_clamp(np, 0.0, 5.0, 2.0) == 2.0

    # Logspace branches
    class SpaceConfig:
        """Mock SpaceConfig class."""

        def __init__(self) -> None:
            """Initialize mock SpaceConfig."""
            self.num = 4
            self.endpoint = True
            self.base = 10.0
            self.dtype = float
            self.axis = 0

    ls_cfg = aux_module._logspace(np, 1.0, 3.0, SpaceConfig())
    assert len(ls_cfg) == 4

    ls_normal = aux_module._logspace(np, 1.0, 3.0, num=4)
    assert len(ls_normal) == 4
    reg_ls = numpy_eager_registry.get("Logspace")
    assert len(reg_ls(np, 1.0, 3.0, num=4)) == 4

    # FromBuffer
    buf = b"\x00\x00\x80?\x00\x00\x00@"
    fb = aux_module._np_frombuffer(np, buf, dtype="float32")
    assert len(fb) == 2
    reg_fb = numpy_eager_registry.get("FromBuffer")
    assert len(reg_fb(np, buf, dtype="float32")) == 2

    # DType branches
    assert isinstance(aux_module._np_dtype_op(np, "float32"), DType)
    assert isinstance(aux_module._np_dtype_op(np, np.array([1.0])), DType)
    assert isinstance(aux_module._np_dtype_op(np, [1, 2]), DType)
    reg_dt = numpy_eager_registry.get("DType")
    assert isinstance(reg_dt(np, "int32"), DType)

    # Gradient, I0, BroadcastedIota
    assert aux_module._eager_Gradient(np, np.array([1.0, 2.0, 4.0])) is not None
    assert aux_module._eager_I0(np, np.array([0.0])) is not None
    assert aux_module._eager_BroadcastedIota(np, 3, (2, 3)).shape == (2, 3)

    # Gradient & I0 fallback branch
    no_grad_mod = types.SimpleNamespace()
    assert aux_module._eager_Gradient(no_grad_mod, 42) == 42
    assert aux_module._eager_I0(no_grad_mod, 42) == 42
