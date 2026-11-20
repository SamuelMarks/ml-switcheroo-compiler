"""Unit tests for NumPy eager auxiliary mathematical operations."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.reference_math.math_auxiliary import (
    _clamp,
    _eager_BroadcastedIota,
    _eager_Gradient,
    _eager_I0,
    _logspace,
    _np_constant_of_shape,
    _np_dtype_op,
    _np_is_non_decreasing,
    _np_is_strictly_increasing,
    _np_l2_normalize,
    _np_rand,
    _np_reduce_euclidean_norm,
    _np_test_eager_op,
    _np_unknown,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_frombuffer
from ml_switcheroo_compiler.backends.eager.reference_math.window_reductions import _np_reduce_window
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.configs import SpaceConfig, WindowConfig
from tests.testing_utils import MockTestStub


class _MinimalRangeBackend(MockTestStub):
    """Minimal backend stub exposing only `arange` to exercise missing-member fallbacks."""

    def arange(self, stop: int) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
        """Generate a 1D integer range array `[0, ..., stop - 1]`.

        Args:
            stop (int): Exclusive upper integer bound.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.int64]]: Generated 1D integer array.
        """
        return np.arange(stop, dtype=np.int64)


def test_constant_of_shape() -> None:
    """Verify `_np_constant_of_shape` and `ConstantOfShape` registry fill constant tensors."""
    res = _np_constant_of_shape(np, (2, 3), value=3.5)
    np.testing.assert_allclose(np.asarray(res), np.full((2, 3), 3.5))

    reg_fn = numpy_eager_registry.get("ConstantOfShape")
    assert reg_fn is not None
    reg_res = reg_fn(np, (2, 2), value=5.0)
    np.testing.assert_allclose(np.asarray(reg_res), np.full((2, 2), 5.0))


def test_reduce_window_real_execution_and_missing_args() -> None:
    """Verify `_np_reduce_window` computes exact sliding-window sums on `np.ones((4, 4))`."""
    cfg = WindowConfig(window_dimensions=(2, 2), window_strides=(2, 2), padding="VALID")
    ones = np.ones((4, 4), dtype=np.float32)

    rw_sum = _np_reduce_window(
        np,
        ones,
        init_value=0.0,
        computation="sum",
        config=cfg,
    )
    expected = np.full((2, 2), 4.0, dtype=np.float32)
    assert np.asarray(rw_sum).shape == (2, 2)
    np.testing.assert_allclose(np.asarray(rw_sum), expected)

    reg_rw = numpy_eager_registry.get("ReduceWindow")
    assert reg_rw is not None
    reg_res = reg_rw(
        np,
        ones,
        init_value=0.0,
        computation="sum",
        config=cfg,
    )
    np.testing.assert_allclose(np.asarray(reg_res), expected)

    with pytest.raises(TypeError):
        _np_reduce_window(np, np.ones((2, 2)))


def test_test_eager_op_and_unknown() -> None:
    """Verify `_np_test_eager_op` and `_np_unknown` direct and registry execution."""
    res = _np_test_eager_op(np)
    np.testing.assert_allclose(np.asarray(res), np.array([1.0, 2.0, 3.0], dtype=np.float32))
    reg_test_op = numpy_eager_registry.get("TestEagerOp")
    assert reg_test_op is not None
    np.testing.assert_allclose(np.asarray(reg_test_op(np)), [1.0, 2.0, 3.0])

    assert _np_unknown(np) == 0.0
    reg_unk = numpy_eager_registry.get("Unknown")
    assert reg_unk is not None
    assert reg_unk(np) == 0.0


def test_rand_dtype_branches() -> None:
    """Verify `_np_rand` handles standard, bfloat16, float8, int4, string typecodes, and invalid dtypes."""
    mod = np
    r_default = _np_rand(mod, 2, 3)
    arr_def = np.asarray(r_default)
    assert arr_def.shape == (2, 3)
    assert arr_def.dtype == np.float32

    r_f64 = _np_rand(mod, 2, 3, dtype="float64")
    assert np.asarray(r_f64).dtype == np.float64

    r_bf16 = _np_rand(mod, 2, 3, dtype="bfloat16")
    assert np.asarray(r_bf16).dtype == np.float32

    r_f8 = _np_rand(mod, 2, 3, dtype="float8_e4m3")
    assert np.asarray(r_f8).dtype == np.float32

    r_i4 = _np_rand(mod, 2, 3, dtype="int4")
    assert np.asarray(r_i4).dtype == np.int8

    r_typecode = _np_rand(mod, 2, 2, dtype="<f8")
    assert np.asarray(r_typecode).dtype == np.float64

    with pytest.raises(TypeError):
        _np_rand(mod, 2, 2, dtype="fake_dtype")

    reg_rand = numpy_eager_registry.get("Rand")
    assert reg_rand is not None
    assert np.asarray(reg_rand(np, 2, 2)).shape == (2, 2)


def test_monotonicity_predicates() -> None:
    """Verify `_np_is_non_decreasing` and `_np_is_strictly_increasing` across scalar and vector inputs."""
    mod = np
    assert bool(_np_is_non_decreasing(mod, np.array([5]))) is True
    assert bool(_np_is_non_decreasing(mod, np.array([1, 2, 2, 3]))) is True
    assert bool(_np_is_non_decreasing(mod, np.array([1, 3, 2]))) is False

    reg_ind = numpy_eager_registry.get("IsNonDecreasing")
    assert reg_ind is not None
    assert bool(reg_ind(np, np.array([1, 2, 2, 3]))) is True

    assert bool(_np_is_strictly_increasing(mod, np.array([5]))) is True
    assert bool(_np_is_strictly_increasing(mod, np.array([1, 2, 3]))) is True
    assert bool(_np_is_strictly_increasing(mod, np.array([1, 2, 2]))) is False

    reg_isi = numpy_eager_registry.get("IsStrictlyIncreasing")
    assert reg_isi is not None
    assert bool(reg_isi(np, np.array([1, 2, 3]))) is True


def test_l2_normalize_and_euclidean_norm() -> None:
    """Verify `_np_l2_normalize` and `_np_reduce_euclidean_norm` produce exact Euclidean results."""
    mod = np
    mat = np.array([[3.0, 4.0]], dtype=np.float32)
    normed = _np_l2_normalize(mod, mat, axis=-1)
    np.testing.assert_allclose(np.asarray(normed), [[0.6, 0.8]], rtol=1e-6)

    reg_l2 = numpy_eager_registry.get("L2Normalize")
    assert reg_l2 is not None
    np.testing.assert_allclose(np.asarray(reg_l2(np, mat, axis=-1)), [[0.6, 0.8]], rtol=1e-6)

    vec = np.array([3.0, 4.0], dtype=np.float32)
    assert np.isclose(float(np.asarray(_np_reduce_euclidean_norm(mod, vec))), 5.0)

    reg_ren = numpy_eager_registry.get("ReduceEuclideanNorm")
    assert reg_ren is not None
    assert np.isclose(float(np.asarray(reg_ren(np, vec))), 5.0)


def test_clamp_logspace_and_frombuffer() -> None:
    """Verify `_clamp`, `_logspace` (kwargs and `SpaceConfig`), and `_np_frombuffer`."""
    mod = np
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(np.asarray(_clamp(mod, 2.0, x, 4.0)), [2.0, 2.0, 3.0, 4.0, 4.0])
    reg_clamp = numpy_eager_registry.get("Clamp")
    assert reg_clamp is not None
    assert float(np.asarray(reg_clamp(np, 0.0, 5.0, 2.0))) == 2.0

    expected_log = np.array([10.0, 100.0, 1000.0])
    ls_kw = _logspace(mod, 1.0, 3.0, num=3)
    np.testing.assert_allclose(np.asarray(ls_kw), expected_log)

    cfg = SpaceConfig(num=3, endpoint=True, base=10.0, dtype="float64", axis=0)
    ls_cfg = _logspace(mod, 1.0, 3.0, cfg)
    np.testing.assert_allclose(np.asarray(ls_cfg), expected_log)

    reg_ls = numpy_eager_registry.get("Logspace")
    assert reg_ls is not None
    np.testing.assert_allclose(np.asarray(reg_ls(np, 1.0, 3.0, num=3)), expected_log)

    buf = np.array([1.0, 2.0], dtype=np.float32).tobytes()
    np.testing.assert_allclose(np.asarray(_np_frombuffer(mod, buf, dtype="float32")), [1.0, 2.0])
    reg_fb = numpy_eager_registry.get("FromBuffer")
    assert reg_fb is not None
    np.testing.assert_allclose(np.asarray(reg_fb(np, buf, dtype="float32")), [1.0, 2.0])


def test_dtype_op_and_gradient_i0_iota_fallbacks() -> None:
    """Verify `_np_dtype_op` branches and `_eager_Gradient`, `_eager_I0`, `_eager_BroadcastedIota`."""
    mod = np
    assert _np_dtype_op(mod, "float32") == DType.Float32
    assert _np_dtype_op(mod, np.array([1.0], dtype=np.float64)) == DType.Float64
    dt_list = _np_dtype_op(mod, [1, 2])
    assert isinstance(dt_list, DType)
    assert dt_list in (DType.Int64, DType.Int32)

    reg_dt = numpy_eager_registry.get("DType")
    assert reg_dt is not None
    assert reg_dt(np, "int32") == DType.Int32

    grad_in = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    np.testing.assert_allclose(np.asarray(_eager_Gradient(mod, grad_in)), np.gradient(grad_in))

    i0_in = np.array([0.0, 1.0])
    np.testing.assert_allclose(np.asarray(_eager_I0(mod, i0_in)), np.i0(i0_in))

    iota_res = _eager_BroadcastedIota(mod, 3, (2, 3))
    np.testing.assert_allclose(np.asarray(iota_res), np.array([[0, 1, 2], [0, 1, 2]]))

    minimal_mod = _MinimalRangeBackend()
    assert _eager_Gradient(minimal_mod, 42) == 42
    assert _eager_I0(minimal_mod, 42) == 42
    np.testing.assert_array_equal(np.asarray(_eager_BroadcastedIota(minimal_mod, 3, (2, 3))), np.array([0, 1, 2]))
