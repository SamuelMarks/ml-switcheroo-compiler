"""Behavioral tests for NumPy eager vecdot, orthogonal polynomials, permutation, image/CSV IO, and debug checks."""

from __future__ import annotations

import numpy as np
import pytest

import ml_switcheroo_compiler.ops as ops_mod
import tests.backends.numpy.eager._math_advanced_test_ns as mod
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, coerce_operand_list, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, require_callable


class _LinalgVecdotBackend(MockTestStub):
    """Backend stub exposing `linalg.vecdot`."""

    class linalg:
        """Nested `linalg` namespace providing `vecdot`."""

        @staticmethod
        def vecdot(x: BackendArg, y: BackendArg, axis: int = -1) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            """Compute inner product along `axis`.

            Args:
                x (BackendArg): First input vector.
                y (BackendArg): Second input vector.
                axis (int): Reduction axis.

            Returns:
                np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Inner product array.
            """
            return ensure_ndarray(np.sum(np.conj(np.asarray(x)) * np.asarray(y), axis=axis))


class _TopLevelVecdotBackend(MockTestStub):
    """Backend stub exposing top-level `vecdot` without `linalg.vecdot`."""

    @staticmethod
    def vecdot(x: BackendArg, y: BackendArg, axis: int = -1) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute inner product along `axis`.

        Args:
            x (BackendArg): First input vector.
            y (BackendArg): Second input vector.
            axis (int): Reduction axis.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Inner product array.
        """
        return ensure_ndarray(np.sum(np.conj(np.asarray(x)) * np.asarray(y), axis=axis))


class _FallbackVecdotBackend(MockTestStub):
    """Backend stub lacking `vecdot` to exercise `iscomplexobj`/`conj`/`sum` fallback."""

    @staticmethod
    def iscomplexobj(x: BackendArg) -> bool:
        """Check whether `x` has a complex dtype.

        Args:
            x (BackendArg): Input array.

        Returns:
            bool: True when `x` is complex-valued.
        """
        return bool(np.iscomplexobj(np.asarray(x)))

    @staticmethod
    def conj(x: BackendArg) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute complex conjugate of `x`.

        Args:
            x (BackendArg): Input array.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Complex-conjugated array.
        """
        return ensure_ndarray(np.conj(np.asarray(x)))

    @staticmethod
    def sum(x: BackendArg, axis: int = -1) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Sum `x` along `axis`.

        Args:
            x (BackendArg): Input array.
            axis (int): Reduction axis.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Reduced array.
        """
        return ensure_ndarray(np.sum(np.asarray(x), axis=axis))


class _DescriptiveAndRemBackend(MockTestStub):
    """Backend stub exposing custom `descriptive` and `rem` methods."""

    @staticmethod
    def descriptive(x: BackendArg) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Return summary moments `[mean, var, std]`.

        Args:
            x (BackendArg): Input array.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Summary vector.
        """
        arr = np.asarray(x)
        return np.array([float(np.mean(arr)), float(np.var(arr)), float(np.std(arr))], dtype=np.float64)

    @staticmethod
    def rem(x: BackendArg, y: BackendArg) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Return elementwise remainder.

        Args:
            x (BackendArg): Dividend array.
            y (BackendArg): Divisor array.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Elementwise remainder.
        """
        return ensure_ndarray(np.remainder(np.asarray(x), np.asarray(y)))


class _CustomDescriptiveMeta(type):
    """Metaclass returning summary moments array when its class is called."""

    def __call__(cls, x: BackendArg) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute `[mean, var, std]` directly on class invocation.

        Args:
            x (BackendArg): Input array.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.generic]]: Summary moments.
        """
        arr = np.asarray(x, dtype=np.float64)
        return np.array([float(np.mean(arr)), float(np.var(arr)), float(np.std(arr))], dtype=np.float64)


class _CustomDescriptiveOp(metaclass=_CustomDescriptiveMeta):
    """Non-`OpDef` callable class exercising `ops` class-dispatch branch."""


class _FailingOpClass:
    """Non-`OpDef` class raising `ValueError` to exercise fallback recovery."""

    def __new__(cls, *args: BackendArg, **kwargs: BackendArg) -> _FailingOpClass:
        """Raise `ValueError` to verify fallback to NumPy implementation.

        Args:
            *args (BackendArg): Positional operands.
            **kwargs (BackendArg): Keyword operands.

        Returns:
            _FailingOpClass: Never returns normally.

        Raises:
            ValueError: Always raised to trigger fallback execution.
        """
        raise ValueError("simulated op constructor failure")


def test_vecdot_dispatch_and_complex_conjugate_fallback() -> None:
    """Verify `Vecdot` dispatches across `linalg.vecdot`, top-level `vecdot`, and real/complex fallback paths."""
    x_real = np.array([1.0, 2.0, 3.0])
    y_real = np.array([4.0, 5.0, 6.0])

    res_linalg = mod._np_vecdot(_LinalgVecdotBackend(), x_real, y_real)
    np.testing.assert_allclose(np.asarray(res_linalg), 32.0)

    res_top = mod._np_vecdot(_TopLevelVecdotBackend(), x_real, y_real)
    np.testing.assert_allclose(np.asarray(res_top), 32.0)

    res_real_fb = mod._np_vecdot(_FallbackVecdotBackend(), x_real, y_real)
    np.testing.assert_allclose(np.asarray(res_real_fb), 32.0)

    x_cplx = np.array([1.0 + 2.0j, 3.0 - 1.0j])
    y_cplx = np.array([2.0 - 1.0j, 1.0 + 1.0j])
    res_cplx_fb = mod._np_vecdot(_FallbackVecdotBackend(), x_cplx, y_cplx)
    np.testing.assert_allclose(np.asarray(res_cplx_fb), 2.0 - 1.0j)


def test_chebyshev_polynomials_and_recurrence_degrees() -> None:
    """Verify Chebyshev `T_n(x)` and `U_n(x)` evaluate exact values across negative, 0, 1, and higher degrees."""
    x_val = np.array([0.5, -0.5])

    np.testing.assert_allclose(mod._np_chebyshev_polynomial_t(np, x_val, np.array(-1)), np.zeros(2))
    np.testing.assert_allclose(mod._np_chebyshev_polynomial_t(np, x_val, np.array(0)), np.ones(2))
    np.testing.assert_allclose(mod._np_chebyshev_polynomial_t(np, x_val, np.array(1)), x_val)

    # T_2(0.5) = 2*(0.25) - 1 = -0.5; T_3(-0.5) = 4*(-0.125) - 3*(-0.5) = 1.0
    t_higher = mod._np_chebyshev_polynomial_t(np, x_val, np.array([2, 3]))
    np.testing.assert_allclose(t_higher, np.array([-0.5, 1.0]))

    # U_2(0.5) = 4*(0.25) - 1 = 0.0; U_3(0.5) = 8*(0.125) - 4*(0.5) = -1.0
    u_higher = mod._np_chebyshev_polynomial_u(np, np.array([0.5, 0.5]), np.array([2, 3]))
    np.testing.assert_allclose(u_higher, np.array([0.0, -1.0]))

    empty_rec = mod._poly_recurrence(
        np.array([], dtype=int),
        np.array([], dtype=np.float64),
        1.0,
        lambda v: v,
        lambda idx, v, p1, p2: 2.0 * v * p1 - p2,
    )
    assert empty_rec.size == 0


def test_random_permutation_scalar_and_vector_inputs() -> None:
    """Verify `RandomPermutation` permutes both 0-D integer counts and 1-D element arrays."""
    perm_scalar = ensure_ndarray(mod._np_randompermutation(np, np.array(5)))
    assert sorted(perm_scalar.tolist()) == [0, 1, 2, 3, 4]

    perm_vec = ensure_ndarray(mod._np_randompermutation(np, np.array([10, 20, 30, 40])))
    assert sorted(perm_vec.tolist()) == [10, 20, 30, 40]


def test_png_csv_tensor_and_buffer_io_operations() -> None:
    """Verify PNG encode/decode, CSV decode, tensor serialize/parse, and `Frombuffer` numerical round-trips."""
    encode_png = numpy_eager_registry.get("EncodePng")
    src_img = np.zeros((2, 3, 3), dtype=np.uint8)
    png_bytes = require_callable(encode_png)(np, src_img)

    decoded_rgb = ensure_ndarray(mod._np_decode_image_camel(np, png_bytes))
    assert decoded_rgb.shape == (2, 3, 3)
    assert decoded_rgb.dtype == np.uint8

    decoded_rgba = ensure_ndarray(mod._np_decode_image_camel(np, png_bytes, channels=4))
    assert decoded_rgba.shape == (2, 3, 4)
    assert ensure_ndarray(mod._np_decode_image_camel(np)).size == 0

    with pytest.raises(RuntimeError, match="Eager execution failed"):
        mod._np_decode_image_camel(np, "not_png_bytes")
    with pytest.raises(RuntimeError, match="Eager execution failed"):
        mod._np_decode_image_camel(np, np.array(b"invalid_png_header"))

    csv_payload = ("1.5,2.5" + chr(10) + "3.5,4.5").encode("utf-8")
    csv_cols = coerce_operand_list(mod._np_decode_csv(np, np.array(csv_payload), record_defaults=[0.0, 0.0]))
    np.testing.assert_allclose(ensure_ndarray(csv_cols[0]), np.array([1.5, 3.5]))
    np.testing.assert_allclose(ensure_ndarray(csv_cols[1]), np.array([2.5, 4.5]))

    with pytest.raises(ValueError, match="Failed to parse CSV"):
        mod._np_decode_csv(np, [b"not_a_float"], record_defaults=[0.0])

    raw_tensor = np.array([1.25, -3.5], dtype=np.float32)
    serialized = mod._np_serialize_tensor_camel(np, raw_tensor)
    parsed = ensure_ndarray(mod._np_parse_tensor_camel(np, serialized, out_type=np.float32))
    np.testing.assert_allclose(parsed, raw_tensor)

    with pytest.raises(RuntimeError, match="Eager execution failed"):
        mod._np_parse_tensor_camel(np, np.array(b"corrupt_pickle"))

    buf_arr = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    from_buf = mod._np_frombuffer(np, buf_arr.tobytes(), dtype=np.float64)
    assert from_buf is not None
    np.testing.assert_allclose(from_buf, buf_arr)
    assert mod._np_frombuffer(np) is None


def test_rrelu_debug_checks_confusion_matrix_and_distributions() -> None:
    """Verify `Rrelu`, `DebugInfs`, `DebugNans`, `ConfusionMatrix`, and `Distributions` behavior."""
    rrelu_in = np.array([3.0, -2.0], dtype=np.float64)
    rrelu_out = ensure_ndarray(mod._np_rrelu(np, rrelu_in, lower=0.1, upper=0.25))
    assert float(rrelu_out[0]) == 3.0
    assert -0.5 <= float(rrelu_out[1]) <= -0.2
    assert mod._np_rrelu(np) is None

    finite = np.array([1.0, -2.0])
    np.testing.assert_array_equal(ensure_ndarray(mod._np_debuginfs(np, finite)), finite)
    np.testing.assert_array_equal(ensure_ndarray(mod._np_debugnans(np, finite)), finite)
    with pytest.raises(ValueError, match="Infinity found"):
        mod._np_debuginfs(np, np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="NaN found"):
        mod._np_debugnans(np, np.array([1.0, np.nan]))

    cm = mod._np_confusion_matrix(np, np.array([0, 1, 1]), np.array([0, 0, 1]))
    assert cm is not None
    np.testing.assert_array_equal(cm, np.array([[1, 0], [1, 1]]))
    assert mod._np_confusion_matrix(np, np.array([0, 1])) is None

    dist = mod._np_distributions(np, np.array([1.0, 2.0, 3.0, 4.0]))
    assert dist is not None
    dist_map = dist
    assert int(np.sum(ensure_ndarray(dist_map["counts"]))) == 4
    assert mod._np_distributions(np) is None


def test_descriptive_and_rem_dispatch_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `descriptive` and `rem` exercise backend methods, `ops` dispatch, error recovery, and NumPy fallbacks.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for overriding `ops` symbols.
    """
    x = np.array([1.0, 3.0], dtype=np.float64)
    expected_moments = np.array([2.0, 1.0, 1.0], dtype=np.float64)

    np.testing.assert_allclose(ensure_ndarray(mod._np_descriptive(np, x)), expected_moments)
    np.testing.assert_allclose(ensure_ndarray(mod._np_descriptive(np)), np.zeros(3))
    np.testing.assert_allclose(
        ensure_ndarray(mod._np_descriptive(_DescriptiveAndRemBackend(), x)),
        expected_moments,
    )

    np.testing.assert_allclose(ensure_ndarray(mod._np_rem(np, np.array([7.0, 8.0]), np.array([3.0, 5.0]))), [1.0, 3.0])
    assert mod._np_rem(np, np.array([7.0])) is None
    np.testing.assert_allclose(
        ensure_ndarray(mod._np_rem(_DescriptiveAndRemBackend(), np.array([7.0]), np.array([4.0]))),
        [3.0],
    )

    monkeypatch.setattr(ops_mod, "descriptive", _CustomDescriptiveOp, raising=False)
    np.testing.assert_allclose(ensure_ndarray(mod._np_descriptive(np, x)), expected_moments)

    monkeypatch.setattr(ops_mod, "descriptive", _FailingOpClass, raising=False)
    np.testing.assert_allclose(ensure_ndarray(mod._np_descriptive(np, x)), expected_moments)

    monkeypatch.setattr(ops_mod, "rem", _FailingOpClass, raising=False)
    np.testing.assert_allclose(ensure_ndarray(mod._np_rem(np, np.array([9.0]), np.array([4.0]))), [1.0])
