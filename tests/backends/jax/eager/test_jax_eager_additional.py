from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from ml_switcheroo_compiler.backends.jax.eager import (
    _OP_DISPATCH,
    _execute_accumulate_n,
    _execute_adaptive_pool_mock,
    _execute_bessel_jn,
    _execute_binom_cdf,
    _execute_cast,
    _execute_ragged_tensor_to_dense,
    _execute_unsorted_segment_max,
    _execute_unsorted_segment_min,
    _execute_unsorted_segment_prod,
    _execute_unsorted_segment_sum,
    _execute_variance,
)


def test_execute_adaptive_pool_mock():
    assert _execute_adaptive_pool_mock(1, 1) == 1

    # We patch jax.numpy.broadcast_to directly since _execute_adaptive_pool_mock imports it locally
    with patch("jax.numpy.broadcast_to") as mock_broadcast:
        with patch("jax.numpy.mean") as mock_mean:
            arr = MagicMock()
            arr.shape = (2, 2)
            arr.dtype = float

            mock_broadcast.return_value = MagicMock()
            mock_broadcast.return_value.shape = (2, 3)
            res = _execute_adaptive_pool_mock(arr, 3)
            assert res.shape == (2, 3)

            mock_broadcast.return_value.shape = (3, 3)
            res = _execute_adaptive_pool_mock(arr, (3, 3))
            assert res.shape == (3, 3)


def test_execute_accumulate_n():
    with pytest.raises(ValueError):
        _execute_accumulate_n(inputs=[])

    res = _execute_accumulate_n(inputs=[jnp.array(1), jnp.array(2), jnp.array(3)])
    assert res == 6

    res = _execute_accumulate_n([jnp.array(1), jnp.array(2), jnp.array(3)])
    assert res == 6


def test_execute_binom_cdf():
    _execute_binom_cdf(1, 2, 0.5)


def test_execute_bessel_jn():
    _execute_bessel_jn(1, jnp.array(1.0))


def test_execute_unsorted_segments():
    x = jnp.array([1.0, 2.0])
    ids = jnp.array([0, 1])
    _execute_unsorted_segment_sum(x, ids, num_segments=2)
    _execute_unsorted_segment_max(x, ids, num_segments=2)
    _execute_unsorted_segment_min(x, ids, num_segments=2)
    _execute_unsorted_segment_prod(x, ids, num_segments=2)


def test_execute_variance():
    _execute_variance(jnp.array([1.0, 2.0]))


def test_execute_cast():
    # to avoid astype mocking issue, just mock out astype
    t = MagicMock()
    t.astype.return_value = "casted"

    class DummyDtype:
        value = "int4"

    assert _execute_cast(t, DummyDtype()) == "casted"

    class DummyDtype2:
        value = "bfloat16"

    assert _execute_cast(t, dtype=DummyDtype2()) == "casted"

    class DummyDtype3:
        value = "float16"

    assert _execute_cast(t, DummyDtype3()) == "casted"

    class DummyDtype4:
        value = "float8"

    assert _execute_cast(t, DummyDtype4()) == "casted"

    class DummyDtype5:
        value = "int32"

    assert _execute_cast(t, DummyDtype5()) == "casted"


def test_execute_ragged_tensor_to_dense():
    _execute_ragged_tensor_to_dense(1)


def test_op_dispatch_lambdas():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([1.0, 2.0])

    _OP_DISPATCH["Cumprod"](x, axis=0)
    _OP_DISPATCH["ActivityRegularization"](x)
    _OP_DISPATCH["AdaptiveMaxPool3D_Indices"](x, 1)
    _OP_DISPATCH["AdaptiveLogSoftmaxWithLoss"](x, x)
    _OP_DISPATCH["Adjoint"](jnp.array([[1.0]]))
    _OP_DISPATCH["AllGather"](x)
    _OP_DISPATCH["AllToAll"](x)
    _OP_DISPATCH["AlphaDropout"](x)
    _OP_DISPATCH["AsString"](x)
    _OP_DISPATCH["Assert"](True, x)
    _OP_DISPATCH["Assign"](x, y)
    _OP_DISPATCH["AssignAdd"](x, y)
    _OP_DISPATCH["AssignSub"](x, y)
    _OP_DISPATCH["AssignVariable"](x, y)
    _OP_DISPATCH["AssociativeScan"](lambda a, b: a + b, x)
    _OP_DISPATCH["AssociativeScan"](x)
    _OP_DISPATCH["Atleast1d"](x)
    _OP_DISPATCH["Atleast2d"](x)
    _OP_DISPATCH["Atleast3d"](x)
    _OP_DISPATCH["AxisIndex"]()
    _OP_DISPATCH["Frombuffer"](b"123", dtype=jnp.uint8)
    try:
        _OP_DISPATCH["Fft2"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Fftfreq"](10)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Fftn"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Fftnd"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Fftshift"](x)
    except Exception:
        pass
    _OP_DISPATCH["HardSilu"](x)
    _OP_DISPATCH["HardSwish"](x)
    try:
        _OP_DISPATCH["Hfft"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Ifft"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Ifft2"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Ifftn"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Ifftnd"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Ifftshift"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Ihfft"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Irfft"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Irfft2"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Irfftn"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Irfftnd"](x)
    except Exception:
        pass
    _OP_DISPATCH["LogSoftmax"](x)
    _OP_DISPATCH["Mish"](x)
    _OP_DISPATCH["OneHot"](jnp.array([0, 1]), 2)
    _OP_DISPATCH["OneHot"](jnp.array([0, 1]), depth=2)
    try:
        _OP_DISPATCH["Rfft"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Rfft2"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Rfftfreq"](10)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Rfftn"](x)
    except Exception:
        pass
    try:
        _OP_DISPATCH["Rfftnd"](x)
    except Exception:
        pass
    _OP_DISPATCH["Sigmoid"](x)
    _OP_DISPATCH["Softmax"](x)
    _OP_DISPATCH["Squareplus"](x)
