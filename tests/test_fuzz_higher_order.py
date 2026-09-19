"""Fuzzing higher-order derivatives."""

import pytest

pytest.importorskip("hypothesis")
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import hvp


@pytest.mark.parametrize("backend_name", ["numpy", "mlx", "pytorch", "jax"])
@settings(max_examples=5, deadline=None)
@given(
    val1=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False),
)
def test_higher_order_equivalence(backend_name, val1):
    """Test higher-order equivalence."""
    from ml_switcheroo_compiler.core.config import config

    prev = config.backend
    try:
        config.backend = backend_name
        backend = get_active_backend()
        arr_x = backend.asarray(np.array([val1], dtype=np.float32))
        arr_v = backend.asarray(np.array([1.0], dtype=np.float32))

        # Test f(x) = x^3
        def f(x):
            return ops.multiply(x, ops.multiply(x, x))

        x = Tensor(arr_x, TensorConfig((1,), DType.Float32, Device("cpu")))
        v = Tensor(arr_v, TensorConfig((1,), DType.Float32, Device("cpu")))

        _, tan = hvp(f, (x,), (v,))
        computed = float(np.asarray(backend.asarray(tan))[0])

        expected = 6.0 * val1
        np.testing.assert_allclose(computed, expected, rtol=1e-2, atol=1e-2)
    finally:
        config.backend = prev
