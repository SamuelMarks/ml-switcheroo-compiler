import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def make_tensor(data):
    return Tensor(np.array(data, dtype=np.float32), TensorConfig(np.shape(data), DType.Float32, Device("cpu")))


@pytest.mark.parametrize("backend_name", ["numpy", "mlx", "pytorch", "jax"])
@settings(max_examples=10, deadline=None)
@given(
    val1=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_higher_order_equivalence(backend_name, val1):
    pass
