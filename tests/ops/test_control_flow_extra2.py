"""Module docstring."""

import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import pmap, vmap
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_vmap_pmap_scalar_fallback() -> object:
    """Function docstring."""

    def f(x: object, y: object) -> object:
        """Function docstring."""
        return x * y

    dev = Device("cpu")
    t = Tensor([1, 2, 3], TensorConfig((3,), DType.Int32, dev))

    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        v_f = vmap(f, in_axes=(0, None))

        with pytest.raises(AttributeError):
            v_f(t, 5)  # 5 is non-tensor

        p_f = pmap(f)
        with pytest.raises(AttributeError):
            p_f(t, 5)  # 5 is non-tensor
        global_tracing_state.stop_tracing()
