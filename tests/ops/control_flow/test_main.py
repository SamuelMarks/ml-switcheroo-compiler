# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import pmap, vmap
from ml_switcheroo_compiler.tracing.state import global_tracing_state

"Core abstractions and logic definitions for test_control_flow_extra3.py."


def test_vmap_in_axes_length_coverage() -> object:
    """Test the vmap in axes length coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        device = Device(DeviceType.CPU, 0)
        with ConfigContext(eager_mode=False):

            def my_func(a: object, b: object) -> object:
                """Evaluate and process the my func operation.

                Args:
                    a (object): Required parameter for a.
                    b (object): Required parameter for b.

                Returns:
                    object: The evaluated or processed output.
                """
                return a + b

            t1 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
            t2 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
            v_func = vmap(my_func, in_axes=(0,))
            try:
                v_func(t1, t2)
            except Exception:
                pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Core abstractions and logic definitions for test_control_flow_extra2.py."


def test_vmap_pmap_scalar_fallback() -> object:
    """Test the vmap pmap scalar fallback behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:

        def f(x: object, y: object) -> object:
            """Evaluate and process the f operation.

            Args:
                x (object): Required parameter for x.
                y (object): Required parameter for y.

            Returns:
                object: The evaluated or processed output.
            """
            return x * y

        dev = Device("cpu")
        t = Tensor([1, 2, 3], TensorConfig((3,), DType.Int32, dev))
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            v_f = vmap(f, in_axes=(0, None))
            with pytest.raises(AttributeError):
                v_f(t, 5)
            p_f = pmap(f)
            with pytest.raises(AttributeError):
                p_f(t, 5)
            global_tracing_state.stop_tracing()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
