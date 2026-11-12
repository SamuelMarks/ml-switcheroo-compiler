"""Tests for test_control_flow_grad."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.control_flow import cond, while_loop
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_cond_grad() -> None:
    """Test function test_cond_grad."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """

        def true_fn(x: EagerValue) -> EagerValue:
            """Test function true_fn.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `true_fn`.
            """
            return (ensure_eager_tensor(x) * ensure_eager_tensor(x)) * ensure_eager_tensor(x)

        def false_fn(x: EagerValue) -> EagerValue:
            """Test function false_fn.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `false_fn`.
            """
            return ensure_eager_tensor(x) * ensure_eager_tensor(x)

        c = ensure_eager_tensor(x) > ensure_eager_tensor(0.0)
        return ensure_callable(cond)(c, lambda: ensure_callable(true_fn)(x), lambda: ensure_callable(false_fn)(x))

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    out, vjp_fn = ensure_callable(vjp)(f, x)
    assert out is not None
    # Test backward pass graph construction
    gx = ensure_callable(vjp_fn)(ensure_callable(Tensor)(ensure_callable(np.array)([1.0], dtype=np.float32), ensure_callable(TensorConfig)((1,), DType.Float32, ensure_callable(Device)("cpu"))))
    assert gx is not None


def test_while_loop_grad() -> None:
    """Test function test_while_loop_grad."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """

        def cond_fn(val: EagerValue) -> EagerValue:
            """Test function cond_fn.

            Args:
                val: Test parameter `val`.

            Returns:
                EagerValue: Computed output value from `cond_fn`.
            """
            return ensure_eager_tensor(ensure_eager_tensor(val)[0] < 10.0)

        def body_fn(val: EagerValue) -> EagerValue:
            """Test function body_fn.

            Args:
                val: Test parameter `val`.

            Returns:
                EagerValue: Computed output value from `body_fn`.
            """
            return (ensure_eager_tensor(val)[0] + x,)

        res = ensure_callable(while_loop)(cond_fn, body_fn, (x,))
        return res[0]

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    out, vjp_fn = ensure_callable(vjp)(f, x)
    assert out is not None
    gx = ensure_callable(vjp_fn)(ensure_callable(Tensor)(ensure_callable(np.array)([1.0], dtype=np.float32), ensure_callable(TensorConfig)((1,), DType.Float32, ensure_callable(Device)("cpu"))))
    assert gx is not None
