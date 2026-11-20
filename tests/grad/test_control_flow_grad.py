"""Tests for test_control_flow_grad."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.control_flow import cond, while_loop
from tests.testing_utils import ensure_tensor


def test_cond_grad() -> None:
    """Test function test_cond_grad."""

    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """

        def true_fn(x: Tensor) -> Tensor:
            """Test function true_fn.

            Args:
                x (Tensor): Test parameter `x`.

            Returns:
                Tensor: Computed output value from `true_fn`.
            """
            return (x * x) * x

        def false_fn(x: Tensor) -> Tensor:
            """Test function false_fn.

            Args:
                x (Tensor): Test parameter `x`.

            Returns:
                Tensor: Computed output value from `false_fn`.
            """
            return x * x

        c = x > 0.0
        return ensure_tensor(cond(c, lambda: true_fn(x), lambda: false_fn(x)))

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    out, vjp_fn = vjp(f, x)
    np.testing.assert_allclose(np.asarray(ensure_tensor(out).numpy()), [8.0])
    # Test backward pass graph construction
    gx = vjp_fn(
        Tensor(
            np.array([1.0], dtype=np.float32),
            TensorConfig((1,), DType.Float32, Device("cpu")),
        )
    )
    assert len(gx) == 1 and float(np.asarray(gx[0])) == 0.0


def test_while_loop_grad() -> None:
    """Test function test_while_loop_grad."""

    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """

        def cond_fn(val: tuple[Tensor]) -> Tensor:
            """Test function cond_fn.

            Args:
                val (tuple[Tensor]): Test parameter `val`.

            Returns:
                Tensor: Computed output value from `cond_fn`.
            """
            return val[0] < 10.0

        def body_fn(val: tuple[Tensor]) -> tuple[Tensor]:
            """Test function body_fn.

            Args:
                val (tuple[Tensor]): Test parameter `val`.

            Returns:
                tuple[Tensor]: Computed output value from `body_fn`.
            """
            return (val[0] + x,)

        res = while_loop(cond_fn, body_fn, (x,))
        return res[0]

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    out, vjp_fn = vjp(f, x)
    np.testing.assert_allclose(np.asarray(ensure_tensor(out).numpy()), [10.0])
    gx = vjp_fn(
        Tensor(
            np.array([1.0], dtype=np.float32),
            TensorConfig((1,), DType.Float32, Device("cpu")),
        )
    )
    np.testing.assert_allclose(np.asarray(gx[0]), [1.0])
