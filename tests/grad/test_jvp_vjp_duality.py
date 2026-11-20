"""Tests for symbolic JVP/VJP adjoint duality validation."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.grad.testing import check_jvp_vjp_duality
from ml_switcheroo_compiler.ops import linalg

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_jvp_vjp_duality_linear_ops() -> None:
    """Test JVP/VJP duality for linear operations: Matmul and Multiply.

    Returns:
        None
    """
    with ConfigContext(eager_mode=True):
        from ml_switcheroo_compiler.core.tensor import Tensor
        from tests.testing_utils import NumpyTestArray, ensure_tensor

        def matmul_fn(
            x: Tensor | NumpyTestArray,
            y: Tensor | NumpyTestArray,
        ) -> BackendArg:
            """Matrix multiplication wrapper.

            Args:
                x: First matrix.
                y: Second matrix.

            Returns:
                BackendArg: Product matrix.
            """
            return linalg.matmul(ensure_tensor(x), ensure_tensor(y))

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y_val = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=np.float32)
        check_jvp_vjp_duality(matmul_fn, (x_val, y_val))

        def scale_fn(
            x: Tensor | NumpyTestArray,
        ) -> Tensor | NumpyTestArray:
            """Scaling operation.

            Args:
                x (Tensor | NumpyTestArray): Input matrix.

            Returns:
                Tensor | NumpyTestArray: Scaled matrix.
            """
            return x * 2.5

        check_jvp_vjp_duality(scale_fn, (x_val,))
        assert np.allclose(ensure_ndarray(scale_fn(x_val.astype(np.float64))), x_val * 2.5)


def test_jvp_vjp_duality_transcendental_ops() -> None:
    """Test JVP/VJP duality for polynomial and composite operations.

    Returns:
        None
    """
    from ml_switcheroo_compiler.core.tensor import Tensor
    from tests.testing_utils import NumpyTestArray

    with ConfigContext(eager_mode=True):

        def poly_fn(
            x: Tensor | NumpyTestArray,
        ) -> Tensor | NumpyTestArray:
            """Polynomial composite function.

            Args:
                x (Tensor | NumpyTestArray): Input array.

            Returns:
                Tensor | NumpyTestArray: Evaluated array.
            """
            return (((x * x) * x) - ((x * x) * 2.0)) + (x * 0.5)

        arr = np.array([0.2, -0.5, 0.8, -1.2], dtype=np.float32)
        check_jvp_vjp_duality(poly_fn, (arr,))
        expected = (arr**3) - 2.0 * (arr**2) + 0.5 * arr
        assert np.allclose(ensure_ndarray(poly_fn(arr.astype(np.float64))), expected)


def test_jvp_vjp_duality_reductions() -> None:
    """Test JVP/VJP duality for composite scaling and linear combinations.

    Returns:
        None
    """
    from ml_switcheroo_compiler.core.tensor import Tensor
    from tests.testing_utils import NumpyTestArray

    with ConfigContext(eager_mode=True):

        def linear_comb_fn(
            x: Tensor | NumpyTestArray,
        ) -> Tensor | NumpyTestArray:
            """Linear combination function.

            Args:
                x (Tensor | NumpyTestArray): Input array.

            Returns:
                Tensor | NumpyTestArray: Linear combination.
            """
            return (x * 1.5) + (x * 0.75)

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        check_jvp_vjp_duality(linear_comb_fn, (arr,))
        assert np.allclose(ensure_ndarray(linear_comb_fn(arr.astype(np.float64))), arr * 2.25)


def test_jvp_vjp_duality_failure_raises() -> None:
    """Test that check_jvp_vjp_duality validates duality identity with arbitrary vectors.

    Returns:
        None
    """
    from ml_switcheroo_compiler.core.tensor import Tensor
    from tests.testing_utils import NumpyTestArray

    with ConfigContext(eager_mode=True):

        def simple_fn(
            x: Tensor | NumpyTestArray,
        ) -> Tensor | NumpyTestArray:
            """Simple multiplication.

            Args:
                x (Tensor | NumpyTestArray): Input array.

            Returns:
                Tensor | NumpyTestArray: Scaled array.
            """
            return x * 2.0

        arr = np.array([1.0, 2.0], dtype=np.float32)
        tan = np.array([0.5, 0.5], dtype=np.float32)
        cot = np.array([1.0, 1.0], dtype=np.float32)
        check_jvp_vjp_duality(simple_fn, (arr,), tangents=(tan,), cotangents=cot)
        assert np.allclose(ensure_ndarray(simple_fn(arr.astype(np.float64))), np.array([2.0, 4.0], dtype=np.float32))


def test_jvp_vjp_duality_tensor_primals_and_failure() -> None:
    """Test JVP/VJP duality with Tensor input and verify failure raises SwitcherooError."""
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.errors import SwitcherooError
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    with ConfigContext(eager_mode=True):
        # 1. Primal as a Tensor with tangents=None (hits line 153)
        t = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))

        def double_fn(x: Tensor) -> BackendArg:
            """Double the input tensor.

            Args:
                x (Tensor): Input tensor.

            Returns:
                BackendArg: Scaled tensor.
            """
            return x * 2.0

        check_jvp_vjp_duality(double_fn, (t,))

        # 2. Strict negative tolerance forces diff > tol, raising SwitcherooError (hits lines 196-197)
        with pytest.raises(SwitcherooError, match="JVP/VJP duality check failed"):
            check_jvp_vjp_duality(double_fn, (t,), atol=-1.0, rtol=-1.0)


def test_check_numerical_grads_coverage() -> None:
    """Test full branch and statement coverage for check_numerical_grads."""
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.errors import SwitcherooError
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.options import GradCheckOptions
    from ml_switcheroo_compiler.grad.testing import check_numerical_grads

    # 1. Scalar float input
    def square_fn(x: float) -> float:
        """Square scalar input.

        Args:
            x (float): Scalar float value.

        Returns:
            float: Squared value.
        """
        return x * x

    check_numerical_grads(square_fn, (3.0,))

    # 2. Tensor input
    def tensor_scale_fn(x: Tensor) -> BackendArg:
        """Scale tensor input.

        Args:
            x (Tensor): Input tensor.

        Returns:
            BackendArg: Scaled tensor.
        """
        return x * 2.5

    t = Tensor(np.array([1.0, 2.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    check_numerical_grads(tensor_scale_fn, (t,))

    # 3. Mismatch error branch with custom GradCheckOptions
    opts = GradCheckOptions(step=1e-4, atol=-1.0, rtol=-1.0)
    with pytest.raises(SwitcherooError, match="Gradient check failed for argument"):
        check_numerical_grads(square_fn, (3.0,), options=opts)
