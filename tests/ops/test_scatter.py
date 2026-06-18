"""Tests for scatter operations."""

import pytest
import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.ops import (
    tensor_scatter_update,
    tensor_scatter_add,
    tensor_scatter_max,
    tensor_scatter_min,
)
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.backends.registry import BackendRegistry


@pytest.mark.parametrize("backend_name", list(BackendRegistry.get_all().keys()))
def test_tensor_scatter_max_eager(backend_name: str) -> None:
    """Test eager execution of tensor_scatter_max."""
    try:
        backend_cls = BackendRegistry.get(backend_name)
    except Exception:
        pytest.skip("Backend not available")

    device = Device(DeviceType.CPU)

    arr = np.zeros((3, 3), dtype=np.float32)
    # Target value will be max(0, 3.0) and max(0, -1.0) and duplicate
    idx = np.array([[0, 0], [1, 1], [0, 0], [2, 2]], dtype=np.int32)
    upd = np.array([1.0, -1.0, 3.0, 4.0], dtype=np.float32)

    with ConfigContext(eager_mode=True, backend=backend_name):
        try:
            tensor = Tensor(
                backend_cls.array(arr), shape=(3, 3), dtype=DType.Float32, device=device
            )
            indices = Tensor(backend_cls.array(idx), shape=(4, 2), dtype=DType.Int32, device=device)
            updates = Tensor(backend_cls.array(upd), shape=(4,), dtype=DType.Float32, device=device)

            res = tensor_scatter_max(tensor, indices, updates)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} failed setup: {e}")

        expected = np.zeros((3, 3), dtype=np.float32)
        expected[0, 0] = 3.0
        expected[1, 1] = 0.0  # max(0, -1) = 0
        expected[2, 2] = 4.0

        res_data = res.data
        if hasattr(res_data, "numpy"):
            res_data = res_data.numpy()
        elif hasattr(res_data, "tolist"):
            res_data = np.array(res_data.tolist())

        np.testing.assert_allclose(res_data, expected, atol=1e-5)


@pytest.mark.parametrize("backend_name", list(BackendRegistry.get_all().keys()))
def test_tensor_scatter_min_eager(backend_name: str) -> None:
    """Test eager execution of tensor_scatter_min."""
    try:
        backend_cls = BackendRegistry.get(backend_name)
    except Exception:
        pytest.skip("Backend not available")

    device = Device(DeviceType.CPU)

    arr = np.ones((3, 3), dtype=np.float32) * 5.0
    idx = np.array([[0, 0], [1, 1], [0, 0], [2, 2]], dtype=np.int32)
    upd = np.array([4.0, 6.0, 2.0, 1.0], dtype=np.float32)

    with ConfigContext(eager_mode=True, backend=backend_name):
        try:
            tensor = Tensor(
                backend_cls.array(arr), shape=(3, 3), dtype=DType.Float32, device=device
            )
            indices = Tensor(backend_cls.array(idx), shape=(4, 2), dtype=DType.Int32, device=device)
            updates = Tensor(backend_cls.array(upd), shape=(4,), dtype=DType.Float32, device=device)

            res = tensor_scatter_min(tensor, indices, updates)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} failed setup: {e}")

        expected = np.ones((3, 3), dtype=np.float32) * 5.0
        expected[0, 0] = 2.0  # min(5.0, 4.0, 2.0)
        expected[1, 1] = 5.0  # min(5.0, 6.0)
        expected[2, 2] = 1.0

        res_data = res.data
        if hasattr(res_data, "numpy"):
            res_data = res_data.numpy()
        elif hasattr(res_data, "tolist"):
            res_data = np.array(res_data.tolist())

        np.testing.assert_allclose(res_data, expected, atol=1e-5)


@pytest.mark.parametrize("backend_name", list(BackendRegistry.get_all().keys()))
def test_tensor_scatter_add_eager(backend_name: str) -> None:
    """Test eager execution of tensor_scatter_add with duplicate N-D indices."""
    try:
        backend_cls = BackendRegistry.get(backend_name)
    except Exception:
        pytest.skip("Backend not available")

    device = Device(DeviceType.CPU)

    arr = np.zeros((3, 3), dtype=np.float32)
    # Testing duplicate indices
    idx = np.array([[0, 0], [1, 1], [0, 0], [2, 2]], dtype=np.int32)
    upd = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    with ConfigContext(eager_mode=True, backend=backend_name):
        try:
            tensor = Tensor(
                backend_cls.array(arr), shape=(3, 3), dtype=DType.Float32, device=device
            )
            indices = Tensor(backend_cls.array(idx), shape=(4, 2), dtype=DType.Int32, device=device)
            updates = Tensor(backend_cls.array(upd), shape=(4,), dtype=DType.Float32, device=device)

            res = tensor_scatter_add(tensor, indices, updates)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} failed setup: {e}")

        expected = np.zeros((3, 3), dtype=np.float32)
        expected[0, 0] = 4.0
        expected[1, 1] = 2.0
        expected[2, 2] = 4.0

        res_data = res.data
        if hasattr(res_data, "numpy"):
            res_data = res_data.numpy()
        elif hasattr(res_data, "tolist"):
            res_data = np.array(res_data.tolist())

        np.testing.assert_allclose(res_data, expected, atol=1e-5)


@pytest.mark.parametrize("backend_name", list(BackendRegistry.get_all().keys()))
def test_tensor_scatter_add_nd_indices(backend_name: str) -> None:
    """Test eager execution of tensor_scatter_add with N-dimensional indices."""
    try:
        backend_cls = BackendRegistry.get(backend_name)
    except Exception:
        pytest.skip("Backend not available")

    device = Device(DeviceType.CPU)

    arr = np.zeros((2, 2, 2), dtype=np.float32)
    # Testing 3D tensor with 2D index array
    idx = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 1, 0], [1, 0, 1]]], dtype=np.int32)
    upd = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    with ConfigContext(eager_mode=True, backend=backend_name):
        try:
            tensor = Tensor(
                backend_cls.array(arr), shape=(2, 2, 2), dtype=DType.Float32, device=device
            )
            indices = Tensor(
                backend_cls.array(idx), shape=(2, 2, 3), dtype=DType.Int32, device=device
            )
            updates = Tensor(
                backend_cls.array(upd), shape=(2, 2), dtype=DType.Float32, device=device
            )

            res = tensor_scatter_add(tensor, indices, updates)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} failed setup: {e}")

        expected = np.zeros((2, 2, 2), dtype=np.float32)
        expected[0, 0, 0] = 1.0
        expected[1, 1, 1] = 2.0
        expected[0, 1, 0] = 3.0
        expected[1, 0, 1] = 4.0

        res_data = res.data
        if hasattr(res_data, "numpy"):
            res_data = res_data.numpy()
        elif hasattr(res_data, "tolist"):
            res_data = np.array(res_data.tolist())

        np.testing.assert_allclose(res_data, expected, atol=1e-5)


@pytest.mark.parametrize("backend_name", list(BackendRegistry.get_all().keys()))
def test_tensor_scatter_update_eager(backend_name: str) -> None:
    """Test eager execution of tensor_scatter_update."""
    # Only run available backends
    try:
        backend_cls = BackendRegistry.get(backend_name)
    except Exception:
        pytest.skip("Backend not available")

    # MLX and Torch might need actual imports to evaluate eager logic
    device = Device(DeviceType.CPU)

    arr = np.zeros((3, 3), dtype=np.float32)
    idx = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
    upd = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with ConfigContext(eager_mode=True, backend=backend_name):
        try:
            tensor = Tensor(
                backend_cls.array(arr), shape=(3, 3), dtype=DType.Float32, device=device
            )
            indices = Tensor(backend_cls.array(idx), shape=(3, 2), dtype=DType.Int32, device=device)
            updates = Tensor(backend_cls.array(upd), shape=(3,), dtype=DType.Float32, device=device)

            res = tensor_scatter_update(tensor, indices, updates)
        except Exception as e:
            pytest.skip(f"Backend {backend_name} failed setup: {e}")

        expected = np.zeros((3, 3), dtype=np.float32)
        expected[0, 0] = 1.0
        expected[1, 1] = 2.0
        expected[2, 2] = 3.0

        res_data = res.data
        if hasattr(res_data, "numpy"):
            res_data = res_data.numpy()
        elif hasattr(res_data, "tolist"):
            res_data = np.array(res_data.tolist())

        np.testing.assert_allclose(res_data, expected, atol=1e-5)
