"""Core abstractions and logic definitions for test_metrics_primitives.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.stats.descriptive import confusion_matrix, trapezoidal_integral
from ml_switcheroo_compiler.tracing import global_tracing_state

if TYPE_CHECKING:
    pass


def test_metrics_primitives_eager_backends() -> None:
    """Verify trapezoidal_integral and confusion_matrix on NumPy eager backend."""
    device = Device(DeviceType.CPU)
    y_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    labels_data = np.array([0, 1, 2], dtype=np.int32)
    preds_data = np.array([0, 2, 2], dtype=np.int32)
    with ConfigContext(eager_mode=True, backend="numpy"):
        backend_cls = BackendRegistry.get("numpy")
        y = Tensor(backend_cls.array(y_data), TensorConfig((2, 3), DType.Float32, device))
        labels = Tensor(backend_cls.array(labels_data), TensorConfig((3,), DType.Int32, device))
        preds = Tensor(backend_cls.array(preds_data), TensorConfig((3,), DType.Int32, device))
        res_trapz = trapezoidal_integral(y, dx=1.0, axis=-1)
        res_cm = confusion_matrix(labels, preds, num_classes=3)
        trapz_data = np.asarray(getattr(res_trapz, "data"))
        cm_data = np.asarray(getattr(res_cm, "data"))
        assert trapz_data.shape == (2,)
        assert cm_data.shape == (3, 3)


def test_metrics_primitives_tracing() -> None:
    """Test the metrics primitives tracing behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            y = Tensor("dummy_y", TensorConfig((2, 3), DType.Float32, device))
            labels = Tensor("dummy_l", TensorConfig((3,), DType.Int32, device))
            preds = Tensor("dummy_p", TensorConfig((3,), DType.Int32, device))
            res_trapz = trapezoidal_integral(y, dx=1.0, axis=-1)
            assert res_trapz is not None
            res_cm = confusion_matrix(labels, preds, num_classes=3)
            assert res_cm is not None
        finally:
            global_tracing_state.stop_tracing()
