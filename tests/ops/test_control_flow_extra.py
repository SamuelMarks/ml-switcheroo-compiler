"""Module docstring."""

import numpy as np
from unittest.mock import MagicMock
from ml_switcheroo_compiler.ops.control_flow import scan
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device
import ml_switcheroo_compiler.core.config as config
import ml_switcheroo_compiler.ops.control_flow as cf


def test_scan_tuple_y(monkeypatch: object) -> None:
    """Docstring."""
    config.eager_mode = True

    def f(carry: int, x: int) -> object:
        """Docstring."""
        return carry + 1, (x + 1, x + 2)

    init = Tensor(np.array(0.0), shape=(), dtype=DType.Float32, device=Device("cpu"))
    xs = Tensor(np.array([1.0, 2.0]), shape=(2,), dtype=DType.Float32, device=Device("cpu"))

    mock_backend = MagicMock()
    mock_array = MagicMock()
    mock_array.shape = (2, 2)
    mock_backend.execute_op.return_value = mock_array
    monkeypatch.setattr(cf, "get_active_backend", lambda: mock_backend)

    carry, y = scan(f, init, xs)
    assert y is not None
