"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.matrix_ops import band_part, diag
from ml_switcheroo_compiler.tracing import global_tracing_state


def test_linalg_band_part_diag_coverage() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU, 0)
    t = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Int32, device))

    with ConfigContext(eager_mode=True):
        res1 = band_part(t, 1, 1)
        assert res1 is not None

        res2 = diag(t, k=0)
        assert res2 is not None

    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            res3 = band_part(t, 1, 1)
            assert res3 is not None

            res4 = diag(t, k=0)
            assert res4 is not None
        finally:
            global_tracing_state.stop_tracing()
