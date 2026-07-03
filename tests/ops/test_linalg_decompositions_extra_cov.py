"""Module docstring."""

from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions import polar, tridiagonal
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_tridiagonal_infer_shape() -> object:
    """Function docstring."""
    device = Device("cpu")
    t1 = Tensor(np.ones((4, 4)), TensorConfig((4, 4), "float32", device))
    t_empty = Tensor(np.ones((4, 0)), TensorConfig((4, 0), "float32", device))

    td = tridiagonal()
    res1 = td.infer_shape(t1)
    assert res1[1] == (3,)
    res2 = td.infer_shape(t_empty)
    assert res2[1] == (4,)


def test_polar_eager() -> object:
    """Function docstring."""
    device = Device("cpu")
    t1 = Tensor(np.ones((2, 2)), TensorConfig((2, 2), "float32", device))
    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            mock_backend.return_value.execute_op.return_value = (np.ones((2, 2)), np.ones((2, 2)))
            u, p = polar(t1)
            assert u is not None and p is not None


def test_polar_tracing() -> object:
    """Function docstring."""
    device = Device("cpu")
    t1 = Tensor(np.ones((2, 2)), TensorConfig((2, 2), "float32", device))
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        with patch("ml_switcheroo_compiler.ops.linalg.decompositions.misc._emit_linalg_node") as mock_emit:
            polar(t1)
            global_tracing_state.stop_tracing()
            assert mock_emit.called
