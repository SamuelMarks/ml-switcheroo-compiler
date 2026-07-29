# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.dropout as do
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig

gru_mod = sys.modules["ml_switcheroo_compiler.ops.nn.gru"]


def test_dropout_coverage():
    config.eager_mode = True
    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))

    assert do.Dropout().infer_shape(t) == (2, 2)
    assert do.AlphaDropout().infer_shape(t) == (2, 2)
    assert do.ActivityRegularization().infer_shape(t) == (2, 2)
    assert do.ActivityRegularization().infer_shape([t]) == [t]

    class DummyShape:
        shape = (2, 2)

    assert do.Dropout1d().infer_shape(DummyShape()) == (2, 2)
    assert do.Dropout2d().infer_shape(DummyShape()) == (2, 2)
    assert do.Dropout3d().infer_shape(DummyShape()) == (2, 2)

    with patch("ml_switcheroo_compiler.ops.nn.dropout.get_op") as mock_get_op:
        config.eager_mode = False
        mock_op = mock_get_op.return_value.return_value
        mock_op.return_value = "dropped"
        assert do.dropout(t) == "dropped"
        assert do.alpha_dropout(t) == "dropped"
        assert do.activity_regularization(t) == "dropped"
        config.eager_mode = True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        with patch("ml_switcheroo_compiler.ops.nn.dropout.get_active_backend") as mock_backend_2:
            mock_backend_2.return_value.execute_op.return_value = np.zeros((2, 2))
            assert do.dropout1d(t) is not None
            assert do.dropout2d(t) is not None
            assert do.dropout3d(t) is not None

        config.eager_mode = False
        import ml_switcheroo_compiler.ops.nn.dropout as d_mod

        def dummy_emit(*args, **kwargs):
            return "emitted"

        d_mod._emit_shape_node = dummy_emit
        assert do.dropout1d(t) == "emitted"
        assert do.dropout2d(t) == "emitted"
        assert do.dropout3d(t) == "emitted"
        config.eager_mode = True
