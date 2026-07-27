# ruff: noqa: D103
"""Tests for loss extras."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.loss import kld, kullback_leibler_divergence, logcosh, mape, mape_loss, msle, msle_loss


def test_loss_extras() -> None:
    backend = get_active_backend()
    dev = Device("cpu")
    t1 = Tensor(backend.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, dev))
    t2 = Tensor(backend.array([1.5, 2.5]), TensorConfig((2,), DType.Float32, dev))

    # Just test that they execute
    config.eager_mode = True
    assert msle_loss(t1, t2) is not None
    assert mape_loss(t1, t2) is not None

    # Aliases
    assert msle is msle_loss
    assert mape is mape_loss
    assert kld is kullback_leibler_divergence
    assert logcosh is not None
