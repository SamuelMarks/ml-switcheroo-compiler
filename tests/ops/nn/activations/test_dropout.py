"""Behavioral tests for dropout, alpha dropout, spatial dropouts, and activity regularization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.dropout as do
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from tests.testing_utils import MockTestStub


class _ShapedDummy(MockTestStub):
    """Operand exposing an explicit `.shape` tuple."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize shaped dummy operand.

        Args:
            shape (tuple[int, ...]): Desired shape tuple.
        """
        self.shape = shape


class _UnshapedDummy(MockTestStub):
    """Operand lacking `.shape`."""

    val: int = 123


def test_dropout_op_classes_infer_shape() -> None:
    """Verify shape inference across `Dropout`, `AlphaDropout`, `ActivityRegularization`, and 1D/2D/3D variants."""
    t_in = Tensor(np.ones((2, 2), dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))

    assert do.Dropout().infer_shape(t_in) == (2, 2)
    assert do.AlphaDropout().infer_shape(t_in) == (2, 2)
    assert do.ActivityRegularization().infer_shape(t_in) == (2, 2)
    assert do.ActivityRegularization().infer_shape((t_in, t_in)) == (t_in, t_in)

    unshaped = _UnshapedDummy()
    assert do.Dropout1d().infer_shape(_ShapedDummy((2, 4))) == (2, 4)
    assert do.Dropout1d().infer_shape(unshaped) == ()
    assert do.Dropout2d().infer_shape(_ShapedDummy((2, 4, 4))) == (2, 4, 4)
    assert do.Dropout2d().infer_shape(unshaped) == ()
    assert do.Dropout3d().infer_shape(_ShapedDummy((2, 4, 4, 4))) == (2, 4, 4, 4)
    assert do.Dropout3d().infer_shape(unshaped) == ()


def test_dropout_eager_and_tracing_ir_emission() -> None:
    """Verify eager execution and symbolic `TracerTape` IR node emission for all dropout functions."""
    cfg = do.DropoutConfig(noise_shape=(2, 2), training=True, seed=123)
    assert cfg.training is True

    t_eager = Tensor(np.ones((2, 2), dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    backend = MagicMock()
    backend.execute_op.return_value = np.ones((2, 2), dtype=np.float32)

    with (
        ConfigContext(eager_mode=True),
        patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=backend),
        patch("ml_switcheroo_compiler.ops.nn.dropout.get_active_backend", return_value=backend),
    ):
        r1 = do.dropout1d(t_eager, p=0.5, training=True)
        r2 = do.dropout2d(t_eager, p=0.5, training=True)
        r3 = do.dropout3d(t_eager, p=0.5, training=True)
        assert isinstance(r1, Tensor) and r1.shape == (2, 2)
        assert isinstance(r2, Tensor) and r2.shape == (2, 2)
        np.testing.assert_allclose(np.asarray(r3), np.ones((2, 2), dtype=np.float32))

    t_sym = Tensor(ProxyTensor("drop_in", (2, 2), "float32"), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("dropout_trace")
        try:
            s_dp = do.dropout(t_sym, rate=0.2, config=cfg)
            s_adp = do.alpha_dropout(t_sym, rate=0.2, config=cfg)
            s_act = do.activity_regularization(t_sym, l1=0.01, l2=0.02)
            s_1d = do.dropout1d(t_sym, p=0.5)
            s_2d = do.dropout2d(t_sym, p=0.5)
            s_3d = do.dropout3d(t_sym, p=0.5)
        finally:
            tape.stop_tracing()

    for out in (s_dp, s_adp, s_act, s_1d, s_2d, s_3d):
        assert isinstance(out, Tensor)

    op_types = {node.op_type for node in traced_graph.nodes.values()}
    assert {"Dropout", "AlphaDropout", "ActivityRegularization", "Dropout1d", "Dropout2d", "Dropout3d"} <= op_types
