"""Tests for test_dropout_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.nn.dropout as dropout_mod
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_dropout_ops_full_coverage() -> None:
    """Test full coverage for dropout operations and configurations."""
    cfg = dropout_mod.DropoutConfig(noise_shape=(2, 2), training=True, seed=123)
    assert cfg.training is True

    t_in = Tensor(np.ones((2, 2), dtype=np.float32), TensorConfig((2, 2), "float32", "cpu"))

    dp_op = dropout_mod.Dropout()
    assert dp_op.infer_shape(t_in) == (2, 2)

    alpha_op = dropout_mod.AlphaDropout()
    assert alpha_op.infer_shape(t_in) == (2, 2)

    act_op = dropout_mod.ActivityRegularization()
    assert act_op.infer_shape(t_in) == (2, 2)
    assert act_op.infer_shape((t_in, t_in)) == (t_in, t_in)

    with patch("ml_switcheroo_compiler.ops.nn.dropout.get_op") as mock_get_op:
        mock_inst = MagicMock()
        mock_inst.return_value = t_in
        mock_get_op.return_value = MagicMock(return_value=mock_inst)

        assert dropout_mod.dropout(t_in, rate=0.2, config=cfg) is t_in
        assert dropout_mod.alpha_dropout(t_in, rate=0.2, config=cfg) is t_in
        assert dropout_mod.activity_regularization(t_in, l1=0.01, l2=0.02) is t_in

    dp1d_op = dropout_mod.Dropout1d()
    assert dp1d_op.infer_shape(DummyWithShape((2, 4))) == (2, 4)
    assert dp1d_op.infer_shape(DummyWithoutShape()) == ()

    dp2d_op = dropout_mod.Dropout2d()
    assert dp2d_op.infer_shape(DummyWithShape((2, 4, 4))) == (2, 4, 4)
    assert dp2d_op.infer_shape(DummyWithoutShape()) == ()

    dp3d_op = dropout_mod.Dropout3d()
    assert dp3d_op.infer_shape(DummyWithShape((2, 4, 4, 4))) == (2, 4, 4, 4)
    assert dp3d_op.infer_shape(DummyWithoutShape()) == ()

    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        with (
            patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_be_reg,
            patch("ml_switcheroo_compiler.ops.nn.dropout.get_active_backend") as mock_be_mod,
        ):
            backend = MagicMock()
            backend.execute_op.return_value = np.ones((2, 2), dtype=np.float32)
            mock_be_reg.return_value = backend
            mock_be_mod.return_value = backend

            res_1d = dropout_mod.dropout1d(t_in, p=0.5, training=True)
            assert isinstance(res_1d, Tensor)

            res_2d = dropout_mod.dropout2d(t_in, p=0.5, training=True)
            assert isinstance(res_2d, Tensor)

            res_3d = dropout_mod.dropout3d(t_in, p=0.5, training=True)
            assert res_3d is not None

        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.nn.dropout._emit_shape_node", return_value=t_in) as mock_emit:
            assert dropout_mod.dropout1d(t_in, p=0.5) is t_in
            assert dropout_mod.dropout2d(t_in, p=0.5) is t_in
            assert dropout_mod.dropout3d(t_in, p=0.5) is t_in
            assert mock_emit.call_count == 3
    finally:
        config.eager_mode = original_eager
