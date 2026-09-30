"""Tests for test_rnn_cell_coverage."""

from __future__ import annotations

import numpy as np

import ml_switcheroo_compiler.ops.nn.rnn_cell as rnn_cell_module
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_simple_rnn_cell() -> None:
    """Test simple RNN cell evaluation with and without bias."""
    from ml_switcheroo_compiler.core.config import config

    orig_eager = config.eager_mode
    config.eager_mode = True
    try:
        cfg_in = TensorConfig(shape=(1, 2), dtype=DType.Float32, device="cpu")
        inputs = Tensor(np.array([[0.5, -0.2]], dtype=np.float32), cfg_in)

        cfg_st = TensorConfig(shape=(1, 3), dtype=DType.Float32, device="cpu")
        state = (Tensor(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), cfg_st),)

        cfg_k = TensorConfig(shape=(2, 3), dtype=DType.Float32, device="cpu")
        kernel = Tensor(np.ones((2, 3), dtype=np.float32) * 0.1, cfg_k)

        cfg_rk = TensorConfig(shape=(3, 3), dtype=DType.Float32, device="cpu")
        recurrent_kernel = Tensor(np.eye(3, dtype=np.float32) * 0.2, cfg_rk)

        # Branch 1: bias is None
        h_no_bias, next_state_no_bias = rnn_cell_module.simple_rnn_cell(
            inputs=inputs,
            state=state,
            kernel=kernel,
            recurrent_kernel=recurrent_kernel,
            bias=None,
        )
        assert isinstance(h_no_bias, Tensor)
        assert len(next_state_no_bias) == 1
        assert h_no_bias is next_state_no_bias[0]
        assert np.asarray(h_no_bias.data).shape == (1, 3)

        # Branch 2: bias is not None
        cfg_b = TensorConfig(shape=(1, 3), dtype=DType.Float32, device="cpu")
        bias = Tensor(np.array([[0.1, 0.1, 0.1]], dtype=np.float32), cfg_b)
        h_bias, next_state_bias = rnn_cell_module.simple_rnn_cell(
            inputs=inputs,
            state=state,
            kernel=kernel,
            recurrent_kernel=recurrent_kernel,
            bias=bias,
        )
        assert isinstance(h_bias, Tensor)
        assert len(next_state_bias) == 1
        assert h_bias is next_state_bias[0]
        assert np.asarray(h_bias.data).shape == (1, 3)
    finally:
        config.eager_mode = orig_eager
