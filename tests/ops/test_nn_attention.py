import numpy as np
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.ops.nn.attention_utils import (
    rope,
    sinusoidal_positional_encoding,
    alibi_mask,
)
from ml_switcheroo_compiler.core.config import config


def test_rope():
    config.eager_mode = True
    x_data = np.random.randn(2, 4, 8).astype(np.float32)
    x = ops.array(x_data)

    y = rope(x, dim=8)
    assert y is not None


def test_sinusoidal_positional_encoding():
    config.eager_mode = True
    pe = sinusoidal_positional_encoding(seq_len=10, dim=16)
    assert pe is not None


def test_alibi_mask():
    config.eager_mode = True
    mask = alibi_mask(seq_len=10, num_heads=4)
    assert mask is not None
