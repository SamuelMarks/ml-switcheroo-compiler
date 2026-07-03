"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.nn.clip_grad import clip_grad_norm


def test_clip_grad_norm() -> object:
    """Function docstring."""
    config.eager_mode = True
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = ops.array(x_data)

    clipped, total_norm = clip_grad_norm(x, max_norm=1.0)
    assert total_norm is not None


def test_clip_grad_norm_list() -> object:
    """Function docstring."""
    config.eager_mode = True
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = ops.array(x_data)
    y = ops.array(x_data * 2)

    clipped, total_norm = clip_grad_norm([x, y], max_norm=1.0)
    assert len(clipped) == 2
    assert total_norm is not None


def test_clip_grad_norm_inf() -> object:
    """Function docstring."""
    config.eager_mode = True
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = ops.array(x_data)
    y = ops.array(x_data * 2)

    clipped, total_norm = clip_grad_norm([x, y], max_norm=1.0, norm_type=float("inf"))
    assert len(clipped) == 2
    assert total_norm is not None


def test_clip_grad_norm_pnorm() -> object:
    """Function docstring."""
    config.eager_mode = True
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = ops.array(x_data)
    y = ops.array(x_data * 2)

    clipped, total_norm = clip_grad_norm([x, y], max_norm=1.0, norm_type=3.0)
    assert len(clipped) == 2
    assert total_norm is not None


def test_clip_grad_norm_empty() -> object:
    """Function docstring."""
    clipped, total_norm = clip_grad_norm([], max_norm=1.0)
    assert len(clipped) == 0
    assert total_norm is not None
