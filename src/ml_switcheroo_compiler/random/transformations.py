"""Random transformations."""

from __future__ import annotations

import numpy as np
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.state import _emit_random_node


def shuffle(
    key: object,
    x: object,
    axis: int = 0,
) -> object:
    """Shuffles a tensor along a given axis.

    Args:
        key (object): The PRNG key.
        x (object): The input tensor.
        axis (int): The axis to shuffle.

    Returns:
        object: The shuffled tensor.
    """
    if config.eager_mode:
        x_data = getattr(x, "data", x)
        x_np = np.asarray(x_data)

        key_data = getattr(key, "data", key)
        seed = [int(v) for v in np.asarray(key_data).ravel()] if np.ndim(key_data) > 0 else None
        rng = np.random.default_rng(seed)

        shuffled = np.copy(x_np)

        if axis == 0:
            rng.shuffle(shuffled)
        else:
            # Move axis to 0, shuffle, then move back
            shuffled = np.moveaxis(shuffled, axis, 0)
            rng.shuffle(shuffled)
            shuffled = np.moveaxis(shuffled, 0, axis)

        return Tensor(
            shuffled,
            TensorConfig(shuffled.shape, getattr(x, "dtype", None), config.default_device),
        )

    return _emit_random_node(
        "RandomShuffle",
        [key, x],
        getattr(x, "shape", ()),
        getattr(x, "dtype", None),
        {"axis": axis},
    )
