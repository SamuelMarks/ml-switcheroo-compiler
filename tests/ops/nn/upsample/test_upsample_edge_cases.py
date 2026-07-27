from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.upsample_ops import _resolve_scale_factor, _upsample_dispatch, _upsample_resolve_size, pixel_shuffle, upsample, upsample_bicubic, upsample_bilinear, upsample_nearest


def test_upsample_brute():
    config.backend = "numpy"
    config.eager_mode = True

    t_2d = Tensor(np.random.rand(2, 3, 4, 4).astype(np.float32), TensorConfig((2, 3, 4, 4), "float32", "cpu"))
    t_1d = Tensor(np.random.rand(2, 3, 4).astype(np.float32), TensorConfig((2, 3, 4), "float32", "cpu"))

    _resolve_scale_factor(t_2d, 2.0, 2)
    _resolve_scale_factor(t_2d, (2.0, 2.0), 2)
    with pytest.raises(ValueError):
        _resolve_scale_factor(t_2d, (2.0,), 2)

    class DummyTensorDynamic:
        shape = (1, 1, 1)

    _resolve_scale_factor(DummyTensorDynamic(), 2.0, 2)

    _upsample_resolve_size(t_2d, 8, None)
    _upsample_resolve_size(t_2d, (8, 8), None)
    _upsample_resolve_size(t_2d, None, 2.0)
    with pytest.raises(ValueError):
        _upsample_resolve_size(t_2d, None, None)
    with pytest.raises(ValueError):
        _upsample_resolve_size(t_2d, 8, 2.0)

    _upsample_dispatch(t_2d, "nearest", 8, None, False)
    _upsample_dispatch(t_2d, "linear", 8, None, False)
    _upsample_dispatch(t_2d, "bilinear", 8, None, False)
    _upsample_dispatch(t_2d, "bicubic", 8, None, False)
    _upsample_dispatch(t_2d, "unknown", 8, None, False)

    _upsample_dispatch(t_1d, "nearest", 8, None, False)
    _upsample_dispatch(t_1d, "linear", 8, None, False)
    _upsample_dispatch(t_1d, "bicubic", 8, None, False)
    _upsample_dispatch(t_1d, "unknown", 8, None, False)

    upsample(t_2d, size=8)

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:

        class DummyBackend:
            def execute_op(self, *args, **kwargs):
                return np.zeros((2, 3, 8, 8)).astype(np.float32)

            def array(self, x):
                return x

        mock_backend.return_value = DummyBackend()

        pixel_shuffle(t_2d, 2)
        upsample_nearest(t_2d, size=8)
        upsample_bilinear(t_2d, size=8)
        upsample_bicubic(t_2d, size=8)

    # Test tracing
    config.eager_mode = False
    with patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node") as mock_emit:
        pixel_shuffle(t_2d, 2)
        upsample_nearest(t_2d, size=8)
        upsample_bilinear(t_2d, size=8)
        upsample_bicubic(t_2d, size=8)
