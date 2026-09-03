# ruff: noqa: E501

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.upsample_ops import pixel_shuffle, upsample, upsample_bicubic, upsample_bilinear, upsample_nearest


def test_upsample_coverage():
    config.eager_mode = True
    t_2d = Tensor(np.ones((1, 1, 2, 2)), TensorConfig(shape=(1, 1, 2, 2), dtype=DType("float32"), device=Device("cpu")))
    t_3d = Tensor(np.ones((1, 1, 2, 2, 2)), TensorConfig(shape=(1, 1, 2, 2, 2), dtype=DType("float32"), device=Device("cpu")))

    assert upsample(t_2d, size=(4, 4)) is not None
    assert upsample(t_2d, scale_factor=2.0) is not None
    assert upsample(t_3d, scale_factor=2.0, mode="linear") is not None
    assert upsample(t_3d, scale_factor=2.0, mode="bicubic") is not None

    with pytest.raises(ValueError):
        upsample(t_2d)
    with pytest.raises(ValueError):
        upsample(t_2d, size=(4, 4), scale_factor=2.0)
    with pytest.raises(ValueError):
        upsample(t_2d, scale_factor=(2.0, 2.0, 2.0))

    with pytest.raises(Exception):
        pixel_shuffle(t_2d, 2)
    upsample_nearest(t_2d, scale_factor=2.0)
    upsample_bilinear(t_2d, scale_factor=2.0)
    upsample_bicubic(t_2d, scale_factor=2.0)

    original_eager = config.eager_mode
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        assert pixel_shuffle(t_2d, 2) is not None
        assert upsample_nearest(t_2d, scale_factor=2.0) is not None
        assert upsample_bilinear(t_2d, scale_factor=2.0) is not None
        assert upsample_bicubic(t_2d, scale_factor=2.0) is not None
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = False


from unittest.mock import patch

from ml_switcheroo_compiler.ops.nn.upsample_ops import _resolve_scale_factor, _upsample_dispatch, _upsample_resolve_size


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
