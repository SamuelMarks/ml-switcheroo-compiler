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
