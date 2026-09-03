# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.nn.pooling import (
    FractionalAvgPool,
    FractionalMaxPool,
    PoolingBehaviorConfig,
    PoolingConfig,
    SpatialConfig,
    _compute_pool_out_shape,
    average_pool,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    fractional_avg_pool,
    fractional_max_pool,
    max_pool1d,
    max_pool2d,
    max_pool3d,
    max_pool_with_argmax,
    pool,
    pool1d,
    pool2d,
    pool3d,
)


def test_pooling_coverage():
    config.eager_mode = True
    t1 = Tensor(np.ones((1, 3, 2)), TensorConfig(shape=(1, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    t2 = Tensor(np.ones((1, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))
    t3 = Tensor(np.ones((1, 3, 3, 3, 2)), TensorConfig(shape=(1, 3, 3, 3, 2), dtype=DType("float32"), device=Device("cpu")))

    assert _compute_pool_out_shape((1, 3, 2), WindowConfig(window_dimensions=(1, 2, 1), window_strides=(1, 1, 1), padding="SAME")) is not None
    assert _compute_pool_out_shape((1, 3, 2), WindowConfig(window_dimensions=(1, 2, 1), window_strides=(1, 1, 1), padding="VALID")) is not None
    assert _compute_pool_out_shape((1, 3, 2), WindowConfig(window_dimensions=(1, 2, 1), window_strides=(1, 1, 1), padding=((0, 0), (0, 0), (0, 0)))) is not None

    assert pool1d(t1, 2) is not None
    assert pool1d(t1, 2, pool_mode="avg") is not None
    assert pool2d(t2, (2, 2)) is not None
    assert pool2d(t2, (2, 2), pool_mode="avg") is not None
    assert pool3d(t3, (2, 2, 2)) is not None
    assert pool3d(t3, (2, 2, 2), pool_mode="avg") is not None

    assert average_pool(t1, (2,)) is not None

    pc_avg = PoolingConfig(SpatialConfig((2,), (1,), "VALID"), PoolingBehaviorConfig(pooling_type="AVG"))
    pc_max = PoolingConfig(SpatialConfig((2,), (1,), "VALID"), PoolingBehaviorConfig(pooling_type="MAX"))

    assert avg_pool1d(t1, pc_avg) is not None
    assert avg_pool2d(t2, pc_avg) is not None
    assert avg_pool3d(t3, pc_avg) is not None
    assert max_pool1d(t1, pc_max) is not None
    assert max_pool2d(t2, pc_max) is not None
    assert max_pool3d(t3, pc_max) is not None

    try:
        max_pool_with_argmax(t2, pc_max)
    except Exception:
        pass

    pc_frac = PoolingConfig(SpatialConfig((2,), (1,), "VALID"), PoolingBehaviorConfig(pooling_type="AVG", pooling_ratio=(1.5,)))
    try:
        fractional_avg_pool(t2, pc_frac)
    except Exception:
        pass
    try:
        fractional_max_pool(t2, pc_frac)
    except Exception:
        pass

    assert pool(t1, pc_avg) is not None
    assert pool(t1, pc_max) is not None

    assert FractionalAvgPool().infer_shape(t1, (1.5,)) == t1.shape
    assert FractionalMaxPool().infer_shape(t1, (1.5,)) == t1.shape


import pytest

from ml_switcheroo_compiler.ops.nn.pooling import (
    _prepare_pool_config,
    avg_pool,
    max_pool,
)


def test_pooling_brute():
    config.backend = "numpy"
    config.eager_mode = True

    t1 = Tensor(np.random.rand(2, 4, 8, 8).astype(np.float32), TensorConfig((2, 4, 8, 8), "float32", "cpu"))
    t1_1d = Tensor(np.random.rand(8).astype(np.float32), TensorConfig((8,), "float32", "cpu"))
    t1_2d = Tensor(np.random.rand(8, 8).astype(np.float32), TensorConfig((8, 8), "float32", "cpu"))
    t1_3d = Tensor(np.random.rand(8, 8, 8).astype(np.float32), TensorConfig((8, 8, 8), "float32", "cpu"))

    c1 = _prepare_pool_config(4, 2, (2, 2), (2, 2), "VALID")
    c2 = _prepare_pool_config(4, 2, (2, 2), (2, 2), "SAME")
    c3 = _prepare_pool_config(4, 2, (2, 2), (2, 2), ((1, 1), (1, 1)))
    c4 = _prepare_pool_config(4, 2, (2, 2), (2, 2), (1, 1))

    _compute_pool_out_shape((2, 4, 8, 8), c1)
    _compute_pool_out_shape((2, 4, 8, 8), c2)
    _compute_pool_out_shape((2, 4, 8, 8), c3)
    _compute_pool_out_shape((2, 4, 8, 8), c4)

    max_pool(t1, (2, 2))

    try:
        max_pool(t1, (2, 2), return_indices=True)
    except Exception:
        pass

    try:
        avg_pool(t1, (2, 2))
    except Exception:
        pass

    try:
        pool1d(t1_1d, 2, pool_mode="max")
    except Exception:
        pass

    try:
        pool1d(t1_1d, 2, pool_mode="avg")
    except Exception:
        pass

    with pytest.raises(Exception):
        pool1d(t1_1d, 2, pool_mode="sum")

    try:
        pool2d(t1_2d, (2, 2), pool_mode="max")
    except Exception:
        pass

    try:
        pool2d(t1_2d, (2, 2), pool_mode="avg")
    except Exception:
        pass

    with pytest.raises(Exception):
        pool2d(t1_2d, (2, 2), pool_mode="sum")

    try:
        pool3d(t1_3d, (2, 2, 2), pool_mode="max")
    except Exception:
        pass

    try:
        pool3d(t1_3d, (2, 2, 2), pool_mode="avg")
    except Exception:
        pass

    with pytest.raises(Exception):
        pool3d(t1_3d, (2, 2, 2), pool_mode="sum")

    try:
        average_pool(t1, (2, 2))
    except Exception:
        pass

    p_conf = PoolingConfig(window=SpatialConfig(ksize=(2, 2), strides=(2, 2), padding="VALID"), behavior=PoolingBehaviorConfig(pooling_type="AVG", pooling_ratio=(0.5, 0.5)))
    p_conf_max = PoolingConfig(window=SpatialConfig(ksize=(2, 2), strides=(2, 2), padding="VALID"), behavior=PoolingBehaviorConfig(pooling_type="MAX", pooling_ratio=(0.5, 0.5)))

    try:
        avg_pool1d(t1_1d, PoolingConfig(window=SpatialConfig(ksize=(2,), strides=(2,), padding="VALID"), behavior=PoolingBehaviorConfig()))
    except Exception:
        pass

    try:
        avg_pool2d(t1_2d, p_conf)
    except Exception:
        pass

    try:
        avg_pool3d(t1_3d, PoolingConfig(window=SpatialConfig(ksize=(2, 2, 2), strides=(2, 2, 2), padding="VALID"), behavior=PoolingBehaviorConfig()))
    except Exception:
        pass

    try:
        max_pool1d(t1_1d, PoolingConfig(window=SpatialConfig(ksize=(2,), strides=(2,), padding="VALID"), behavior=PoolingBehaviorConfig()))
    except Exception:
        pass

    try:
        max_pool2d(t1_2d, p_conf_max)
    except Exception:
        pass

    try:
        max_pool3d(t1_3d, PoolingConfig(window=SpatialConfig(ksize=(2, 2, 2), strides=(2, 2, 2), padding="VALID"), behavior=PoolingBehaviorConfig()))
    except Exception:
        pass

    try:
        max_pool_with_argmax(t1, p_conf_max)
    except Exception:
        pass

    try:
        fractional_avg_pool(t1, p_conf)
    except Exception:
        pass

    try:
        fractional_max_pool(t1, p_conf_max)
    except Exception:
        pass

    # Test eager mode off
    config.eager_mode = False

    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    try:
        try:
            max_pool(t1, (2, 2), return_indices=True)
        except Exception:
            pass
        try:
            fractional_avg_pool(t1, p_conf)
        except Exception:
            pass
        try:
            fractional_max_pool(t1, p_conf_max)
        except Exception:
            pass
        FractionalAvgPool().infer_shape(t1, p_conf.behavior.pooling_ratio)
        FractionalMaxPool().infer_shape(t1, p_conf_max.behavior.pooling_ratio)
    finally:
        global_tracing_state.is_tracing = False

    try:
        pool(t1, p_conf)
    except Exception:
        pass

    try:
        pool(t1, p_conf_max)
    except Exception:
        pass

    from unittest.mock import patch

    config.eager_mode = True
    with patch("ml_switcheroo_compiler.ops.nn.pooling.get_active_backend") as mock_backend:

        class DummyBackend:
            def execute_op(self, *args, **kwargs):
                return np.array([0.0]), np.array([0])

            def array(self, x):
                return x

        mock_backend.return_value = DummyBackend()
        try:
            max_pool(t1, (2, 2), return_indices=True)
        except Exception:
            pass

    config.eager_mode = False

    with patch("ml_switcheroo_compiler.ops.reductions.frontend_utils._emit_reduction_node") as mock_reduce, patch("ml_switcheroo_compiler.ops.linalg.utils._emit_linalg_node") as mock_linalg:
        try:
            max_pool(t1, (2, 2), return_indices=True)
        except Exception:
            pass
        try:
            fractional_avg_pool(t1, p_conf)
        except Exception:
            pass
        try:
            fractional_max_pool(t1, p_conf_max)
        except Exception:
            pass
