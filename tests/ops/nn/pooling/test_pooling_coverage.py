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
