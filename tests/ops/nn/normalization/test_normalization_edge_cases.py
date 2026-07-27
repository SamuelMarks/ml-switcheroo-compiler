import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.normalization import (
    BatchNormConfig,
    BatchNormGlobalConfig,
    LRNConfig,
    batch_norm,
    batch_norm_with_global_normalization,
    batch_normalization,
    group_norm,
    instance_norm,
    l2_normalize,
    layer_norm,
    local_response_normalization,
    lrn,
    moments,
    normalize_moments,
    rms_norm,
    rms_normalization,
    sufficient_statistics,
    weighted_moments,
    zero_fraction,
)


def test_normalization_brute():
    config.backend = "numpy"
    config.eager_mode = True

    t1 = Tensor(np.random.rand(2, 4, 8, 8).astype(np.float32), TensorConfig((2, 4, 8, 8), "float32", "cpu"))
    t1_1d = Tensor(np.random.rand(8).astype(np.float32), TensorConfig((8,), "float32", "cpu"))
    mean_t = Tensor(np.random.rand(8).astype(np.float32), TensorConfig((8,), "float32", "cpu"))
    var_t = Tensor(np.abs(np.random.rand(8).astype(np.float32)) + 0.1, TensorConfig((8,), "float32", "cpu"))
    scale = Tensor(np.ones((8,)).astype(np.float32), TensorConfig((8,), "float32", "cpu"))
    offset = Tensor(np.zeros((8,)).astype(np.float32), TensorConfig((8,), "float32", "cpu"))

    local_response_normalization(t1)
    batch_normalization(t1, mean_t, var_t, axis=-1)
    batch_normalization(t1, mean_t, var_t, axis=-1, config=BatchNormConfig(scale=scale, offset=offset))
    rms_normalization(t1, scale)
    batch_norm_with_global_normalization(t1, mean_t, var_t, offset, scale)
    batch_norm_with_global_normalization(t1, mean_t, var_t, offset, scale, config=BatchNormGlobalConfig(scale_after_normalization=False))
    lrn(t1)
    lrn(t1, LRNConfig())
    l2_normalize(t1, axis=-1)
    moments(t1, axes=-1)
    normalize_moments(t1, t1, t1, t1)
    sufficient_statistics(t1, axes=(-1,))
    weighted_moments(t1, axes=(-1,), frequency_weights=t1)
    zero_fraction(t1)

    layer_norm(t1, normalized_shape=(8,), scale=scale, offset=offset)
    layer_norm(t1, normalized_shape=(8,))

    group_norm(t1, num_groups=2, scale=Tensor(np.ones(4).astype(np.float32), TensorConfig((4,), "float32", "cpu")), offset=Tensor(np.zeros(4).astype(np.float32), TensorConfig((4,), "float32", "cpu")))
    group_norm(t1, num_groups=2)
    with pytest.raises(ValueError):
        group_norm(t1, num_groups=3)

    instance_norm(t1, scale=Tensor(np.ones(4).astype(np.float32), TensorConfig((4,), "float32", "cpu")), offset=Tensor(np.zeros(4).astype(np.float32), TensorConfig((4,), "float32", "cpu")))
    instance_norm(t1)

    batch_norm(t1, mean_t, var_t, axis=-1)
    rms_norm(t1, scale)
