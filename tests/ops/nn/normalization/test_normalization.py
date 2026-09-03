# ruff: noqa: E501

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.normalization.basic import GroupMean, GroupNorm, GroupVariance
from ml_switcheroo_compiler.ops.normalization.frontend import group_mean, group_variance, spectral_normalization
from ml_switcheroo_compiler.ops.normalization.frontend import group_norm as frontend_group_norm


def test_normalization_coverage():
    config.eager_mode = True
    t = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))

    assert GroupMean().infer_shape(t) == ()
    assert GroupVariance().infer_shape(t) == ()
    assert GroupNorm().infer_shape(t) == ()

    import sys

    norm_frontend = sys.modules["ml_switcheroo_compiler.ops.normalization.frontend"]

    class DummyOp:
        device = "cpu"

        def __call__(self, *args, **kwargs):
            return "normed"

    orig_get_op = norm_frontend.get_op
    norm_frontend.get_op = lambda x: DummyOp
    try:
        assert group_mean(t, 1) == "normed"
        assert group_variance(t, 1) == "normed"
        assert frontend_group_norm(t, 1) == "normed"
    finally:
        norm_frontend.get_op = orig_get_op

    with pytest.raises(Exception):
        spectral_normalization(t, t)


def test_tracing_normalization_ops():
    from ml_switcheroo_compiler.ops.nn.normalization import normalize_moments, sufficient_statistics, weighted_moments
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    config.eager_mode = False
    global_tracing_state.start_tracing("test_norm_trace")

    t = Tensor(np.ones((1, 2)), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    t.id = "tensor_n0"

    try:
        # Test normalize_moments tracing
        mean, variance = normalize_moments(t, t, t, t)
        assert isinstance(mean, Tensor)
        assert isinstance(variance, Tensor)
        assert mean.shape == (1, 2)
        assert variance.shape == (1, 2)

        # Test sufficient_statistics tracing
        counts, mean_ss, variance_ss, shift = sufficient_statistics(t, 1)
        assert isinstance(counts, Tensor)
        assert isinstance(mean_ss, Tensor)
        assert isinstance(variance_ss, Tensor)
        assert isinstance(shift, Tensor)

        # Test weighted_moments tracing
        w_mean, w_variance = weighted_moments(t, 1, t)
        assert isinstance(w_mean, Tensor)
        assert isinstance(w_variance, Tensor)
        assert w_mean.shape == (1, 2)
        assert w_variance.shape == (1, 2)
    finally:
        global_tracing_state.stop_tracing()
        config.eager_mode = True


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
