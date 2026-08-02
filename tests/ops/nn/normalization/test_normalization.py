# ruff: noqa: E501

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.normalization.basic import GroupMean, GroupNorm, GroupVariance
from ml_switcheroo_compiler.ops.normalization.frontend import group_mean, group_norm, group_variance, spectral_normalization


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
        assert group_norm(t, 1) == "normed"
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
