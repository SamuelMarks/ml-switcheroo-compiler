import pytest

from ml_switcheroo_compiler import random as rn
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.tracing.tracer import _tracer, ProxyTensor


def test_random_ops():
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()

        keys = rn.split(rn.PRNGKey(0))
        k1 = keys  # pass the whole split tensor for now to test ops

        # Test random ops
        rn.uniform(k1, (2,), minval=0.0, maxval=1.0)
        rn.normal(k1, (2,))
        rn.randint(k1, (2,), minval=0, maxval=10)
        rn.bernoulli(k1, 0.5, (2,))
        t_mock = rn.uniform(k1, (2,))
        rn.categorical(k1, t_mock)
        rn.permutation(k1, t_mock)
        rn.choice(k1, t_mock)
        rn.truncated_normal(k1, -1.0, 1.0, (2,))

        rn.fold_in(k1, 5)

        _tracer.stop_tracing()

    with ConfigContext(eager_mode=True):
        with pytest.raises(NotImplementedError):
            rn.state._emit_random_node("FakeOp", [], (), DType.Float32, {})


def test_stateless_beta():
    from ml_switcheroo_compiler.core.config import ConfigContext
    import numpy as np
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.random_stateless import stateless_beta
    from ml_switcheroo_compiler.tracing.tracer import _tracer

    seed = Tensor(np.array([0, 0]), TensorConfig((2,), DType.Int32, None))
    alpha = Tensor(np.array(1.0), TensorConfig((), DType.Float32, None))
    beta_param = Tensor(np.array(1.0), TensorConfig((), DType.Float32, None))

    with ConfigContext(eager_mode=True):
        res = stateless_beta((2, 2), seed, alpha, beta_param)
        assert res.shape == (2, 2)

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            p_seed = Tensor(
                ProxyTensor(id="mock_seed", shape=(2,), dtype=DType.Int32.value),
                TensorConfig((2,), DType.Int32, None),
            )
            p_alpha = Tensor(
                ProxyTensor(id="mock_a", shape=(), dtype=DType.Float32.value),
                TensorConfig((), DType.Float32, None),
            )
            p_beta = Tensor(
                ProxyTensor(id="mock_b", shape=(), dtype=DType.Float32.value),
                TensorConfig((), DType.Float32, None),
            )
            res = stateless_beta((2, 2), p_seed, p_alpha, p_beta)
            assert res is not None
        finally:
            _tracer.stop_tracing()


def test_stateless_shuffle():
    from ml_switcheroo_compiler.core.config import ConfigContext
    import numpy as np
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.random_stateless import stateless_shuffle
    from ml_switcheroo_compiler.tracing.tracer import _tracer

    seed = Tensor(np.array([0, 0]), TensorConfig((2,), DType.Int32, None))
    x = Tensor(np.array([1, 2, 3, 4]), TensorConfig((4,), DType.Int32, None))
    y = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Int32, None))

    with ConfigContext(eager_mode=True):
        res = stateless_shuffle(x, seed)
        assert res.shape == (4,)
        res2 = stateless_shuffle(y, seed, axis=1)
        assert res2.shape == (2, 2)

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            p_seed = Tensor(
                ProxyTensor(id="mock_seed", shape=(2,), dtype=DType.Int32.value),
                TensorConfig((2,), DType.Int32, None),
            )
            p_x = Tensor(
                ProxyTensor(id="mock_x", shape=(4,), dtype=DType.Int32.value),
                TensorConfig((4,), DType.Int32, None),
            )
            res = stateless_shuffle(p_x, p_seed)
            assert res is not None
        finally:
            _tracer.stop_tracing()
