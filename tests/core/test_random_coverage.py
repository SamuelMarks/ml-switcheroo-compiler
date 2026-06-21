import pytest

from ml_switcheroo_compiler import random as rn
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.tracing.tracer import _tracer


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
