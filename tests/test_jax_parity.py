"""Tests for JAX parity."""

import numpy as np

import ml_switcheroo_compiler.grad as grad_module
from ml_switcheroo_compiler import nn, ops, random
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor


def test_grad_coverage() -> None:
    """Tests the gradient API."""

    def f(x: object) -> object:
        """Docstring."""
        return x

    g1 = grad_module.ir_grad(f)
    g2 = grad_module.grad(f)
    g3 = grad_module.value_and_grad(f)

    assert g1(1) == 1
    assert g2(1) == 1
    assert g3(1) == (1, 1)

    assert grad_module.jit(f)(1) == 1
    with grad_module.disable_jit():
        pass
    assert grad_module.eval_shape(f, 1) == 1


def test_nn_coverage() -> None:
    """Tests the NN API."""
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType

    config.eager_mode = True
    t = Tensor(np.array(1.0, dtype=np.float32), (), DType.Float32, Device("cpu"))
    assert nn.gelu(t) is not None
    assert nn.logsumexp(t) is not None
    assert nn.one_hot(t, 2) is not None
    assert nn.softmax(t) is not None
    assert nn.sigmoid(t) is not None
    assert nn.log_sigmoid(t) is not None
    assert nn.relu(t) is not None
    assert nn.relu6(t) is not None
    assert nn.hard_sigmoid(t) is not None
    assert nn.hard_tanh(t) is not None
    assert nn.swish(t) is not None
    assert nn.silu(t) is not None
    assert nn.elu(t) is not None
    assert nn.celu(t) is not None
    assert nn.selu(t) is not None
    assert nn.log_softmax(t) is not None

    key = random.PRNGKey(0)
    z = nn.zeros(key, (2,))
    assert z.shape == (2,)
    o = nn.ones(key, (2,))
    assert o.shape == (2,)

    c = nn.constant(1.0)
    assert c(key, (2,)).shape == (2,)

    for init_fn in [nn.uniform, nn.normal, nn.truncated_normal]:
        assert init_fn()(key, (2,)).shape == (2,)

    for init_fn in [
        nn.glorot_uniform,
        nn.glorot_normal,
        nn.lecun_uniform,
        nn.lecun_normal,
        nn.he_uniform,
        nn.he_normal,
    ]:
        assert init_fn()(key, (2,)).shape == (2,)

    from ml_switcheroo_compiler.ops.configs import InitializerConfig

    for init_fn in [nn.orthogonal, nn.delta_orthogonal, nn.variance_scaling]:
        if init_fn == nn.variance_scaling:
            assert init_fn(InitializerConfig(scale=1.0, mode="fan_in", distribution="uniform"))(
                key, (2,)
            ).shape == (2,)
        else:
            assert init_fn()(key, (2,)).shape == (2,)


def test_random_coverage() -> None:
    """Tests the random API."""
    key = random.PRNGKey(123)
    assert key.shape == (2,)
    assert random.split(key, 2).shape == (2, 2)
    assert random.fold_in(key, 1).shape == (2,)

    assert random.uniform(key, (2,)).shape == (2,)
    assert random.normal(key, (2,)).shape == (2,)
    assert random.randint(key, (2,), 0, 10).shape == (2,)
    assert random.bernoulli(key, shape=(2,)).shape == (2,)
    assert random.bernoulli(key).shape == ()
    assert random.categorical(key, None, shape=(2,)).shape == (2,)
    assert random.categorical(key, None).shape == ()
    assert random.permutation(key, key).shape == (2,)
    assert random.choice(key, key).shape == ()
    assert random.truncated_normal(key, -2.0, 2.0, (2,)).shape == (2,)


def test_ops_composite_coverage() -> None:
    """Tests composite ops."""
    t1 = Tensor(np.array([1, 2]), (2,), "int32", "cpu")
    t2 = Tensor(np.array([1, 2]), (2,), "int32", "cpu")

    config.eager_mode = True
    assert ops.clamp(1, t1, 2) is not None
    assert ops.clip(t1, 1, 2) is not None
    assert ops.broadcast_shapes((1,), (2,)) == (2,)

    from ml_switcheroo_compiler.ops.configs import SpaceConfig

    ls = ops.logspace(1, 2)
    assert ls is not None
    ls2 = ops.logspace(1, 2, SpaceConfig(base=2.0))
    assert ls2 is not None

    assert ops.rint(t1) is not None
    assert ops.broadcast(t1, (2, 2)) is not None

    config.eager_mode = False
    try:
        t1.data = type("obj", (), {"id": "1"})()
        t2.data = type("obj", (), {"id": "2"})()
        ops.dynamic_update_slice(t1, t2, (0,))
    except Exception:
        pass
    config.eager_mode = True

    assert ops.select(Tensor(np.array(True), (), "bool", "cpu"), t1, t2) is not None

    # array_equal eager
    config.eager_mode = True
