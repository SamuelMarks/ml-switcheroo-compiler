import numpy as np
from ml_switcheroo_compiler.backends.numpy.eager.nn import (
    _gelu,
    _np_relu,
    _np_alpha_dropout,
    _np_activity_regularization,
    _np_dropout,
    _np_time_distributed,
)
import ml_switcheroo_compiler.backends.eager as eager_mod


def test_numpy_nn_eager_extra():
    _gelu(np.array([-1.0, 2.0, 5.0]))

    res = _np_relu(np, np.array([-1.0, 2.0, 5.0]))
    assert np.allclose(res, [0.0, 2.0, 5.0])

    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert res.shape == (2, 2)

    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.0)
    assert np.allclose(res, 1.0)

    res = _np_alpha_dropout(
        np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True
    )
    assert res.shape == (2, 2)

    res = _np_activity_regularization(np, np.ones((2, 2)), l1=0.1, l2=0.2)
    assert res.shape == (2, 2)

    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert res.shape == (2, 2)

    res = _np_dropout(np, np.ones((2, 2)), rate=0.0)
    assert np.allclose(res, 1.0)

    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert res.shape == (2, 2)


def test_np_time_distributed(monkeypatch):
    def mock_exec(backend, op_name, x, **kwargs):
        return x

    monkeypatch.setattr(eager_mod, "execute_generic_op", mock_exec)

    res = _np_time_distributed(np, np.ones((2, 2)), wrapped_op_name="Dummy")
    assert res.shape == (2, 2)

    res = _np_time_distributed(np, np.ones((2, 3, 4)), wrapped_op_name="Dummy")
    assert res.shape == (2, 3, 4)
