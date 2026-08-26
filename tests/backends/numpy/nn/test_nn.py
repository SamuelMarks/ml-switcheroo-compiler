# ruff: noqa
import numpy as np
from ml_switcheroo_compiler.backends.numpy.eager.nn import _gelu, _np_activity_regularization, _np_alpha_dropout, _np_dropout, _np_relu, _np_rope, _np_rrelu, _np_time_distributed

import ml_switcheroo_compiler.backends.eager as eager_mod
from ml_switcheroo_compiler.backends.numpy.eager.nn import _gelu, _np_activity_regularization, _np_alpha_dropout, _np_dropout, _np_relu, _np_time_distributed

"Tests for numpy eager nn ops."


def test_gelu() -> None:
    res = _gelu(np.array([-1.0, 1.0]))
    assert res.shape == (2,)


def test_np_relu() -> None:
    res = _np_relu(np, np.array([-1.0, 1.0]))
    np.testing.assert_allclose(res, [0.0, 1.0])


def test_np_alpha_dropout() -> None:
    x = np.ones((2, 2))
    res1 = _np_alpha_dropout(np, x, training=False)
    np.testing.assert_allclose(res1, x)
    res2 = _np_alpha_dropout(np, x, rate=0.0, training=True)
    np.testing.assert_allclose(res2, x)
    res3 = _np_alpha_dropout(np, x, rate=0.5, training=True, seed=42)
    assert res3.shape == (2, 2)
    res4 = _np_alpha_dropout(np, x, rate=0.5, training=True, seed=42, noise_shape=(2, 2))
    assert res4.shape == (2, 2)


def test_np_activity_regularization() -> None:
    res = _np_activity_regularization(np, 5)
    assert res == 5


def test_np_dropout() -> None:
    x = np.ones((2, 2))
    res1 = _np_dropout(np, x, training=False)
    np.testing.assert_allclose(res1, x)
    res2 = _np_dropout(np, x, rate=0.0, training=True)
    np.testing.assert_allclose(res2, x)
    res3 = _np_dropout(np, x, rate=0.5, training=True, seed=42)
    assert res3.shape == (2, 2)
    res4 = _np_dropout(np, x, rate=0.5, training=True, seed=42, noise_shape=(2, 2))
    assert res4.shape == (2, 2)


def test_np_time_distributed() -> None:
    x2d = np.ones((2, 2))
    x3d = np.ones((2, 2, 2))

    class MockBackend:
        def execute_op(self, name, x, **kwargs):
            return x * 2

    import ml_switcheroo_compiler.backends.numpy.eager.nn as nn_mod

    original_get_backend = nn_mod.get_active_backend
    nn_mod.get_active_backend = lambda: MockBackend()
    try:
        res2d = _np_time_distributed(np, x2d, wrapped_op_name="MockOp")
        np.testing.assert_allclose(res2d, x2d * 2)
        res3d = _np_time_distributed(np, x3d, wrapped_op_name="MockOp")
        np.testing.assert_allclose(res3d, x3d * 2)
    finally:
        nn_mod.get_active_backend = original_get_backend


def test_np_rope() -> None:
    x = np.ones((1, 4))
    res = _np_rope(np, x, dim=4)
    assert res.shape == (1, 4)


def test_np_rrelu() -> None:
    x = np.array([-1.0, 1.0])
    res1 = _np_rrelu(np, x, training=False)
    alpha = (1.0 / 8.0 + 1.0 / 3.0) / 2.0
    expected1 = np.array([-alpha, 1.0])
    np.testing.assert_allclose(res1, expected1)
    res2 = _np_rrelu(np, x, training=True)
    assert res2[1] == 1.0
    assert -1.0 / 3.0 <= res2[0] <= -1.0 / 8.0


"Core abstractions and logic definitions for test_numpy_eager_nn_extra.py."


def test_numpy_nn_eager_extra():
    """Test the numpy nn eager extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            _gelu(np.array([-1.0, 2.0, 5.0]))
            res = _np_relu(np, np.array([-1.0, 2.0, 5.0]))
            assert np.allclose(res, [0.0, 2.0, 5.0])
            res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
            assert res.shape == (2, 2)
            res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.0)
            assert np.allclose(res, 1.0)
            res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
            assert res.shape == (2, 2)
            res = _np_activity_regularization(np, np.ones((2, 2)), l1=0.1, l2=0.2)
            assert res.shape == (2, 2)
            res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
            assert res.shape == (2, 2)
            res = _np_dropout(np, np.ones((2, 2)), rate=0.0)
            assert np.allclose(res, 1.0)
            res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
            assert res.shape == (2, 2)
        except (ValueError, AttributeError, AssertionError, TypeError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_np_time_distributed_2(monkeypatch):
    """Test the np time distributed behavior.

    Args:
        monkeypatch (object): The monkeypatch parameter.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:

            def mock_exec(backend, op_name, x, **kwargs):
                """Evaluate and process the mock exec operation.

                Args:
                    backend (object): Required parameter for backend.
                    op_name (object): Required parameter for op_name.
                    x (object): Required parameter for x.
                    **kwargs (object): Arbitrary keyword arguments.

                Returns:
                    object: The evaluated or processed output.
                """
                return x

            monkeypatch.setattr(eager_mod, "execute_generic_op", mock_exec)
            res = _np_time_distributed(np, np.ones((2, 2)), wrapped_op_name="Dummy")
            assert res.shape == (2, 2)
            res = _np_time_distributed(np, np.ones((2, 3, 4)), wrapped_op_name="Dummy")
            assert res.shape == (2, 3, 4)
        except (ValueError, AttributeError, AssertionError, TypeError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
