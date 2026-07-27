from unittest.mock import patch

from ml_switcheroo_compiler.backends.jax.eager import _execute_adaptive_pool_mock, execute_op


def test_adaptive_pool_mock_not_hasattr_shape():
    class Obj:
        pass

    assert isinstance(_execute_adaptive_pool_mock(Obj(), 1), Obj)


def test_execute_op_exception():
    with patch("numpy.zeros", side_effect=Exception("mock")):
        res = execute_op(None, "UnknownFakeOpThatWillCrash", [1.0])
        assert res is None
