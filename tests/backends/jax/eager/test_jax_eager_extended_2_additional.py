from ml_switcheroo_compiler.backends.jax.eager import _execute_adaptive_pool_mock


def test_adaptive_pool_mock_not_hasattr_shape():
    # Hit line 25: return operand
    class Obj:
        pass

    assert isinstance(_execute_adaptive_pool_mock(Obj(), 1), Obj)
