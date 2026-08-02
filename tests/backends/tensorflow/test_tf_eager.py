def test_tf_eager_execute_op():
    import sys
    from unittest.mock import MagicMock, patch

    mock_tf = MagicMock()

    with patch.dict(sys.modules, {"tensorflow": mock_tf}):
        # We need to reset _OP_MAPPING inside module to None to test the lazy load
        import ml_switcheroo_compiler.backends.tensorflow.eager as tf_eager
        from ml_switcheroo_compiler.backends.tensorflow.eager import _get_op_mapping, execute_op

        tf_eager._OP_MAPPING = None

        # Test op_mapping population and lookup
        # Add is in the dict
        mock_tf.add.return_value = "tf_add_res"
        res = execute_op(None, "Add")
        assert res == "tf_add_res"

        # Now _OP_MAPPING is not None, test the early return in _get_op_mapping
        res2 = _get_op_mapping()
        assert res2 is not None

        # Test registered func intercept
        def dummy_func(module, *args, **kwargs):
            return "dummy_res"

        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=dummy_func):
            res3 = execute_op(None, "UnknownOp")
            assert res3 == "dummy_res"


def test_tf_eager_execute_op_fallback():
    from unittest.mock import patch

    import pytest

    from ml_switcheroo_compiler.backends.tensorflow.eager import execute_op
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
        with patch("ml_switcheroo_compiler.backends.tensorflow.eager._get_op_mapping", return_value={}):
            with pytest.raises(BackendNotSupportedError):
                execute_op(None, "OpThatDoesNotExist")
