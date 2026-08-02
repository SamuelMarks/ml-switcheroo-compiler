def test_frontend_utils_frompyfunc():
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.creation.frontend_utils import frompyfunc

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "dummy_frompyfunc"
        assert frompyfunc(lambda x: x, 1, 1) == "dummy_frompyfunc"
