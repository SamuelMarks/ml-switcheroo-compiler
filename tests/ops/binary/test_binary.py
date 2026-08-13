from ml_switcheroo_compiler.ops.binary import divide_no_nan, polar, view_as_complex, view_as_real


def test_divide_no_nan():
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    config.eager_mode = True
    try:
        t1 = Tensor(data=np.array([1.0, 2.0]), config=TensorConfig((2,), "float32", None))
        t2 = Tensor(data=np.array([0.0, 2.0]), config=TensorConfig((2,), "float32", None))

        # Needs to evaluate
        res = divide_no_nan(t1, t2)
        assert res is not None
    finally:
        config.eager_mode = False


def test_polar_view():
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend_getter:
        mock_backend = mock_backend_getter.return_value

        mock_backend.execute_op.return_value = "polar_res"
        res = polar(1, 2)
        assert res == "polar_res"

        mock_backend.execute_op.return_value = "complex_res"
        res = view_as_complex(1)
        assert res == "complex_res"

        mock_backend.execute_op.return_value = "real_res"
        res = view_as_real(1)
        assert res == "real_res"
