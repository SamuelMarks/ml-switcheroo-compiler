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


def test_binary_imports_exceptions():
    import importlib
    import sys
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.binary

    ops_base_mod = sys.modules["ml_switcheroo_compiler.ops.base"]

    # We patch ml_switcheroo_compiler.ops.base.get_op safely
    with patch.object(ops_base_mod, "get_op", side_effect=KeyError("Boom")):
        # We need to forcefully reload ml_switcheroo_compiler.ops.binary to hit KeyErrors
        importlib.reload(ml_switcheroo_compiler.ops.binary)

        assert ml_switcheroo_compiler.ops.binary.add is None
        assert ml_switcheroo_compiler.ops.binary.divide is None
        assert ml_switcheroo_compiler.ops.binary.multiply is None
        assert ml_switcheroo_compiler.ops.binary.legendre_polynomial_p is None

    # After test, we should reload cleanly
    importlib.reload(ml_switcheroo_compiler.ops.binary)
