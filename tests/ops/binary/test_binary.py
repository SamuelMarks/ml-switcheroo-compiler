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


import numpy as np
import pytest

import ml_switcheroo_compiler.ops.binary.math as math_ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.binary import clip, rem


def test_binary_init_coverage():
    config.eager_mode = True
    t_a = Tensor(np.array(1.0), TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))
    t_b = Tensor(np.array(2.0), TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))

    assert divide_no_nan(t_a, t_b) is not None
    with pytest.raises(Exception):
        polar(t_a, t_b)
    with pytest.raises(Exception):
        view_as_complex(t_a)
    with pytest.raises(Exception):
        view_as_real(t_a)

    assert clip(t_a, 0.0, 1.0) is not None
    try:
        config.eager_mode = False
        with pytest.raises(Exception):
            clip(t_a, 0.0, 1.0)
    finally:
        config.eager_mode = True

    assert rem(t_a, t_b) is not None


def test_binary_math_coverage():
    class DummyShape:
        shape = (1, 2)

    op = math_ops.BinaryMathOp()
    assert op.infer_shape(DummyShape(), DummyShape()) is not None

    op_beta = math_ops.Betainc()
    assert op_beta.infer_shape(DummyShape(), DummyShape(), DummyShape()) is not None
    assert op_beta.infer_shape(DummyShape(), DummyShape()) is not None

    ops = [
        math_ops.DivideNoNan(),
        math_ops.MultiplyNoNan(),
        math_ops.SquaredDifference(),
        math_ops.Xdivy(),
        math_ops.Xlog1py(),
        math_ops.TruncateDiv(),
        math_ops.TruncateMod(),
        math_ops.ChebyshevPolynomialT(),
        math_ops.ChebyshevPolynomialU(),
        math_ops.ShiftedChebyshevPolynomialT(),
        math_ops.ShiftedChebyshevPolynomialU(),
        math_ops.ShiftedChebyshevPolynomialV(),
        math_ops.ShiftedChebyshevPolynomialW(),
        math_ops.HermitePolynomialH(),
        math_ops.HermitePolynomialHe(),
        math_ops.LaguerrePolynomialL(),
        math_ops.LegendrePolynomialP(),
        math_ops.IgammaGradA(),
        math_ops.RandomGammaGrad(),
        math_ops.SortKeyVal(),
        math_ops.Atan2(),
    ]
    for o in ops:
        assert o.op_name is not None

    poly_ops = [math_ops.Polyadd(), math_ops.Polysub(), math_ops.Polymul(), math_ops.Polydiv(), math_ops.Polyval(), math_ops.Poly(), math_ops.Polyder(), math_ops.Polyfit(), math_ops.Polyint(), math_ops.Roots()]
    for p in poly_ops:
        assert p.infer_shape(DummyShape()) is not None
        assert p.infer_shape() == ()

    c = math_ops.Clip()
    assert c.infer_shape(DummyShape()) == (1, 2)
    assert c.infer_shape() == ()
