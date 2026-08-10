import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.control_flow import cond, while_loop


def test_cond_grad():
    def f(x):
        def true_fn(x):
            return x * x * x

        def false_fn(x):
            return x * x

        c = x > 0.0
        return cond(c, lambda: true_fn(x), lambda: false_fn(x))

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    out, vjp_fn = vjp(f, x)
    assert out is not None
    # Test backward pass graph construction
    import pytest

    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError, match="Operation If is not implemented in interpreter."):
        gx = vjp_fn(Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu"))))


def test_while_loop_grad():
    def f(x):
        def cond_fn(val):
            return val[0] < 10.0

        def body_fn(val):
            return (val[0] + x,)

        res = while_loop(cond_fn, body_fn, (x,))
        return res[0]

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    out, vjp_fn = vjp(f, x)
    assert out is not None
    import pytest

    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    with pytest.raises(UnimplementedMathError, match="Operation Loop is not implemented in interpreter."):
        gx = vjp_fn(Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu"))))
