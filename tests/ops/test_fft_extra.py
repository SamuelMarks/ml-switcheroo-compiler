import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.tracing.tracer import _tracer
from ml_switcheroo_compiler.ops.linalg.fft import fftfreq, hfft, ihfft, rfftfreq


def _test_op(func, *args, **kwargs):
    with ConfigContext(eager_mode=True):
        out_eager = func(*args, **kwargs)
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            out_traced = func(*args, **kwargs)
        finally:
            _tracer.stop_tracing()
    return out_eager, out_traced


def test_fft_ops():
    a = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))

    e, t = _test_op(fftfreq, 2)
    assert e.shape == (2,)

    e, t = _test_op(hfft, a)
    assert e.shape == (2,)

    e, t = _test_op(ihfft, a)
    assert e.shape == (2,)

    e, t = _test_op(rfftfreq, 2)
    assert e.shape == (2,)


def test_opdefs_infer_shapes():
    from ml_switcheroo_compiler.ops.linalg.fft import Fftfreq, Hfft, Ihfft, Rfftfreq

    assert Fftfreq().infer_shape(2) == (2,)
    assert Hfft().infer_shape() == ()
    assert Ihfft().infer_shape() == ()
    assert Rfftfreq().infer_shape(2) == (2,)
