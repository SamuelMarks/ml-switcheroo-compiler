import numpy as np
import pytest


def test_signal_missing():
    import ml_switcheroo_compiler.ops.signal as sig

    class FakeShape:
        shape = None

    with pytest.raises(ValueError):
        sig._validate_conv2d_args(FakeShape(), FakeShape())

    class FakeShape2:
        shape = (1, 1)

    with pytest.raises(ValueError):
        sig._validate_conv2d_args(FakeShape2(), FakeShape())

    with pytest.raises(ValueError):
        sig._validate_conv2d_args(FakeShape(), FakeShape2())


def test_signal_classes_infer_shape():
    import ml_switcheroo_compiler.ops.signal as sig

    class DummyData:
        shape = (1, 1)

    assert sig.Rfft().infer_shape(DummyData()) == (1, 1)
    assert sig.Fft2().infer_shape(DummyData()) == (1, 1)
    assert sig.Fftfreq().infer_shape(5) == (5,)
    assert sig.Fftfreq().infer_shape(DummyData()) == ()
    assert sig.Irfft().infer_shape(DummyData()) == (1, 1)
    assert sig.Ihfft().infer_shape(DummyData()) == (1, 1)


def test_signal_real_dsp_validation():
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.signal import fftn, ifft2, ifftn, irfft2, irfftn

    t_1d = Tensor(np.array([1.0, 2.0, 3.0, 4.0]), TensorConfig((4,), "float32", "cpu"))
    t_2d = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            assert isinstance(fftn(t_2d), Tensor)
            assert isinstance(ifftn(t_2d), Tensor)
            assert isinstance(irfftn(t_2d), Tensor)
            assert isinstance(ifft2(t_2d), Tensor)
            assert isinstance(irfft2(t_2d), Tensor)
