# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.signal import (
    Convolve2d,
    Fft,
    Fftconvolve,
    Fftfreq,
    Fftn,
    Fftnd,
    Fftshift,
    Hfft,
    Ifft,
    Ifft2,
    Ifftn,
    Ifftnd,
    Ifftshift,
    Irfft2,
    Irfftn,
    Irfftnd,
    Istft,
    Rfft2,
    Rfftfreq,
    Rfftn,
    Rfftnd,
    Stft,
    Welch,
    WindowHamming,
    WindowHann,
    _calculate_padding,
    _validate_conv2d_args,
    convolve2d,
    fft,
    fftconvolve,
    fftn,
    fftnd,
    fftshift,
    hfft,
    ifft,
    ifft2,
    ifftn,
    ifftnd,
    ifftshift,
    irfft2,
    irfftn,
    irfftnd,
    istft,
    rfft2,
    rfftfreq,
    rfftn,
    rfftnd,
    stft,
    welch,
    window_hamming,
    window_hann,
)


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_signal_classes_infer_shape():
    t = MockTensor((2, 3))
    assert Convolve2d().infer_shape(t, t) == (2, 3)
    assert Fftconvolve().infer_shape(t, t) == (2, 3)
    assert Welch().infer_shape(t) == (2, 3)
    for Cls in [Fft, Ifft, Fftn, Ifftn, Rfftn, Irfftn, Ifft2, Rfft2, Irfft2, Fftnd, Ifftnd, Rfftnd, Irfftnd, Fftshift, Ifftshift, Hfft, Rfftfreq]:
        assert Cls().infer_shape(t) == (2, 3)
    assert Fftfreq().infer_shape(5) == (5,)
    assert WindowHann().infer_shape(5) == (5,)
    assert WindowHamming().infer_shape(5) == (5,)
    assert Stft().infer_shape(MockTensor((10,)), 4) == (3, 2)
    assert Istft().infer_shape(MockTensor((3, 5)), 4) == (20,)


def test_validate_conv2d_args():
    _validate_conv2d_args(MockTensor((2, 3)), MockTensor((2, 3)))


def test_calculate_padding():
    assert isinstance(_calculate_padding("full", "fill", 0.0), dict)


def test_signal_funcs(mocker):
    t = Tensor(MockTensor((10,)).data, TensorConfig((10,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.signal._emit_signal_node", return_value="node")
    mocker.patch("ml_switcheroo_compiler.ops.signal._emit_linalg_node", return_value=("node", "node"))
    mocker.patch("ml_switcheroo_compiler.ops.signal._emit_shape_node", return_value="node")
    assert convolve2d(t, t) == "node"
    assert fftconvolve(t, t) == "node"
    assert welch(t) == ("node", "node")
    from ml_switcheroo_compiler.ops.signal import WelchConfig

    assert welch(t, WelchConfig()) == ("node", "node")
    assert fft(t) == "node"
    assert ifft(t) == "node"
    assert fftn(t) == "node"
    assert ifftn(t) == "node"
    assert rfftn(t) == "node"
    assert irfftn(t) == "node"
    assert ifft2(t) == "node"
    assert rfft2(t) == "node"
    assert irfft2(t) == "node"
    assert fftnd(t) == "node"
    assert ifftnd(t) == "node"
    assert rfftnd(t) == "node"
    assert irfftnd(t) == "node"
    assert fftshift(t) == "node"
    assert ifftshift(t) == "node"
    assert hfft(t) == "node"
    assert rfftfreq(t) == "node"
    assert window_hann(5) == "node"
    assert window_hamming(5) == "node"
    assert stft(t, 4) == "node"
    assert istft(t, 4) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.signal.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((10,))
    mock_backend.array.side_effect = lambda x: MockTensor((10,))
    assert fft(t).config.shape == (10,)
    assert ifft(t).config.shape == (10,)
    assert fftn(t).config.shape == (10,)
    assert ifftn(t).config.shape == (10,)
    assert rfftn(t).config.shape == (10,)
    assert irfftn(t).config.shape == (10,)
    assert ifft2(t).config.shape == (10,)
    assert rfft2(t).config.shape == (10,)
    assert irfft2(t).config.shape == (10,)
    assert fftnd(t).config.shape == (10,)
    assert ifftnd(t).config.shape == (10,)
    assert rfftnd(t).config.shape == (10,)
    assert irfftnd(t).config.shape == (10,)
    assert fftshift(t).config.shape == (10,)
    assert ifftshift(t).config.shape == (10,)
    assert hfft(t).config.shape == (10,)
    assert rfftfreq(t).config.shape == (10,)
    assert window_hann(5).config.shape == (10,)
    assert window_hamming(5).config.shape == (10,)
    assert stft(t, 4).config.shape == (10,)
    assert istft(t, 4).config.shape == (10,)


def test_signal_funcs_eager_extra(mocker):
    t = Tensor(MockTensor((10,)).data, TensorConfig((10,), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.signal.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((10,))
    res1 = convolve2d(t, t)
    assert res1.config.shape == (10,)
    mock_backend.execute_op.return_value = MockTensor((10,))
    res2 = fftconvolve(t, t)
    assert res2.config.shape == (10,)
    mock_backend.execute_op.return_value = (MockTensor((10,)), MockTensor((10,)))
    res3 = welch(t)
    assert res3[0].config.shape == (10,)
    assert res3[1].config.shape == (10,)
