"""Tests for test_main."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue

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
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_signal_classes_infer_shape() -> None:
    """Test function test_signal_classes_infer_shape."""
    t = MockTensor((2, 3))
    assert as_callable(as_callable(Convolve2d)().infer_shape)(t, t) == (2, 3)
    assert as_callable(as_callable(Fftconvolve)().infer_shape)(t, t) == (2, 3)
    assert as_callable(as_callable(Welch)().infer_shape)(t) == (2, 3)
    for Cls in [Fft, Ifft, Fftn, Ifftn, Rfftn, Irfftn, Ifft2, Rfft2, Irfft2, Fftnd, Ifftnd, Rfftnd, Irfftnd, Fftshift, Ifftshift, Hfft, Rfftfreq]:
        assert Cls().infer_shape(as_tensor(t)) == (2, 3)
    assert as_callable(Fftfreq)().infer_shape(5) == (5,)
    assert WindowHann().infer_shape(5) == (5,)
    assert WindowHamming().infer_shape(5) == (5,)
    assert as_callable(Stft)().infer_shape(as_callable(MockTensor)((10,)), 4) == (3, 2)
    assert as_callable(Istft)().infer_shape(as_callable(MockTensor)((3, 5)), 4) == (20,)


def test_validate_conv2d_args() -> None:
    """Test function test_validate_conv2d_args."""
    as_callable(_validate_conv2d_args)(as_callable(MockTensor)((2, 3)), as_callable(MockTensor)((2, 3)))
    assert callable(_validate_conv2d_args)


def test_calculate_padding() -> None:
    """Test function test_calculate_padding."""
    assert isinstance(_calculate_padding("full", "fill", 0.0), dict)


def _test_signal_funcs_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_signal_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    mocker = _state["mocker"]
    t = Tensor(MockTensor((10,)).data, TensorConfig((10,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.signal.conv_ops._emit_signal_node", return_value="node")
    mocker.patch("ml_switcheroo_compiler.ops.signal.fft_ops._emit_signal_node", return_value="node")
    mocker.patch("ml_switcheroo_compiler.ops.signal.spectral_ops._emit_linalg_node", return_value=("node", "node"))
    mocker.patch("ml_switcheroo_compiler.ops.signal.spectral_ops._emit_shape_node", return_value="node")
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
    getattr(_state, "update")({"WelchConfig": WelchConfig, "t": t})
    _state.update(locals())


def _test_signal_funcs_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_signal_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    mocker, t = _state["mocker"], _state["t"]
    assert as_callable(fftnd)(t) == "node"
    assert as_callable(ifftnd)(t) == "node"
    assert as_callable(rfftnd)(t) == "node"
    assert as_callable(irfftnd)(t) == "node"
    assert as_callable(fftshift)(t) == "node"
    assert as_callable(ifftshift)(t) == "node"
    assert as_callable(hfft)(t) == "node"
    assert as_callable(rfftfreq)(t) == "node"
    assert window_hann(5) == "node"
    assert window_hamming(5) == "node"
    assert as_callable(stft)(t, 4) == "node"
    assert as_callable(istft)(t, 4) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.signal.conv_ops.get_active_backend").return_value
    mocker.patch("ml_switcheroo_compiler.ops.signal.fft_ops.get_active_backend", return_value=mock_backend)
    mocker.patch("ml_switcheroo_compiler.ops.signal.spectral_ops.get_active_backend", return_value=mock_backend)
    setattr(mock_backend.execute_op, "return_value", MockTensor((10,)))
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((10,)))
    assert as_tensor(getattr(as_callable(fft)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(ifft)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(fftn)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(ifftn)(t), "config")).shape == (10,)
    getattr(_state, "update")({"mock_backend": mock_backend})
    _state.update(locals())


def _test_signal_funcs_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_signal_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]
    assert as_tensor(getattr(as_callable(rfftn)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(irfftn)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(ifft2)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(rfft2)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(irfft2)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(fftnd)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(ifftnd)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(rfftnd)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(irfftnd)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(fftshift)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(ifftshift)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(hfft)(t), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(rfftfreq)(t), "config")).shape == (10,)
    assert as_tensor(getattr(window_hann(5), "config")).shape == (10,)
    assert as_tensor(getattr(window_hamming(5), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(stft)(t, 4), "config")).shape == (10,)
    assert as_tensor(getattr(as_callable(istft)(t, 4), "config")).shape == (10,)


def test_signal_funcs(mocker: EagerTensorProtocol) -> None:
    """Test function test_signal_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
    _state: dict[str, EagerTensorProtocol] = {"mocker": mocker}
    _test_signal_funcs_part_1(_state)
    _test_signal_funcs_part_2(_state)
    _test_signal_funcs_part_3(_state)
    assert len(_state) > 0


def test_signal_funcs_eager_extra(mocker: EagerTensorProtocol) -> None:
    """Test function test_signal_funcs_eager_extra.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((10,)).data, TensorConfig((10,), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.signal.conv_ops.get_active_backend").return_value
    mocker.patch("ml_switcheroo_compiler.ops.signal.fft_ops.get_active_backend", return_value=mock_backend)
    mocker.patch("ml_switcheroo_compiler.ops.signal.spectral_ops.get_active_backend", return_value=mock_backend)
    setattr(mock_backend.execute_op, "return_value", MockTensor((10,)))
    res1 = convolve2d(t, t)
    assert as_tensor(getattr(res1, "config")).shape == (10,)
    setattr(mock_backend.execute_op, "return_value", MockTensor((10,)))
    res2 = fftconvolve(t, t)
    assert as_tensor(getattr(res2, "config")).shape == (10,)
    setattr(mock_backend.execute_op, "return_value", (MockTensor((10,)), MockTensor((10,))))
    res3 = welch(t)
    assert as_tensor(getattr(as_tensor(getattr(res3, "__getitem__")(0)), "config")).shape == (10,)
    assert as_tensor(getattr(as_tensor(getattr(res3, "__getitem__")(1)), "config")).shape == (10,)
