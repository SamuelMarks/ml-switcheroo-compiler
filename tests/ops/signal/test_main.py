"""Tests for test_main."""

from __future__ import annotations

import typing

import pytest
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.backends.eager_registry import BackendArg

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
from tests.testing_utils import MockTensorStub, coerce_operand_list, ensure_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor(MockTensorStub):
    """Test class MockTensor."""

    def __init__(self, shape: BackendArg = ()) -> None:
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
    assert Convolve2d().infer_shape(t, t) == (2, 3)
    assert Fftconvolve().infer_shape(t, t) == (2, 3)
    assert Welch().infer_shape(t) == (2, 3)
    for Cls in [
        Fft,
        Ifft,
        Fftn,
        Ifftn,
        Rfftn,
        Irfftn,
        Ifft2,
        Rfft2,
        Irfft2,
        Fftnd,
        Ifftnd,
        Rfftnd,
        Irfftnd,
        Fftshift,
        Ifftshift,
        Hfft,
        Rfftfreq,
    ]:
        assert Cls().dispatch_infer_shape(t) == (2, 3)
    assert Fftfreq().infer_shape(5) == (5,)
    assert WindowHann().infer_shape(5) == (5,)
    assert WindowHamming().infer_shape(5) == (5,)
    assert Stft().infer_shape(MockTensor((10,)), 4) == (3, 2)
    assert Istft().infer_shape(MockTensor((3, 5)), 4) == (20,)


def test_validate_conv2d_args() -> None:
    """Test function test_validate_conv2d_args."""
    _validate_conv2d_args(MockTensor((2, 3)), MockTensor((2, 3)))
    assert callable(_validate_conv2d_args)


def test_calculate_padding() -> None:
    """Test function test_calculate_padding."""
    assert isinstance(_calculate_padding("full", "fill", 0.0), dict)


class _TestSignalFuncsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_signal_funcs` split helpers."""

    mocker: MockerFixture
    t: Tensor


def _test_signal_funcs_part_1(_state: _TestSignalFuncsState) -> None:
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
    _state.update({"t": t})
    _state.update({"mocker": mocker})


def _test_signal_funcs_part_2(_state: _TestSignalFuncsState) -> None:
    """Execute part 2 of test_signal_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    mocker, t = _state["mocker"], _state["t"]
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
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.signal.conv_ops.get_active_backend").return_value
    mocker.patch("ml_switcheroo_compiler.ops.signal.fft_ops.get_active_backend", return_value=mock_backend)
    mocker.patch("ml_switcheroo_compiler.ops.signal.spectral_ops.get_active_backend", return_value=mock_backend)
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((10,)), raising=False)
    pytest.MonkeyPatch().setattr(mock_backend.array, "side_effect", lambda x: MockTensor((10,)), raising=False)
    assert ensure_tensor(fft(t)).config.shape == (10,)
    assert ensure_tensor(ifft(t)).config.shape == (10,)
    assert ensure_tensor(fftn(t)).config.shape == (10,)
    assert ensure_tensor(ifftn(t)).config.shape == (10,)
    _state.update({"mocker": mocker, "t": t})


def _test_signal_funcs_part_3(_state: _TestSignalFuncsState) -> None:
    """Execute part 3 of test_signal_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]
    assert ensure_tensor(rfftn(t)).config.shape == (10,)
    assert ensure_tensor(irfftn(t)).config.shape == (10,)
    assert ensure_tensor(ifft2(t)).config.shape == (10,)
    assert ensure_tensor(rfft2(t)).config.shape == (10,)
    assert ensure_tensor(irfft2(t)).config.shape == (10,)
    assert ensure_tensor(fftnd(t)).config.shape == (10,)
    assert ensure_tensor(ifftnd(t)).config.shape == (10,)
    assert ensure_tensor(rfftnd(t)).config.shape == (10,)
    assert ensure_tensor(irfftnd(t)).config.shape == (10,)
    assert ensure_tensor(fftshift(t)).config.shape == (10,)
    assert ensure_tensor(ifftshift(t)).config.shape == (10,)
    assert ensure_tensor(hfft(t)).config.shape == (10,)
    assert ensure_tensor(rfftfreq(t)).config.shape == (10,)
    assert ensure_tensor(window_hann(5)).config.shape == (10,)
    assert ensure_tensor(window_hamming(5)).config.shape == (10,)
    assert ensure_tensor(stft(t, 4)).config.shape == (10,)
    assert ensure_tensor(istft(t, 4)).config.shape == (10,)


def test_signal_funcs(mocker: MockerFixture) -> None:
    """Test function test_signal_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
    _state: _TestSignalFuncsState = {"mocker": mocker}
    _test_signal_funcs_part_1(_state)
    _test_signal_funcs_part_2(_state)
    _test_signal_funcs_part_3(_state)
    assert len(_state) > 0


def test_signal_funcs_eager_extra(mocker: MockerFixture) -> None:
    """Test function test_signal_funcs_eager_extra.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((10,)).data, TensorConfig((10,), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.signal.conv_ops.get_active_backend").return_value
    mocker.patch("ml_switcheroo_compiler.ops.signal.fft_ops.get_active_backend", return_value=mock_backend)
    mocker.patch("ml_switcheroo_compiler.ops.signal.spectral_ops.get_active_backend", return_value=mock_backend)
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((10,)), raising=False)
    res1 = convolve2d(t, t)
    assert ensure_tensor(res1).config.shape == (10,)
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((10,)), raising=False)
    res2 = fftconvolve(t, t)
    assert ensure_tensor(res2).config.shape == (10,)
    pytest.MonkeyPatch().setattr(
        mock_backend.execute_op, "return_value", (MockTensor((10,)), MockTensor((10,))), raising=False
    )
    res3 = welch(t)
    assert ensure_tensor(coerce_operand_list(res3)[0]).config.shape == (10,)
    assert ensure_tensor(coerce_operand_list(res3)[1]).config.shape == (10,)
