"""Tests for test_audio_ops."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.audio.ops import Dct, Frame, HammingWindow, HannWindow, Idct, InverseMdct, Istft, KaiserWindow, Mdct, MelFilterbank, MelSpectrogram, Mfcc, MfccsFromLogMelSpectrograms, OverlapAndAdd, Stft
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_audio_ops_transformations() -> None:
    """Test function test_audio_ops_transformations."""
    import unittest.mock as mock

    t_audio = Tensor(np.random.randn(1, 16000).astype("float32"), TensorConfig((1, 16000), "float32", "cpu"))
    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            stft = Stft()
            assert isinstance(stft.infer_shape(t_audio), tuple)
            istft = Istft()
            assert isinstance(istft.infer_shape(t_audio), tuple)
            mel = MelSpectrogram()
            assert isinstance(mel.infer_shape(t_audio), tuple)
            mfcc = Mfcc()
            assert isinstance(mfcc.infer_shape(t_audio), tuple)
            hann = HannWindow()
            assert isinstance(hann.infer_shape(128), tuple)


class MockArray:
    """Test class MockArray."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = as_callable(tuple)(shape)


class MockArg:
    """Test class MockArg."""

    def __init__(self, value: EagerValue) -> None:
        """Test function __init__.

        Args:
            value: Test parameter `value`.
        """
        self.value = value


def _test_audio_ops_infer_shape_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    op = Stft()
    assert as_callable(op.infer_shape)(as_callable(MockArray)((2, 100)), frame_step=10, frame_length=20) == (2, 9, 11)

    class NoShape:
        """Test class NoShape."""

        pass

    assert as_callable(op.infer_shape)(as_callable(NoShape)()) == ()
    assert op.infer_shape() == ()
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    op = typing.cast("Stft", MelSpectrogram())
    assert as_callable(op.infer_shape)(as_callable(MockArray)((2, 100)), frame_step=10, frame_length=20, num_mel_bins=8) == (2, 9, 8)
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) == ()
    assert op.infer_shape() == ()
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    op = typing.cast("Stft", Istft())
    assert as_callable(op.infer_shape)(as_callable(MockArray)((2, 9, 11)), frame_step=10, frame_length=20) == (2, 100)
    assert as_callable(op.infer_shape)(as_callable(NoShape)()) == ()
    assert op.infer_shape() == ()
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    assert as_callable(op.infer_shape)(as_callable(MockArray)((10,))) == (10,)
    op = typing.cast("Stft", MelFilterbank())
    assert op.infer_shape(num_mel_bins=12, fft_length=20) == (11, 12)
    assert op.infer_shape() == (129, 128)
    getattr(_state, "update")({"NoShape": NoShape, "op": op})
    _state.update(locals())


def _test_audio_ops_infer_shape_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    NoShape = _state["NoShape"]
    op = Mfcc()
    assert as_callable(op.infer_shape)(as_callable(MockArray)((2, 100)), frame_step=10, frame_length=20, num_mel_bins=16, num_mfccs=4) == (2, 9, 4)
    assert op.infer_shape(NoShape()) == ()
    assert op.infer_shape() == ()
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    op = typing.cast("Mfcc", MfccsFromLogMelSpectrograms())
    assert as_callable(op.infer_shape)(as_callable(MockArray)((2, 9, 16)), num_mfccs=4) == (2, 9, 4)
    assert op.infer_shape(NoShape()) == ()
    assert op.infer_shape() == ()
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    op1 = HannWindow()
    assert op1.infer_shape(window_length=5) == (5,)
    assert as_callable(op1.infer_shape)(as_callable(MockArg)(6), window_length=None) == (6,)
    assert as_callable(op1.infer_shape)(as_callable(MockArray)(()), window_length=256) == (256,)
    op2 = HammingWindow()
    assert op2.infer_shape(window_length=6) == (6,)
    assert as_callable(op2.infer_shape)(as_callable(MockArg)(7), window_length=None) == (7,)
    assert as_callable(op2.infer_shape)(as_callable(MockArray)(()), window_length=256) == (256,)
    op3 = KaiserWindow()
    assert op3.infer_shape(window_length=7) == (7,)
    assert as_callable(op3.infer_shape)(as_callable(MockArg)(8), window_length=None) == (8,)
    assert as_callable(op3.infer_shape)(as_callable(MockArray)(()), window_length=256) == (256,)
    getattr(_state, "update")({"op": op, "op1": op1, "op2": op2, "op3": op3})
    _state.update(locals())


def _test_audio_ops_infer_shape_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    op = Dct()
    assert as_callable(op.infer_shape)(as_callable(MockArray)((10, 20))) == (10, 20)
    op = typing.cast("Dct", Idct())
    assert as_callable(op.infer_shape)(as_callable(MockArray)((10, 20))) == (10, 20)
    op = typing.cast("Dct", Mdct())
    res = as_callable(op.infer_shape)(as_callable(MockArray)((2, 100)), frame_length=20, frame_step=10)
    assert as_callable(len)(res) == 3
    assert as_tensor(getattr(res, "__getitem__")(0)) == 2
    assert as_tensor(getattr(res, "__getitem__")(2)) == 10
    res_pad = as_callable(op.infer_shape)(as_callable(MockArray)((2, 100)), frame_length=20, frame_step=10, pad_end=True)
    assert as_callable(len)(res_pad) == 3
    res_pre = as_callable(op.infer_shape)(as_callable(MockArray)((2, 5, 20)), frame_length=20)
    assert res_pre == (2, 5, 10)
    res_short = as_callable(op.infer_shape)(as_callable(MockArray)((2, 10)), frame_length=20)
    assert res_short == (2, 5)
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    assert as_callable(op.infer_shape)() == ()
    op = typing.cast("Dct", InverseMdct())
    res = as_callable(op.infer_shape)(as_callable(MockArray)((2, 5, 10)), frame_length=20)
    assert as_callable(len)(res) == 3
    assert res == (2, 5, 20)
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    getattr(_state, "update")({"op": op, "res": res, "res_pad": res_pad, "res_pre": res_pre})
    getattr(_state, "update")({"res_short": res_short})
    _state.update(locals())


def _test_audio_ops_infer_shape_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    (NoShape, op) = (_state["NoShape"], _state["op"])
    assert op.infer_shape(NoShape()) == ()
    assert op.infer_shape() == ()
    op = as_tensor(Frame())
    assert as_callable(op.infer_shape)(as_callable(MockArray)((2, 100)), frame_length=20, frame_step=10) == (2, 9, 20)
    assert as_callable(op.infer_shape)(as_callable(MockArray)((2, 100)), frame_length=20, frame_step=10, pad_end=True) == (2, 10, 20)
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    assert op.infer_shape(NoShape()) == ()
    assert op.infer_shape() == ()
    op = as_tensor(OverlapAndAdd())
    assert as_callable(op.infer_shape)(as_callable(MockArray)((2, 9, 20)), frame_step=10) == (2, 100)
    assert as_callable(op.infer_shape)(as_callable(MockArray)(())) == ()
    with pytest.raises(AttributeError):
        op.infer_shape(NoShape())
    with pytest.raises(TypeError):
        op.infer_shape()
    getattr(_state, "update")({"op": op})


def test_audio_ops_infer_shape() -> None:
    """Test function test_audio_ops_infer_shape."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_audio_ops_infer_shape_part_1(_state)
    _test_audio_ops_infer_shape_part_2(_state)
    _test_audio_ops_infer_shape_part_3(_state)
    _test_audio_ops_infer_shape_part_4(_state)
    assert len(_state) > 0
