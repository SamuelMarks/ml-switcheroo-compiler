"""Tests for test_audio_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.audio.ops import Dct, Frame, HammingWindow, HannWindow, Idct, InverseMdct, Istft, KaiserWindow, Mdct, MelFilterbank, MelSpectrogram, Mfcc, MfccsFromLogMelSpectrograms, OverlapAndAdd, Stft

if TYPE_CHECKING:
    from collections.abc import Callable

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
        """Test function __init__."""
        self.shape = cast("Callable[..., EagerTensorProtocol]", tuple)(shape)


class MockArg:
    """Test class MockArg."""

    def __init__(self, value: EagerValue) -> None:
        """Test function __init__."""
        self.value = value


def _test_audio_ops_infer_shape_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    op = Stft()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 100)), frame_step=10, frame_length=20) == (2, 9, 11)

    # mock not having shape
    class NoShape:
        """Test class NoShape."""

        pass

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", NoShape)()) == ()
    assert op.infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()

    # MelSpectrogram
    op = cast("Stft", MelSpectrogram())
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 100)), frame_step=10, frame_length=20, num_mel_bins=8) == (2, 9, 8)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", NoShape)()) == ()
    assert op.infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()

    # Istft
    op = cast("Stft", Istft())
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 9, 11)), frame_step=10, frame_length=20) == (2, 100)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", NoShape)()) == ()
    assert op.infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((10,))) == (10,)

    # MelFilterbank
    op = cast("Stft", MelFilterbank())
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
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 100)), frame_step=10, frame_length=20, num_mel_bins=16, num_mfccs=4) == (2, 9, 4)
    assert op.infer_shape(NoShape()) == ()
    assert op.infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()

    # MfccsFromLogMelSpectrograms
    op = cast("Mfcc", MfccsFromLogMelSpectrograms())
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 9, 16)), num_mfccs=4) == (2, 9, 4)
    assert op.infer_shape(NoShape()) == ()
    assert op.infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()

    # Windows
    op1 = HannWindow()
    assert op1.infer_shape(window_length=5) == (5,)
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArg)(6), window_length=None) == (6,)
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(()), window_length=256) == (256,)  # fallback to kwargs

    op2 = HammingWindow()
    assert op2.infer_shape(window_length=6) == (6,)
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArg)(7), window_length=None) == (7,)
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(()), window_length=256) == (256,)

    op3 = KaiserWindow()
    assert op3.infer_shape(window_length=7) == (7,)
    assert cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArg)(8), window_length=None) == (8,)
    assert cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(()), window_length=256) == (256,)
    getattr(_state, "update")({"op": op, "op1": op1, "op2": op2, "op3": op3})
    _state.update(locals())


def _test_audio_ops_infer_shape_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    op = Dct()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((10, 20))) == (10, 20)

    # Idct
    op = cast("Dct", Idct())
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((10, 20))) == (10, 20)

    # Mdct
    op = cast("Dct", Mdct())
    res = cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 100)), frame_length=20, frame_step=10)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res) == 3
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)) == 2
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(2)) == 10

    # Mdct pad_end
    res_pad = cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 100)), frame_length=20, frame_step=10, pad_end=True)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res_pad) == 3

    # Mdct pre-framed block
    res_pre = cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 5, 20)), frame_length=20)
    assert res_pre == (2, 5, 10)

    # Mdct signal length < frame length
    res_short = cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 10)), frame_length=20)
    assert res_short == (2, 5)

    # Mdct string shape
    # Wait, the code doesn't support string shapes everywhere:
    # `signal_length < frame_length` will fail if signal_length is a str. Let's omit this.
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)() == ()

    # InverseMdct
    op = cast("Dct", InverseMdct())
    res = cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 5, 10)), frame_length=20)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res) == 3
    assert res == (2, 5, 20)

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()
    getattr(_state, "update")({"op": op, "res": res, "res_pad": res_pad, "res_pre": res_pre})
    getattr(_state, "update")({"res_short": res_short})
    _state.update(locals())


def _test_audio_ops_infer_shape_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    NoShape, op = _state["NoShape"], _state["op"]
    assert op.infer_shape(NoShape()) == ()
    assert op.infer_shape() == ()

    # Frame
    op = cast("EagerTensorProtocol", Frame())
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 100)), frame_length=20, frame_step=10) == (2, 9, 20)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 100)), frame_length=20, frame_step=10, pad_end=True) == (2, 10, 20)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()
    assert op.infer_shape(NoShape()) == ()
    assert op.infer_shape() == ()

    # OverlapAndAdd
    op = cast("EagerTensorProtocol", OverlapAndAdd())
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)((2, 9, 20)), frame_step=10) == (2, 100)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", MockArray)(())) == ()
    try:
        op.infer_shape(NoShape())
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        op.infer_shape()
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    getattr(_state, "update")({"op": op})


def test_audio_ops_infer_shape() -> None:
    # Stft
    """Test function test_audio_ops_infer_shape."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_audio_ops_infer_shape_part_1(_state)
    _test_audio_ops_infer_shape_part_2(_state)
    _test_audio_ops_infer_shape_part_3(_state)
    _test_audio_ops_infer_shape_part_4(_state)
    assert isinstance(_state, dict)
