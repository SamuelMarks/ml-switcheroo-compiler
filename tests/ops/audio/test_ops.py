"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.audio.ops import Dct, Frame, HammingWindow, HannWindow, Idct, InverseMdct, Istft, KaiserWindow, Mdct, MelFilterbank, MelSpectrogram, Mfcc, MfccsFromLogMelSpectrograms, OverlapAndAdd, Stft

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape


def test_audio_ops() -> None:
    """Test function test_audio_ops."""
    assert Stft().infer_shape() == ()
    assert MelSpectrogram().infer_shape() == ()
    assert Istft().infer_shape() == ()
    assert MelFilterbank().infer_shape() == (129, 128)
    assert Mfcc().infer_shape() == ()
    assert MfccsFromLogMelSpectrograms().infer_shape() == ()
    assert HannWindow().infer_shape() == (256,)
    assert HammingWindow().infer_shape() == (256,)
    assert KaiserWindow().infer_shape() == (256,)

    t = DummyNode((10,))
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", Dct)().infer_shape)(t) == (10,)
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", Idct)().infer_shape)(t) == (10,)

    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", Mdct)().infer_shape)(t) == (5,)
    assert typing.cast("EagerCallableProtocol", Mdct)().infer_shape(typing.cast("EagerCallableProtocol", DummyNode)(())) == ()

    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", InverseMdct)().infer_shape)(t) == (20,)
    assert typing.cast("EagerCallableProtocol", InverseMdct)().infer_shape(typing.cast("EagerCallableProtocol", DummyNode)(())) == ()

    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", Frame)().infer_shape)(t, frame_length=2, frame_step=2) == (5, 2)
    assert typing.cast("EagerCallableProtocol", Frame)().infer_shape(typing.cast("EagerCallableProtocol", DummyNode)(())) == ()

    tf = DummyNode((5, 2))
    assert typing.cast("EagerCallableProtocol", typing.cast("EagerCallableProtocol", OverlapAndAdd)().infer_shape)(tf, frame_step=2) == (10,)
    assert typing.cast("EagerCallableProtocol", OverlapAndAdd)().infer_shape(typing.cast("EagerCallableProtocol", DummyNode)(())) == ()
