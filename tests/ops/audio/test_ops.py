"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ops.audio.ops import Dct, Frame, HammingWindow, HannWindow, Idct, InverseMdct, Istft, KaiserWindow, Mdct, MelFilterbank, MelSpectrogram, Mfcc, MfccsFromLogMelSpectrograms, OverlapAndAdd, Stft

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__."""
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
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", Dct)().infer_shape)(t) == (10,)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", Idct)().infer_shape)(t) == (10,)

    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", Mdct)().infer_shape)(t) == (5,)
    assert cast("Callable[..., EagerTensorProtocol]", Mdct)().infer_shape(cast("Callable[..., EagerTensorProtocol]", DummyNode)(())) == ()

    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", InverseMdct)().infer_shape)(t) == (20,)
    assert cast("Callable[..., EagerTensorProtocol]", InverseMdct)().infer_shape(cast("Callable[..., EagerTensorProtocol]", DummyNode)(())) == ()

    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", Frame)().infer_shape)(t, frame_length=2, frame_step=2) == (5, 2)
    assert cast("Callable[..., EagerTensorProtocol]", Frame)().infer_shape(cast("Callable[..., EagerTensorProtocol]", DummyNode)(())) == ()

    tf = DummyNode((5, 2))
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", OverlapAndAdd)().infer_shape)(tf, frame_step=2) == (10,)
    assert cast("Callable[..., EagerTensorProtocol]", OverlapAndAdd)().infer_shape(cast("Callable[..., EagerTensorProtocol]", DummyNode)(())) == ()
