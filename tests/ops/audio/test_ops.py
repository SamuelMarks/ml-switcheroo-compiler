"""Test module."""

from ml_switcheroo_compiler.ops.audio.ops import Dct, Frame, HammingWindow, HannWindow, Idct, InverseMdct, Istft, KaiserWindow, Mdct, MelFilterbank, MelSpectrogram, Mfcc, MfccsFromLogMelSpectrograms, OverlapAndAdd, Stft


class DummyNode:
    def __init__(self, shape):
        self.shape = shape


def test_audio_ops():
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
    assert Dct().infer_shape(t) == (10,)
    assert Idct().infer_shape(t) == (10,)

    assert Mdct().infer_shape(t) == (5,)
    assert Mdct().infer_shape(DummyNode(())) == ()

    assert InverseMdct().infer_shape(t) == (20,)
    assert InverseMdct().infer_shape(DummyNode(())) == ()

    assert Frame().infer_shape(t, frame_length=2, frame_step=2) == (5, 2)
    assert Frame().infer_shape(DummyNode(())) == ()

    tf = DummyNode((5, 2))
    assert OverlapAndAdd().infer_shape(tf, frame_step=2) == (10,)
    assert OverlapAndAdd().infer_shape(DummyNode(())) == ()
