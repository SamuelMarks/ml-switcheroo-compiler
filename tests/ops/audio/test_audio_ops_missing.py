from ml_switcheroo_compiler.ops.audio.ops import HammingWindow, HannWindow, Istft, KaiserWindow, MelSpectrogram, Mfcc, MfccsFromLogMelSpectrograms, Stft


def test_audio_ops_missing_branches():
    class DummyEmptyShape:
        shape = ()

    # Stft 22->32: len(shape) <= 0
    assert Stft().infer_shape(DummyEmptyShape()) == ()

    # MelSpectrogram 52->60: len(shape) <= 0
    assert MelSpectrogram().infer_shape(DummyEmptyShape()) == ()

    # Istft 80->87: len(shape) < 2
    class Dummy1DShape:
        shape = (10,)

    assert Istft().infer_shape(Dummy1DShape()) == (10,)

    # Mfcc 127->135: len(shape) <= 0
    assert Mfcc().infer_shape(DummyEmptyShape()) == ()

    # MfccsFromLogMelSpectrograms 155->158: len(shape) <= 0
    assert MfccsFromLogMelSpectrograms().infer_shape(DummyEmptyShape()) == ()

    # HannWindow, HammingWindow, KaiserWindow:
    # If len(args) > 0 but not isinstance(val, int)
    assert HannWindow().infer_shape("not_an_int") == (256,)
    assert HammingWindow().infer_shape("not_an_int") == (256,)
    assert KaiserWindow().infer_shape("not_an_int") == (256,)
