"""Test shape inference for audio operations."""

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.audio.ops import (
    Frame,
    HammingWindow,
    HannWindow,
    InverseMdct,
    Istft,
    KaiserWindow,
    Mdct,
    MelFilterbank,
    MelSpectrogram,
    Mfcc,
    MfccsFromLogMelSpectrograms,
    Stft,
)


def test_audio_shape_inferences() -> None:
    """Test shape inference algorithms for all audio operations."""
    # Create input signals
    t_1d = Tensor(np.ones(1024), TensorConfig((1024,), "float32", "cpu"))
    t_stft_out = Tensor(np.ones((7, 129)), TensorConfig((7, 129), "float32", "cpu"))
    t_mel_out = Tensor(np.ones((7, 128)), TensorConfig((7, 128), "float32", "cpu"))

    # Stft shape inference
    assert Stft().infer_shape(t_1d, frame_length=256, frame_step=128) == (7, 129)
    assert Stft().infer_shape() == ()

    # MelSpectrogram shape inference
    assert MelSpectrogram().infer_shape(t_1d, frame_length=256, frame_step=128, num_mel_bins=64) == (7, 64)
    assert MelSpectrogram().infer_shape() == ()

    # Istft shape inference
    assert Istft().infer_shape(t_stft_out, frame_length=256, frame_step=128) == (1024,)
    assert Istft().infer_shape() == ()

    # MelFilterbank shape inference
    assert MelFilterbank().infer_shape(fft_length=256, num_mel_bins=64) == (129, 64)

    # Mfcc shape inference
    assert Mfcc().infer_shape(t_1d, frame_length=256, frame_step=128, num_mfccs=13) == (7, 13)
    assert Mfcc().infer_shape() == ()

    # MfccsFromLogMelSpectrograms shape inference
    assert MfccsFromLogMelSpectrograms().infer_shape(t_mel_out, num_mfccs=15) == (7, 15)
    assert MfccsFromLogMelSpectrograms().infer_shape() == ()

    # Window ops shape inference
    assert HannWindow().infer_shape(100) == (100,)
    assert HannWindow().infer_shape(window_length=150) == (150,)

    assert HammingWindow().infer_shape(100) == (100,)
    assert HammingWindow().infer_shape(window_length=150) == (150,)

    assert KaiserWindow().infer_shape(100) == (100,)
    assert KaiserWindow().infer_shape(window_length=150) == (150,)

    # Mdct shape inference
    assert Mdct().infer_shape(t_1d, frame_length=256, frame_step=128) == (7, 128)
    assert Mdct().infer_shape(t_1d, frame_length=256, frame_step=128, pad_end=True) == (8, 128)
    assert Mdct().infer_shape() == ()
    # Pre-framed block (last dim == frame_length)
    t_pre_framed_mdct = Tensor(np.ones((7, 256)), TensorConfig((7, 256), "float32", "cpu"))
    assert Mdct().infer_shape(t_pre_framed_mdct, frame_length=256) == (7, 128)

    # InverseMdct shape inference
    t_spectral = Tensor(np.ones((7, 128)), TensorConfig((7, 128), "float32", "cpu"))
    # Pre-framed block (last dim == frame_length // 2)
    assert InverseMdct().infer_shape(t_spectral, frame_length=256) == (7, 256)
    # Continuous reconstruction (..., num_frames, frame_length // 2) -> (..., reconstructed_length)
    assert InverseMdct().infer_shape(t_spectral, frame_length=256, frame_step=128) == (1024,)
    assert InverseMdct().infer_shape() == ()

    # Frame shape inference
    assert Frame().infer_shape(t_1d, frame_length=256, frame_step=128) == (7, 256)
    assert Frame().infer_shape(t_1d, frame_length=256, frame_step=128, pad_end=True) == (8, 256)
    assert Frame().infer_shape() == ()
