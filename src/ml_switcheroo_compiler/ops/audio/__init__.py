"""Audio operations module."""

from ml_switcheroo_compiler.ops.audio import ops
from ml_switcheroo_compiler.ops.audio.frontend import (
    dct,
    frame,
    hamming_window,
    hann_window,
    idct,
    inverse_mdct,
    inverse_stft,
    inverse_stft_window_fn,
    istft,
    kaiser_bessel_derived_window,
    kaiser_window,
    linear_to_mel_weight_matrix,
    mdct,
    mel_filterbank,
    mel_spectrogram,
    mfcc,
    mfccs_from_log_mel_spectrograms,
    overlap_and_add,
    stft,
    vorbis_window,
)

__all__ = [
    "dct",
    "frame",
    "hamming_window",
    "hann_window",
    "idct",
    "inverse_mdct",
    "inverse_stft",
    "inverse_stft_window_fn",
    "istft",
    "kaiser_bessel_derived_window",
    "kaiser_window",
    "linear_to_mel_weight_matrix",
    "mdct",
    "mel_filterbank",
    "mel_spectrogram",
    "mfcc",
    "mfccs_from_log_mel_spectrograms",
    "overlap_and_add",
    "stft",
    "vorbis_window",
]
_ = ops
