"""Test module."""

from ml_switcheroo_compiler.backends.eager.audio import (
    MelFilterbankConfig,
    MFCCConfig,
    _apply_dct,
    _apply_istft_batch,
    _apply_stft_batch,
    _compute_filterbank_weights,
    _convert_to_np,
    _generate_mel_filterbank_matrix,
    _get_window,
    _hz_to_mel,
    _mel_to_hz,
    _mfcc_eager_tf,
    _power_to_db,
    _run_scipy_istft,
    _to_backend_tensor,
    _to_backend_tensor_complex,
    istft_eager,
    mel_filterbank_eager,
    mfcc_eager,
    stft_eager,
)
from ml_switcheroo_compiler.ops.configs import STFTConfig


def test_audio():
    assert _get_window(None, "hann", 10) == 0
    assert _run_scipy_istft(None, None, (1, 2, 3), False) == 0
    assert _apply_istft_batch(None, None, None, STFTConfig(1, 2, 3)) == 0
    assert istft_eager(None, None, STFTConfig(1, 2, 3)) == 0

    assert _hz_to_mel(None, 100.0) == 0
    assert _mel_to_hz(0.0) == 0.0

    assert _compute_filterbank_weights(None, 10, 20, None, None) == 0
    assert _generate_mel_filterbank_matrix(None, MelFilterbankConfig(num_mel_bins=10)) == 0
    assert mel_filterbank_eager(None, None, MelFilterbankConfig(num_mel_bins=10)) == 0

    assert _apply_dct(None, 10) == 0
    assert _power_to_db(None, None) == 0
    assert _mfcc_eager_tf(None, None, MFCCConfig(num_mfccs=10)) == 0

    assert _convert_to_np(None, None, False, False) == 0
    assert _to_backend_tensor("test", None, None, None) == 0
    assert mfcc_eager(None, None, MFCCConfig(num_mfccs=10)) == 0

    assert _apply_stft_batch(None, None, None, STFTConfig(1, 2, 3)) == 0
    assert _to_backend_tensor_complex("test", None, None, None) == 0
    assert stft_eager(None, None, STFTConfig(1, 2, 3)) == 0
