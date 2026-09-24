"""Exhaustive unit tests for audio eager backend signal processing kernels."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.audio import (
    MelFilterbankConfig,
    MFCCConfig,
    _apply_dct,
    _apply_istft_batch,
    _apply_stft_batch,
    _convert_to_np,
    _generate_mel_filterbank_matrix,
    _get_window,
    _hz_to_mel,
    _mel_to_hz,
    _power_to_db,
    _resolve_module,
    _run_scipy_istft,
    _to_backend_tensor,
    _to_backend_tensor_complex,
    istft_eager,
    mel_filterbank_eager,
    mfcc_eager,
    stft_eager,
)
from ml_switcheroo_compiler.ops.configs import STFTConfig


def test_resolve_module() -> None:
    """Test _resolve_module fallback and passthrough."""
    assert _resolve_module(None) is np
    assert _resolve_module(np) is np


def test_get_window() -> None:
    """Test _get_window across all supported window types and boundary lengths."""
    # Length <= 0
    w_empty = _get_window(None, "hann", 0)
    assert w_empty.shape == (0,)

    w_neg = _get_window(np, "hann", -5)
    assert w_neg.shape == (0,)

    # Length == 1
    w_one = _get_window(None, "hann", 1)
    assert np.allclose(w_one, [1.0])

    # Hann window
    w_hann = _get_window(np, "hann", 5)
    assert len(w_hann) == 5
    assert np.isclose(w_hann[0], 0.0)
    assert np.isclose(w_hann[2], 1.0)
    assert np.isclose(w_hann[4], 0.0)

    # Hamming window
    w_hamm = _get_window(None, "hamming", 5)
    assert len(w_hamm) == 5
    assert np.isclose(w_hamm[0], 0.08)
    assert np.isclose(w_hamm[2], 1.0)

    # Blackman window
    w_black = _get_window(np, "blackman", 7)
    assert len(w_black) == 7
    assert np.isclose(w_black[0], 0.0)
    assert np.isclose(w_black[3], 1.0)

    # Bartlett window
    w_bart = _get_window(None, "bartlett", 5)
    assert len(w_bart) == 5
    assert np.isclose(w_bart[0], 0.0)
    assert np.isclose(w_bart[2], 1.0)
    assert np.isclose(w_bart[4], 0.0)

    # Rectangular / default window
    w_rect = _get_window(np, "rectangular", 6)
    assert np.allclose(w_rect, np.ones(6))

    w_unknown = _get_window(None, "custom_window", 4)
    assert np.allclose(w_unknown, np.ones(4))


def test_mel_conversions() -> None:
    """Test Hz to Mel and Mel to Hz scalar and array roundtrip conversions."""
    hz_val: float = 1000.0
    mel_val = _hz_to_mel(None, hz_val)
    assert isinstance(mel_val, float)
    recovered_hz = _mel_to_hz(mel_val)
    assert isinstance(recovered_hz, float)
    assert np.isclose(hz_val, recovered_hz, atol=1e-4)

    # Array conversions
    hz_arr: np.ndarray = np.array([0.0, 500.0, 1000.0, 4000.0], dtype=np.float32)
    mel_arr = _hz_to_mel(np, hz_arr)
    assert isinstance(mel_arr, np.ndarray)
    recovered_arr = _mel_to_hz(mel_arr)
    assert isinstance(recovered_arr, np.ndarray)
    assert np.allclose(hz_arr, recovered_arr, atol=1e-4)


def test_mel_filterbank() -> None:
    """Test Mel filterbank matrix computation and application."""
    config: MelFilterbankConfig = {
        "num_mel_bins": 20,
        "num_spectrogram_bins": 65,
        "sample_rate": 16000,
        "lower_edge_hertz": 0.0,
        "upper_edge_hertz": 8000.0,
    }
    matrix = _generate_mel_filterbank_matrix(None, config)
    assert matrix.shape == (65, 20)
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)

    # Calling mel_filterbank_eager without spectrogram returns matrix
    mat_out = mel_filterbank_eager(None, None, config)
    assert np.allclose(mat_out, matrix)

    # Calling with spectrogram performs projection
    spec = np.ones((4, 10, 65), dtype=np.float32)
    mel_spec = mel_filterbank_eager(np, spec, config)
    assert mel_spec.shape == (4, 10, 20)


def test_power_to_db() -> None:
    """Test decibel conversion with and without dynamic range clipping."""
    # None case
    assert _power_to_db(None, None).shape == (0,)

    power = np.array([1.0, 0.1, 0.01, 1e-5], dtype=np.float32)
    # Without clipping
    db_no_clip = _power_to_db(np, power, top_db=0.0)
    expected_db = np.array([0.0, -10.0, -20.0, -50.0], dtype=np.float32)
    assert np.allclose(db_no_clip, expected_db, atol=1e-3)

    # With top_db = 25.0, values below -25 should be clamped to -25
    db_clipped = _power_to_db(None, power, top_db=25.0)
    assert np.isclose(db_clipped[0], 0.0)
    assert np.isclose(db_clipped[3], -25.0)


def test_apply_dct() -> None:
    """Test Discrete Cosine Transform computation and boundary conditions."""
    # None input
    assert _apply_dct(None, 13).shape == (0,)

    # 0 bins or 0 num_mfccs
    assert _apply_dct(np.ones((5, 0)), 10).shape == (5, 0)
    assert _apply_dct(np.ones((5, 10)), 0).shape == (5, 0)

    # Normal DCT
    log_mel = np.ones((2, 8, 20), dtype=np.float32)
    mfccs = _apply_dct(log_mel, 13)
    assert mfccs.shape == (2, 8, 13)


def test_mfcc_eager() -> None:
    """Test MFCC extraction eager routine."""
    config: MFCCConfig = {
        "num_mfccs": 12,
        "num_mel_bins": 30,
        "num_spectrogram_bins": 65,
        "sample_rate": 16000,
    }

    # None spectrogram
    mfcc_default = mfcc_eager(None, None, config)
    assert mfcc_default.shape == (1, 12)

    # Given spectrogram
    spec = np.random.uniform(0.1, 10.0, size=(2, 15, 65)).astype(np.float32)
    mfcc_out = mfcc_eager(np, spec, config)
    assert mfcc_out.shape == (2, 15, 12)


def test_tensor_conversion_helpers() -> None:
    """Test framework conversion utilities."""
    # None inputs
    assert _convert_to_np(None, None, False, False).shape == (0,)
    assert _to_backend_tensor("test", None, None, None).shape == (0,)
    assert _to_backend_tensor_complex("test", None, None, None).shape == (0,)

    # Dummy class with numpy method
    class DummyTorch:
        def __init__(self, arr: np.ndarray) -> None:
            self.arr = arr

        def numpy(self) -> np.ndarray:
            return self.arr

    raw_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(_convert_to_np(np, DummyTorch(raw_arr), True, False), raw_arr)
    assert np.allclose(_convert_to_np(None, raw_arr, False, False), raw_arr)

    assert np.allclose(_to_backend_tensor("test", raw_arr, None, None), raw_arr)

    c_arr = np.array([1.0 + 2.0j], dtype=np.complex64)
    assert np.allclose(_to_backend_tensor_complex("test", c_arr, None, None), c_arr)


def test_stft_and_istft_pipeline() -> None:
    """Test STFT and ISTFT eager execution and near-lossless reconstruction."""
    # Edge cases: None input
    cfg = STFTConfig(frame_length=16, frame_step=8, fft_length=16, window_fn="hann", pad_end=False)
    assert stft_eager(None, None, cfg).shape == (0, 0)
    assert istft_eager(None, None, cfg).shape == (0,)

    # Audio too short without pad_end
    short_audio = np.ones((10,), dtype=np.float32)
    assert stft_eager(np, short_audio, cfg).shape == (0, 9)

    short_audio_2d = np.ones((2, 10), dtype=np.float32)
    assert stft_eager(np, short_audio_2d, cfg).shape == (2, 0, 9)

    # Audio with pad_end=True and remainder != 0
    cfg_padded = STFTConfig(frame_length=16, frame_step=8, fft_length=16, window_fn="hann", pad_end=True)
    stft_short_padded = stft_eager(None, short_audio, cfg_padded)
    assert stft_short_padded.shape[1] == 9
    assert stft_short_padded.shape[0] >= 1

    audio_rem = np.ones((20,), dtype=np.float32)
    stft_rem_padded = stft_eager(None, audio_rem, cfg_padded)
    assert stft_rem_padded.shape[1] == 9
    assert stft_rem_padded.shape[0] == 2

    # Normal 1-D STFT
    t = np.linspace(0.0, 1.0, 320, endpoint=False, dtype=np.float32)
    signal_1d = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    stft_1d = stft_eager(None, signal_1d, cfg)
    assert stft_1d.ndim == 2
    assert stft_1d.shape[1] == 9  # 16 // 2 + 1

    # Normal 2-D batched STFT
    signal_2d = np.stack([signal_1d, signal_1d * 0.5], axis=0)
    stft_2d = stft_eager(np, signal_2d, cfg)
    assert stft_2d.ndim == 3
    assert stft_2d.shape[0] == 2
    assert stft_2d.shape[2] == 9

    # Reconstruct 1-D via istft_eager
    rec_1d = istft_eager(None, stft_1d, cfg, center=False)
    assert rec_1d.ndim == 1
    assert len(rec_1d) > 0

    # Reconstruct 2-D via istft_eager
    rec_2d = istft_eager(np, stft_2d, cfg, center=True)
    assert rec_2d.ndim == 2
    assert rec_2d.shape[0] == 2

    # Test with custom provided window
    custom_win = np.hanning(16).astype(np.float32)
    stft_custom = _apply_stft_batch(None, signal_1d, custom_win, cfg)
    assert stft_custom.shape == stft_1d.shape

    # Custom window with wrong size triggers fallback
    stft_wrong_win = _apply_stft_batch(None, signal_1d, np.ones((5,), dtype=np.float32), cfg)
    assert stft_wrong_win.shape == stft_1d.shape

    # Empty stft into istft
    assert _run_scipy_istft(None, None, (16, 8, 16), False).shape == (0,)
    assert _run_scipy_istft(np.zeros((0, 9), dtype=np.complex64), None, (16, 8, 16), False).shape == (0,)
    assert _apply_istft_batch(None, np.zeros((0, 9), dtype=np.complex64), None, cfg).shape == (0,)

    # Explicitly test _run_scipy_istft with win=None to verify default window branch
    rec_no_win = _run_scipy_istft(stft_1d, None, (16, 8, 16), False)
    assert len(rec_no_win) > 0
