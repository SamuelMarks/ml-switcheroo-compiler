"""Audio signal processing utilities and eager mathematical kernels."""

from __future__ import annotations

import math
import typing
from types import ModuleType

import numpy as np

from ml_switcheroo_compiler.ops.configs import STFTConfig

MEL_SCALE_MULTIPLIER: float = 2595.0
MEL_SCALE_DIVISOR: float = 700.0
DEFAULT_LOWER_EDGE_HERTZ: float = 0.0
DEFAULT_UPPER_EDGE_HERTZ: float = 4000.0


class MelFilterbankConfig(typing.TypedDict, total=False):
    """Mel filterbank configuration dictionary.

    Attributes:
        num_mel_bins: Number of output Mel frequency bins.
        num_spectrogram_bins: Number of input linear spectrogram bins.
        sample_rate: Sampling frequency in Hertz.
        lower_edge_hertz: Lower frequency boundary in Hertz.
        upper_edge_hertz: Upper frequency boundary in Hertz.
    """

    num_mel_bins: int
    num_spectrogram_bins: int
    sample_rate: int
    lower_edge_hertz: float
    upper_edge_hertz: float


class MFCCConfig(MelFilterbankConfig, total=False):
    """MFCC computation configuration dictionary.

    Attributes:
        num_mfccs: Number of Mel-frequency cepstral coefficients to extract.
    """

    num_mfccs: int


def _resolve_module(module: ModuleType | None) -> ModuleType:
    """Resolve backend module to NumPy if unspecified.

    Args:
        module: Optional execution backend module.

    Returns:
        ModuleType: Resolved backend module.
    """
    if module is None:
        return np
    return module


def _get_window(np_mod: ModuleType | None, window: str, frame_length: int) -> np.ndarray:
    """Generate a mathematical window vector of the specified length and type.

    Args:
        np_mod: Optional execution backend module.
        window: Name of the window function ('hann', 'hamming', 'blackman', 'bartlett', 'rectangular').
        frame_length: Positive integer representing the number of points in the window.

    Returns:
        np.ndarray: 1-D array containing the window coefficients.
    """
    mod: ModuleType = _resolve_module(np_mod)
    if frame_length <= 0:
        return mod.zeros((0,), dtype=mod.float32)
    if frame_length == 1:
        return mod.ones((1,), dtype=mod.float32)

    n: np.ndarray = mod.arange(frame_length, dtype=mod.float32)
    w_norm: str = window.lower()
    denom: float = float(frame_length - 1)

    if w_norm == "hann":
        out: np.ndarray = 0.5 - 0.5 * mod.cos(2.0 * math.pi * n / denom)
    elif w_norm == "hamming":
        out = 0.54 - 0.46 * mod.cos(2.0 * math.pi * n / denom)
    elif w_norm == "blackman":
        val: np.ndarray = 0.42 - 0.5 * mod.cos(2.0 * math.pi * n / denom) + 0.08 * mod.cos(4.0 * math.pi * n / denom)
        out = mod.maximum(0.0, val)
    elif w_norm == "bartlett":
        out = 1.0 - 2.0 * mod.abs(n - denom / 2.0) / denom
    else:
        out = mod.ones((frame_length,), dtype=mod.float32)
    return out


def _hz_to_mel(np_mod: ModuleType | None, hz: float | np.ndarray) -> float | np.ndarray:
    """Convert frequencies from Hertz to the perceptual Mel scale.

    Args:
        np_mod: Optional execution backend module.
        hz: Scalar frequency or array of frequencies in Hertz.

    Returns:
        float | np.ndarray: Frequency value or array transformed to the Mel scale.
    """
    mod: ModuleType = _resolve_module(np_mod)
    if isinstance(hz, (int, float)):
        return float(MEL_SCALE_MULTIPLIER * math.log10(1.0 + float(hz) / MEL_SCALE_DIVISOR))
    hz_arr: np.ndarray = mod.asarray(hz, dtype=mod.float32)
    return MEL_SCALE_MULTIPLIER * mod.log10(1.0 + hz_arr / MEL_SCALE_DIVISOR)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert frequencies from the perceptual Mel scale back to Hertz.

    Args:
        mel: Scalar or array of frequency values in Mel units.

    Returns:
        float | np.ndarray: Frequency values converted back to Hertz.
    """
    if isinstance(mel, (int, float)):
        return float(MEL_SCALE_DIVISOR * (10.0 ** (float(mel) / MEL_SCALE_MULTIPLIER) - 1.0))
    mel_arr: np.ndarray = np.asarray(mel, dtype=np.float32)
    return MEL_SCALE_DIVISOR * (10.0 ** (mel_arr / MEL_SCALE_MULTIPLIER) - 1.0)


def _compute_filterbank_weights(
    np_mod: ModuleType | None,
    num_spectrogram_bins: int,
    num_mel_bins: int,
    bin_freqs: np.ndarray,
    hz_pts: np.ndarray,
) -> np.ndarray:
    """Compute triangular weighting matrix mapping linear FFT bins to Mel frequency bins.

    Args:
        np_mod: Optional execution backend module.
        num_spectrogram_bins: Number of linear frequency bins.
        num_mel_bins: Number of Mel frequency bands.
        bin_freqs: Array of center frequencies for linear FFT bins.
        hz_pts: Array of boundary and peak frequencies in Hertz for each Mel band.

    Returns:
        np.ndarray: Weight matrix of shape (num_spectrogram_bins, num_mel_bins).
    """
    mod: ModuleType = _resolve_module(np_mod)
    weights: np.ndarray = mod.zeros((num_spectrogram_bins, num_mel_bins), dtype=mod.float32)

    for m in range(num_mel_bins):
        lower: float = float(hz_pts[m])
        center: float = float(hz_pts[m + 1])
        upper: float = float(hz_pts[m + 2])

        up_denom: float = max(center - lower, 1e-12)
        down_denom: float = max(upper - center, 1e-12)

        up_slope: np.ndarray = (bin_freqs - lower) / up_denom
        down_slope: np.ndarray = (upper - bin_freqs) / down_denom

        triangular: np.ndarray = mod.maximum(0.0, mod.minimum(up_slope, down_slope))
        weights[:, m] = triangular

    return weights


def _generate_mel_filterbank_matrix(np_mod: ModuleType | None, config: MelFilterbankConfig) -> np.ndarray:
    """Generate the linear-to-Mel filterbank transformation matrix.

    Args:
        np_mod: Optional execution backend module.
        config: Mel filterbank parameter configuration.

    Returns:
        np.ndarray: Filterbank matrix of shape (num_spectrogram_bins, num_mel_bins).
    """
    mod: ModuleType = _resolve_module(np_mod)
    num_mel_bins: int = config.get("num_mel_bins", 40)
    num_spectrogram_bins: int = config.get("num_spectrogram_bins", 129)
    sample_rate: int = config.get("sample_rate", 16000)
    lower_edge_hertz: float = float(config.get("lower_edge_hertz", DEFAULT_LOWER_EDGE_HERTZ))
    upper_edge_hertz: float = float(config.get("upper_edge_hertz", DEFAULT_UPPER_EDGE_HERTZ))

    bin_freqs: np.ndarray = mod.linspace(0.0, float(sample_rate) / 2.0, num_spectrogram_bins, dtype=mod.float32)

    mel_min: float = float(_hz_to_mel(mod, lower_edge_hertz))
    mel_max: float = float(_hz_to_mel(mod, upper_edge_hertz))
    mel_points: np.ndarray = mod.linspace(mel_min, mel_max, num_mel_bins + 2, dtype=mod.float32)
    hz_pts: np.ndarray = mod.asarray(_mel_to_hz(mel_points), dtype=mod.float32)

    return _compute_filterbank_weights(mod, num_spectrogram_bins, num_mel_bins, bin_freqs, hz_pts)


def mel_filterbank_eager(
    backend_module: ModuleType | None,
    spectrogram: np.ndarray | None,
    config: MelFilterbankConfig,
) -> np.ndarray:
    """Apply Mel filterbank matrix transformation to input linear spectrograms.

    Args:
        backend_module: Optional execution backend module.
        spectrogram: Optional input spectrogram tensor of shape (..., num_spectrogram_bins).
        config: Mel filterbank parameter configuration.

    Returns:
        np.ndarray: Mel-scaled spectrogram or raw filterbank weighting matrix.
    """
    mod: ModuleType = _resolve_module(backend_module)
    matrix: np.ndarray = _generate_mel_filterbank_matrix(mod, config)
    if spectrogram is None:
        return matrix
    spec_arr: np.ndarray = mod.asarray(spectrogram, dtype=mod.float32)
    return mod.matmul(spec_arr, matrix)


def _power_to_db(
    np_mod: ModuleType | None,
    mel_spec: np.ndarray | None,
    amin: float = 1e-10,
    top_db: float = 80.0,
) -> np.ndarray:
    """Convert power spectrogram to decibel (dB) representation with dynamic range clipping.

    Args:
        np_mod: Optional execution backend module.
        mel_spec: Spectrogram tensor with power values.
        amin: Minimum clipping value to avoid division by zero or log of zero.
        top_db: Maximum dynamic range threshold below peak power.

    Returns:
        np.ndarray: Log-scaled decibel spectrogram.
    """
    mod: ModuleType = _resolve_module(np_mod)
    if mel_spec is None:
        return mod.zeros((0,), dtype=mod.float32)
    spec_arr: np.ndarray = mod.asarray(mel_spec, dtype=mod.float32)
    log_spec: np.ndarray = 10.0 * mod.log10(mod.maximum(spec_arr, amin))
    if top_db > 0.0:
        max_val: float = float(mod.max(log_spec))
        log_spec = mod.maximum(log_spec, max_val - top_db)
    return log_spec


def _apply_dct(log_mel_spec: np.ndarray | None, num_mfccs: int) -> np.ndarray:
    """Apply Type-II Discrete Cosine Transform (DCT-II) along the innermost frequency axis.

    Args:
        log_mel_spec: Log-mel spectrogram array of shape (..., num_mel_bins).
        num_mfccs: Number of low-frequency cepstral coefficients to retain.

    Returns:
        np.ndarray: Mel-frequency cepstral coefficients tensor.
    """
    if log_mel_spec is None:
        return np.zeros((0,), dtype=np.float32)
    spec: np.ndarray = np.asarray(log_mel_spec, dtype=np.float32)
    n_bins: int = spec.shape[-1] if spec.ndim > 0 else 1
    if n_bins == 0 or num_mfccs <= 0:
        out_shape: tuple[int, ...] = (*spec.shape[:-1], 0) if spec.ndim > 0 else (0,)
        return np.zeros(out_shape, dtype=np.float32)

    n_indices: np.ndarray = np.arange(n_bins, dtype=np.float32)
    k_indices: np.ndarray = np.arange(num_mfccs, dtype=np.float32)[:, None]

    basis: np.ndarray = np.cos((math.pi * k_indices * (2.0 * n_indices + 1.0)) / (2.0 * float(n_bins)))
    norm: np.ndarray = np.full((num_mfccs, 1), math.sqrt(2.0 / float(n_bins)), dtype=np.float32)
    norm[0, 0] = math.sqrt(1.0 / float(n_bins))
    basis = basis * norm

    return np.matmul(spec, basis.T)


def _mfcc_eager_tf(
    backend_module: ModuleType | None,
    spectrogram: np.ndarray | None,
    config: MFCCConfig,
) -> np.ndarray:
    """Compute MFCCs conforming to standard execution specifications.

    Args:
        backend_module: Optional execution backend module.
        spectrogram: Spectrogram array.
        config: MFCC parameter configuration.

    Returns:
        np.ndarray: Extracted Mel-frequency cepstral coefficients.
    """
    mod: ModuleType = _resolve_module(backend_module)
    num_mfccs: int = config.get("num_mfccs", 13)
    num_spectrogram_bins: int = config.get("num_spectrogram_bins", 129)

    spec_arr: np.ndarray
    if spectrogram is None:
        spec_arr = mod.ones((1, num_spectrogram_bins), dtype=mod.float32)
    else:
        spec_arr = mod.asarray(spectrogram, dtype=mod.float32)

    mel_spec: np.ndarray = mel_filterbank_eager(mod, spec_arr, config)
    log_mel_spec: np.ndarray = _power_to_db(mod, mel_spec)
    return _apply_dct(log_mel_spec, num_mfccs)


def _convert_to_np(
    np_mod: ModuleType | None,
    x: np.ndarray | None,
    is_torch: bool,
    is_mlx: bool,
) -> np.ndarray:
    """Convert tensor to NumPy ndarray across various framework bindings.

    Args:
        np_mod: Optional execution backend module.
        x: Input tensor or array-like object.
        is_torch: Boolean flag indicating if input originated from PyTorch.
        is_mlx: Boolean flag indicating if input originated from Apple MLX.

    Returns:
        np.ndarray: Converted NumPy ndarray.
    """
    del is_torch, is_mlx
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())  # type: ignore[no-untyped-call]
    return np.asarray(x)


def _to_backend_tensor(
    name: str,
    mfccs: np.ndarray | None,
    spectrogram: np.ndarray | None,
    np_mod: ModuleType | None,
) -> np.ndarray:
    """Wrap computed NumPy array back to target backend representation.

    Args:
        name: Name of operation or output variable.
        mfccs: Extracted MFCC features array.
        spectrogram: Original input spectrogram.
        np_mod: Target execution backend module.

    Returns:
        np.ndarray: Backend array representation.
    """
    del name, spectrogram
    mod: ModuleType = _resolve_module(np_mod)
    if mfccs is None:
        return mod.zeros((0,), dtype=mod.float32)
    return mod.asarray(mfccs)


def mfcc_eager(
    backend_module: ModuleType | None,
    spectrogram: np.ndarray | None,
    config: MFCCConfig,
) -> np.ndarray:
    """Eager computation entrypoint for extracting MFCCs from audio spectrograms.

    Args:
        backend_module: Optional execution backend module.
        spectrogram: Input linear or power spectrogram.
        config: MFCC configuration options.

    Returns:
        np.ndarray: Extracted Mel-frequency cepstral coefficients.
    """
    return _mfcc_eager_tf(backend_module, spectrogram, config)


def _apply_stft_batch(
    np_mod: ModuleType | None,
    audio_np: np.ndarray | None,
    win: np.ndarray | None,
    config: STFTConfig,
) -> np.ndarray:
    """Compute Short-Time Fourier Transform over a batch of 1-D or 2-D audio signals.

    Args:
        np_mod: Optional execution backend module.
        audio_np: Input audio waveform array of shape (num_samples,) or (batch_size, num_samples).
        win: 1-D window vector of length equal to frame_length.
        config: STFT configuration parameters.

    Returns:
        np.ndarray: Complex frequency-domain STFT tensor of shape (..., num_frames, fft_length // 2 + 1).
    """
    mod: ModuleType = _resolve_module(np_mod)
    if audio_np is None:
        return mod.zeros((0, 0), dtype=mod.complex64)

    audio: np.ndarray = mod.asarray(audio_np, dtype=mod.float32)
    orig_1d: bool = audio.ndim == 1
    if orig_1d:
        audio = audio[None, :]

    batch_size: int = audio.shape[0]
    total_samples: int = audio.shape[1]
    frame_length: int = config.frame_length
    frame_step: int = config.frame_step
    fft_length: int = config.fft_length if config.fft_length is not None else frame_length

    if total_samples < frame_length:
        if config.pad_end:
            pad_amount: int = int(frame_length - total_samples)
            zeros_pad: np.ndarray = mod.zeros((audio.shape[0], pad_amount), dtype=audio.dtype)
            audio = mod.concatenate([audio, zeros_pad], axis=1)
            total_samples = audio.shape[1]
        else:
            empty_shape: tuple[int, ...] = (0, fft_length // 2 + 1) if orig_1d else (batch_size, 0, fft_length // 2 + 1)
            return mod.zeros(empty_shape, dtype=mod.complex64)

    if config.pad_end:
        remainder: int = int((total_samples - frame_length) % frame_step)
        if remainder != 0:
            pad_needed: int = int(frame_step - remainder)
            zeros_pad = mod.zeros((audio.shape[0], pad_needed), dtype=audio.dtype)
            audio = mod.concatenate([audio, zeros_pad], axis=1)
            total_samples = audio.shape[1]

    num_frames: int = 1 + (total_samples - frame_length) // frame_step

    window_vec: np.ndarray
    if win is not None and win.size == frame_length:
        window_vec = mod.asarray(win, dtype=mod.float32)
    else:
        window_vec = _get_window(mod, config.window_fn or "hann", frame_length)

    stft_out: np.ndarray = mod.zeros((batch_size, num_frames, fft_length // 2 + 1), dtype=mod.complex64)

    for i in range(num_frames):
        start: int = i * frame_step
        frame: np.ndarray = audio[:, start : start + frame_length] * window_vec
        stft_out[:, i, :] = mod.fft.rfft(frame, n=fft_length, axis=-1)

    if orig_1d:
        return stft_out[0]
    return stft_out


def _to_backend_tensor_complex(
    name: str,
    out: np.ndarray | None,
    np_mod: ModuleType | None,
    backend_module: ModuleType | None,
) -> np.ndarray:
    """Convert complex NumPy STFT output back to the appropriate backend tensor.

    Args:
        name: Name of operation or output variable.
        out: Complex STFT array.
        np_mod: Backend module.
        backend_module: Target framework module.

    Returns:
        np.ndarray: Backend complex tensor.
    """
    del name, backend_module
    mod: ModuleType = _resolve_module(np_mod)
    if out is None:
        return mod.zeros((0,), dtype=mod.complex64)
    return mod.asarray(out)


def stft_eager(
    backend_module: ModuleType | None,
    input_tensor: np.ndarray | None,
    config: STFTConfig,
) -> np.ndarray:
    """Eager computation entrypoint for computing the Short-Time Fourier Transform.

    Args:
        backend_module: Optional execution backend module.
        input_tensor: Input audio waveform tensor.
        config: STFT configuration parameters.

    Returns:
        np.ndarray: Computed STFT representation.
    """
    mod: ModuleType = _resolve_module(backend_module)
    win: np.ndarray = _get_window(mod, config.window_fn or "hann", config.frame_length)
    return _apply_stft_batch(mod, input_tensor, win, config)


def _run_scipy_istft(
    stft_np_flat: np.ndarray | None,
    win: np.ndarray | None,
    frame_params: tuple[int, int, int],
    center: bool,
) -> np.ndarray:
    """Execute overlap-add reconstruction to recover the time-domain signal from an STFT matrix.

    Args:
        stft_np_flat: 2-D complex STFT array of shape (num_frames, fft_bins).
        win: Optional 1-D window array.
        frame_params: Tuple containing (frame_length, frame_step, fft_length).
        center: Boolean flag indicating if centering padding was applied during STFT.

    Returns:
        np.ndarray: Reconstructed 1-D time-domain waveform.
    """
    if stft_np_flat is None or stft_np_flat.size == 0:
        return np.zeros((0,), dtype=np.float32)

    frame_length, frame_step, fft_length = frame_params
    num_frames: int = stft_np_flat.shape[0]

    window_vec: np.ndarray
    if win is not None and win.size == frame_length:
        window_vec = np.asarray(win, dtype=np.float32)
    else:
        window_vec = _get_window(np, "hann", frame_length)

    expected_samples: int = (num_frames - 1) * frame_step + frame_length
    time_signal: np.ndarray = np.zeros((expected_samples,), dtype=np.float32)
    window_sum: np.ndarray = np.zeros((expected_samples,), dtype=np.float32)

    frames_time: np.ndarray = np.fft.irfft(stft_np_flat, n=fft_length, axis=-1)
    frames_time = frames_time[:, :frame_length]

    win_sq: np.ndarray = window_vec**2

    for i in range(num_frames):
        start: int = i * frame_step
        time_signal[start : start + frame_length] += frames_time[i] * window_vec
        window_sum[start : start + frame_length] += win_sq

    non_zero: np.ndarray = window_sum > 1e-10
    time_signal[non_zero] /= window_sum[non_zero]

    if center and expected_samples > frame_length:
        half_len: int = frame_length // 2
        return time_signal[half_len:-half_len]
    return time_signal


def _apply_istft_batch(
    np_mod: ModuleType | None,
    stft_np: np.ndarray | None,
    win: np.ndarray | None,
    config: STFTConfig,
    center: bool = True,
) -> np.ndarray:
    """Compute batched inverse Short-Time Fourier Transform.

    Args:
        np_mod: Optional execution backend module.
        stft_np: Input complex STFT tensor of shape (..., num_frames, fft_bins).
        win: Optional 1-D synthesis window array.
        config: STFT parameter configuration.
        center: Boolean flag indicating if centering padding was used.

    Returns:
        np.ndarray: Reconstructed real-valued audio waveform array.
    """
    mod: ModuleType = _resolve_module(np_mod)
    if stft_np is None or stft_np.size == 0:
        return mod.zeros((0,), dtype=mod.float32)

    stft_arr: np.ndarray = mod.asarray(stft_np, dtype=mod.complex64)
    orig_2d: bool = stft_arr.ndim == 2

    if orig_2d:
        stft_arr = stft_arr[None, ...]

    batch_size: int = stft_arr.shape[0]
    frame_length: int = config.frame_length
    frame_step: int = config.frame_step
    fft_length: int = config.fft_length if config.fft_length is not None else frame_length
    params: tuple[int, int, int] = (frame_length, frame_step, fft_length)

    reconstructed_list: list[np.ndarray] = []
    for b in range(batch_size):
        rec: np.ndarray = _run_scipy_istft(stft_arr[b], win, params, center)
        reconstructed_list.append(rec)

    reconstructed: np.ndarray = mod.asarray(reconstructed_list, dtype=mod.float32)
    if orig_2d:
        return reconstructed[0]
    return reconstructed


def istft_eager(
    backend_module: ModuleType | None,
    stft_tensor: np.ndarray | None,
    config: STFTConfig,
    center: bool = True,
) -> np.ndarray:
    """Eager computation entrypoint for the Inverse Short-Time Fourier Transform.

    Args:
        backend_module: Optional execution backend module.
        stft_tensor: Complex STFT input tensor.
        config: STFT configuration parameters.
        center: Boolean flag indicating if centering padding was applied.

    Returns:
        np.ndarray: Reconstructed audio waveform.
    """
    mod: ModuleType = _resolve_module(backend_module)
    win: np.ndarray = _get_window(mod, config.window_fn or "hann", config.frame_length)
    return _apply_istft_batch(mod, stft_tensor, win, config, center=center)
