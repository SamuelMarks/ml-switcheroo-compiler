"""Audio signal processing utilities and eager mathematical kernels."""

from __future__ import annotations

import math
import types

import pydantic

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_float,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.ops.configs import STFTConfig

MEL_SCALE_MULTIPLIER: float = 2595.0

MEL_SCALE_DIVISOR: float = 700.0

DEFAULT_LOWER_EDGE_HERTZ: float = 0.0

DEFAULT_UPPER_EDGE_HERTZ: float = 4000.0


class MelFilterbankConfig(pydantic.BaseModel):
    """Mel filterbank configuration model.

    Attributes:
        num_mel_bins: Number of output Mel frequency bins.
        num_spectrogram_bins: Number of input linear spectrogram bins.
        sample_rate: Sampling frequency in Hertz.
        lower_edge_hertz: Lower frequency boundary in Hertz.
        upper_edge_hertz: Upper frequency boundary in Hertz.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    num_mel_bins: int = 40
    num_spectrogram_bins: int = 129
    sample_rate: int = 16000
    lower_edge_hertz: float = DEFAULT_LOWER_EDGE_HERTZ
    upper_edge_hertz: float = DEFAULT_UPPER_EDGE_HERTZ


class MFCCConfig(MelFilterbankConfig):
    """MFCC computation configuration model.

    Attributes:
        num_mfccs: Number of Mel-frequency cepstral coefficients to extract.
    """

    num_mfccs: int = 13


def _resolve_module(module: BackendModuleProtocol | types.ModuleType | None) -> BackendModuleProtocol:
    """Validate and resolve the provided execution backend module.

    Args:
        module: Execution backend module.

    Returns:
        BackendModuleProtocol: Resolved backend module.
    """
    from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod

    return _resolve_eager_mod(module)


def _get_window(
    np_mod: BackendModuleProtocol | types.ModuleType | None, window: str, frame_length: int
) -> EagerTensorProtocol:
    """Generate a mathematical window vector of the specified length and type.

    Args:
        np_mod: Execution backend module.
        window: Name of the window function ('hann', 'hamming', 'blackman', 'bartlett', 'rectangular').
        frame_length: Positive integer representing the number of points in the window.

    Returns:
        EagerTensorProtocol: 1-D array containing the window coefficients.
    """
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if frame_length <= 0:
        return mod.zeros((0,), dtype=mod.float32)
    if frame_length == 1:
        return mod.ones((1,), dtype=mod.float32)

    n: EagerTensorProtocol = mod.arange(frame_length, dtype=mod.float32)
    w_norm: str = window.lower()
    denom: float = float(frame_length - 1)

    if w_norm == "hann":
        out: EagerTensorProtocol = 0.5 - 0.5 * mod.cos(2.0 * math.pi * n / denom)
    elif w_norm == "hamming":
        out = 0.54 - 0.46 * mod.cos(2.0 * math.pi * n / denom)
    elif w_norm == "blackman":
        val: EagerTensorProtocol = (
            0.42 - 0.5 * mod.cos(2.0 * math.pi * n / denom) + 0.08 * mod.cos(4.0 * math.pi * n / denom)
        )
        out = mod.maximum(0.0, val)
    elif w_norm == "bartlett":
        out = 1.0 - 2.0 * mod.abs(n - denom / 2.0) / denom
    else:
        out = mod.ones((frame_length,), dtype=mod.float32)
    return out


def _hz_to_mel(np_mod: BackendModuleProtocol | types.ModuleType | None, hz: float | BackendArg) -> float | BackendArg:
    """Convert frequencies from Hertz to the perceptual Mel scale.

    Args:
        np_mod: Execution backend module.
        hz: Scalar frequency or array of frequencies in Hertz.

    Returns:
        float | BackendArg: Frequency value or array transformed to the Mel scale.
    """
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if isinstance(hz, (int, float)):
        return float(MEL_SCALE_MULTIPLIER * math.log10(1.0 + float(hz) / MEL_SCALE_DIVISOR))
    hz_arr: EagerTensorProtocol = mod.asarray(hz, dtype=mod.float32)
    return MEL_SCALE_MULTIPLIER * mod.log10(1.0 + hz_arr / MEL_SCALE_DIVISOR)


def _mel_to_hz(mel: float | BackendArg, np_mod: BackendModuleProtocol | None = None) -> float | BackendArg:
    """Convert frequencies from the perceptual Mel scale back to Hertz.

    Args:
        mel: Scalar or array of frequency values in Mel units.
        np_mod: Optional execution backend module for array conversion.

    Returns:
        float | BackendArg: Frequency values converted back to Hertz.
    """
    if isinstance(mel, (int, float)):
        return float(MEL_SCALE_DIVISOR * (10.0 ** (float(mel) / MEL_SCALE_MULTIPLIER) - 1.0))
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    mel_arr = (
        _resolve_module(np_mod).asarray(mel, dtype=_resolve_module(np_mod).float32)
        if np_mod is not None
        else coerce_numeric(mel)
    )
    return MEL_SCALE_DIVISOR * (10.0 ** (mel_arr / MEL_SCALE_MULTIPLIER) - 1.0)


def _compute_filterbank_weights(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    num_spectrogram_bins: int,
    num_mel_bins: int,
    bin_freqs: BackendArg,
    hz_pts: BackendArg,
) -> BackendArg:
    """Compute triangular weighting matrix mapping linear FFT bins to Mel frequency bins.

    Args:
        np_mod: Optional execution backend module.
        num_spectrogram_bins: Number of linear frequency bins.
        num_mel_bins: Number of Mel frequency bands.
        bin_freqs: Array of center frequencies for linear FFT bins.
        hz_pts: Array of boundary and peak frequencies in Hertz for each Mel band.

    Returns:
        BackendArg: Weight matrix of shape (num_spectrogram_bins, num_mel_bins).
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array, coerce_float

    mod: BackendModuleProtocol = _resolve_module(np_mod)
    weights = mod.zeros((num_spectrogram_bins, num_mel_bins), dtype=mod.float32)
    hz_pts_arr = coerce_array(mod, hz_pts)
    bin_freqs_arr = coerce_array(mod, bin_freqs)

    for m in range(num_mel_bins):
        lower: float = coerce_float(hz_pts_arr[m])
        center: float = coerce_float(hz_pts_arr[m + 1])
        upper: float = coerce_float(hz_pts_arr[m + 2])

        up_denom: float = max(center - lower, 1e-12)
        down_denom: float = max(upper - center, 1e-12)

        up_slope: BackendArg = (bin_freqs_arr - lower) / up_denom
        down_slope: BackendArg = (upper - bin_freqs_arr) / down_denom

        triangular: BackendArg = mod.maximum(0.0, mod.minimum(up_slope, down_slope))
        weights[:, m] = triangular

    return weights


def _generate_mel_filterbank_matrix(
    np_mod: BackendModuleProtocol | types.ModuleType | None, config: MelFilterbankConfig | dict[str, int | float]
) -> BackendArg:
    """Generate the linear-to-Mel filterbank transformation matrix.

    Args:
        np_mod: Optional execution backend module.
        config: Mel filterbank parameter configuration.

    Returns:
        BackendArg: Filterbank matrix of shape (num_spectrogram_bins, num_mel_bins).
    """
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    cfg = config if isinstance(config, MelFilterbankConfig) else MelFilterbankConfig.model_validate(config)
    num_mel_bins: int = cfg.num_mel_bins
    num_spectrogram_bins: int = cfg.num_spectrogram_bins
    sample_rate: int = cfg.sample_rate
    lower_edge_hertz: float = float(cfg.lower_edge_hertz)
    upper_edge_hertz: float = float(cfg.upper_edge_hertz)

    bin_freqs: BackendArg = mod.linspace(0.0, float(sample_rate) / 2.0, num_spectrogram_bins, dtype=mod.float32)

    mel_min: float = float(coerce_float(_hz_to_mel(mod, lower_edge_hertz)))
    mel_max: float = float(coerce_float(_hz_to_mel(mod, upper_edge_hertz)))
    mel_points: BackendArg = mod.linspace(mel_min, mel_max, num_mel_bins + 2, dtype=mod.float32)
    hz_pts: BackendArg = mod.asarray(_mel_to_hz(mel_points, mod), dtype=mod.float32)

    return _compute_filterbank_weights(mod, num_spectrogram_bins, num_mel_bins, bin_freqs, hz_pts)


def mel_filterbank_eager(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    spectrogram: BackendArg | None,
    config: MelFilterbankConfig | dict[str, int | float],
) -> BackendArg:
    """Apply Mel filterbank matrix transformation to input linear spectrograms.

    Args:
        backend_module: Execution backend module.
        spectrogram: Optional input spectrogram tensor of shape (..., num_spectrogram_bins).
        config: Mel filterbank parameter configuration.

    Returns:
        BackendArg: Mel-scaled spectrogram or raw filterbank weighting matrix.
    """
    mod: BackendModuleProtocol = _resolve_module(backend_module)
    matrix: BackendArg = _generate_mel_filterbank_matrix(mod, config)
    if spectrogram is None:
        return matrix
    spec_arr: BackendArg = mod.asarray(spectrogram, dtype=mod.float32)
    return mod.matmul(spec_arr, matrix)


def _power_to_db(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    mel_spec: BackendArg | None,
    amin: float = 1e-10,
    top_db: float = 80.0,
) -> BackendArg:
    """Convert power spectrogram to decibel (dB) representation with dynamic range clipping.

    Args:
        np_mod: Execution backend module.
        mel_spec: Spectrogram tensor with power values.
        amin: Minimum clipping value to avoid division by zero or log of zero.
        top_db: Maximum dynamic range threshold below peak power.

    Returns:
        BackendArg: Log-scaled decibel spectrogram.
    """
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if mel_spec is None:
        return mod.zeros((0,), dtype=mod.float32)
    spec_arr: BackendArg = mod.asarray(mel_spec, dtype=mod.float32)
    log_spec: BackendArg = 10.0 * mod.log10(mod.maximum(spec_arr, amin))
    if top_db > 0.0:
        max_val: float = float(mod.max(log_spec))
        log_spec = mod.maximum(log_spec, max_val - top_db)
    return log_spec


def _apply_dct(
    log_mel_spec: BackendArg | None,
    num_mfccs: int,
    np_mod: BackendModuleProtocol | None = None,
) -> BackendArg:
    """Apply Type-II Discrete Cosine Transform (DCT-II) along the innermost frequency axis.

    Args:
        log_mel_spec: Log-mel spectrogram array of shape (..., num_mel_bins).
        num_mfccs: Number of low-frequency cepstral coefficients to retain.
        np_mod: Execution backend module.

    Returns:
        BackendArg: Mel-frequency cepstral coefficients tensor.
    """
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if log_mel_spec is None:
        return mod.zeros((0,), dtype=mod.float32)
    spec: EagerTensorProtocol = mod.asarray(log_mel_spec, dtype=mod.float32)
    spec_shape = spec.shape
    n_bins: int = spec_shape[-1] if spec.ndim > 0 else 1
    if n_bins == 0 or num_mfccs <= 0:
        out_shape: tuple[int, ...] = (*spec_shape[:-1], 0) if spec.ndim > 0 else (0,)
        return mod.zeros(out_shape, dtype=mod.float32)

    n_indices: EagerTensorProtocol = mod.arange(n_bins, dtype=mod.float32)
    k_indices: EagerTensorProtocol = mod.arange(num_mfccs, dtype=mod.float32)[:, None]

    basis: EagerTensorProtocol = mod.cos((math.pi * k_indices * (2.0 * n_indices + 1.0)) / (2.0 * float(n_bins)))
    norm: EagerTensorProtocol = mod.full((num_mfccs, 1), math.sqrt(2.0 / float(n_bins)), dtype=mod.float32)
    norm[0, 0] = math.sqrt(1.0 / float(n_bins))
    basis = basis * norm

    return mod.matmul(spec, basis.T)


def _mfcc_eager_tf(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    spectrogram: BackendArg | None,
    config: MFCCConfig | dict[str, int | float],
) -> BackendArg:
    """Compute MFCCs conforming to standard execution specifications.

    Args:
        backend_module: Execution backend module.
        spectrogram: Spectrogram array.
        config: MFCC parameter configuration.

    Returns:
        BackendArg: Extracted Mel-frequency cepstral coefficients.
    """
    mod: BackendModuleProtocol = _resolve_module(backend_module)
    cfg = config if isinstance(config, MFCCConfig) else MFCCConfig.model_validate(config)
    num_mfccs: int = cfg.num_mfccs
    num_spectrogram_bins: int = cfg.num_spectrogram_bins

    spec_arr: BackendArg
    if spectrogram is None:
        spec_arr = mod.ones((1, num_spectrogram_bins), dtype=mod.float32)
    else:
        spec_arr = mod.asarray(spectrogram, dtype=mod.float32)

    mel_spec: BackendArg = mel_filterbank_eager(mod, spec_arr, cfg)
    log_mel_spec: BackendArg = _power_to_db(mod, mel_spec)
    return _apply_dct(log_mel_spec, num_mfccs, mod)


def _convert_to_np(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg | None,
    is_torch: bool,
    is_mlx: bool,
) -> BackendArg:
    """Convert tensor to backend array across various framework bindings.

    Args:
        np_mod: Execution backend module.
        x: Input tensor or array-like object.
        is_torch: Boolean flag indicating if input originated from PyTorch.
        is_mlx: Boolean flag indicating if input originated from Apple MLX.

    Returns:
        BackendArg: Converted backend array.
    """
    del is_torch, is_mlx
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if x is None:
        return mod.zeros((0,), dtype=mod.float32)
    np_fn = adapt_backend_module(x).get_symbol("numpy", None)
    if callable(np_fn):
        return mod.asarray(np_fn())
    return mod.asarray(x)


def _to_backend_tensor(
    name: str,
    mfccs: BackendArg | None,
    spectrogram: BackendArg | None,
    np_mod: BackendModuleProtocol | None,
) -> BackendArg:
    """Wrap computed NumPy array back to target backend representation.

    Args:
        name: Name of operation or output variable.
        mfccs: Extracted MFCC features array.
        spectrogram: Original input spectrogram.
        np_mod: Target execution backend module.

    Returns:
        BackendArg: Backend array representation.
    """
    del name, spectrogram
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if mfccs is None:
        return mod.zeros((0,), dtype=mod.float32)
    return mod.asarray(mfccs)


def mfcc_eager(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    spectrogram: BackendArg | None,
    config: MFCCConfig | dict[str, int | float],
) -> BackendArg:
    """Eager computation entrypoint for extracting MFCCs from audio spectrograms.

    Args:
        backend_module: Optional execution backend module.
        spectrogram: Input linear or power spectrogram.
        config: MFCC configuration options.

    Returns:
        BackendArg: Extracted Mel-frequency cepstral coefficients.
    """
    return _mfcc_eager_tf(backend_module, spectrogram, config)


def _apply_stft_batch(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    audio_np: BackendArg | None,
    win: BackendArg | None,
    config: STFTConfig,
) -> BackendArg:
    """Compute Short-Time Fourier Transform over a batch of 1-D or 2-D audio signals.

    Args:
        np_mod: Optional execution backend module.
        audio_np: Input audio waveform array of shape (num_samples,) or (batch_size, num_samples).
        win: 1-D window vector of length equal to frame_length.
        config: STFT configuration parameters.

    Returns:
        BackendArg: Complex frequency-domain STFT tensor of shape (..., num_frames, fft_length // 2 + 1).
    """
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if audio_np is None:
        return mod.zeros((0, 0), dtype=mod.complex64)

    audio: EagerTensorProtocol = mod.asarray(audio_np, dtype=mod.float32)
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
            zeros_pad: BackendArg = mod.zeros((audio.shape[0], pad_amount), dtype=audio.dtype)
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

    window_vec: EagerTensorProtocol
    if win is not None and adapt_backend_module(win).get_symbol("size", -1) == frame_length:
        window_vec = mod.asarray(win, dtype=mod.float32)
    else:
        window_vec = _get_window(mod, config.window_fn or "hann", frame_length)

    stft_out: EagerTensorProtocol = mod.zeros((batch_size, num_frames, fft_length // 2 + 1), dtype=mod.complex64)

    for i in range(num_frames):
        start: int = i * frame_step
        frame: BackendArg = audio[:, start : start + frame_length] * window_vec
        stft_out[:, i, :] = mod.fft.rfft(frame, n=fft_length, axis=-1)

    if orig_1d:
        return stft_out[0]
    return stft_out


def _to_backend_tensor_complex(
    name: str,
    out: BackendArg | None,
    np_mod: BackendModuleProtocol | None,
    backend_module: BackendModuleProtocol | None,
) -> BackendArg:
    """Convert complex NumPy STFT output back to the appropriate backend tensor.

    Args:
        name: Name of operation or output variable.
        out: Complex STFT array.
        np_mod: Backend module.
        backend_module: Target framework module.

    Returns:
        BackendArg: Backend complex tensor.
    """
    del name, backend_module
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if out is None:
        return mod.zeros((0,), dtype=mod.complex64)
    return mod.asarray(out)


def stft_eager(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    input_tensor: BackendArg | None,
    config: STFTConfig,
) -> BackendArg:
    """Eager computation entrypoint for computing the Short-Time Fourier Transform.

    Args:
        backend_module: Optional execution backend module.
        input_tensor: Input audio waveform tensor.
        config: STFT configuration parameters.

    Returns:
        BackendArg: Computed STFT representation.
    """
    mod: BackendModuleProtocol = _resolve_module(backend_module)
    win: BackendArg = _get_window(mod, config.window_fn or "hann", config.frame_length)
    return _apply_stft_batch(mod, input_tensor, win, config)


def _run_scipy_istft(
    stft_np_flat: BackendArg | None,
    win: BackendArg | None,
    frame_params: tuple[int, int, int],
    center: bool,
    np_mod: BackendModuleProtocol | None = None,
) -> BackendArg:
    """Execute overlap-add reconstruction to recover the time-domain signal from an STFT matrix.

    Args:
        stft_np_flat: 2-D complex STFT array of shape (num_frames, fft_bins).
        win: Optional 1-D window array.
        frame_params: Tuple containing (frame_length, frame_step, fft_length).
        center: Boolean flag indicating if centering padding was applied during STFT.
        np_mod: Optional execution backend module.

    Returns:
        BackendArg: Reconstructed 1-D time-domain waveform.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_shape

    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if stft_np_flat is None or adapt_backend_module(stft_np_flat).get_symbol("size", 0) == 0:
        return mod.zeros((0,), dtype=mod.float32)

    frame_length, frame_step, fft_length = frame_params
    num_frames: int = coerce_shape(adapt_backend_module(stft_np_flat).require_symbol("shape"))[0]

    window_vec: EagerTensorProtocol
    if win is not None and adapt_backend_module(win).get_symbol("size", -1) == frame_length:
        window_vec = mod.asarray(win, dtype=mod.float32)
    else:
        window_vec = _get_window(mod, "hann", frame_length)

    expected_samples: int = (num_frames - 1) * frame_step + frame_length
    time_signal: EagerTensorProtocol = mod.zeros((expected_samples,), dtype=mod.float32)
    window_sum: EagerTensorProtocol = mod.zeros((expected_samples,), dtype=mod.float32)

    frames_time: EagerTensorProtocol = mod.fft.irfft(stft_np_flat, n=fft_length, axis=-1)
    frames_time = frames_time[:, :frame_length]

    win_sq: BackendArg = window_vec**2

    for i in range(num_frames):
        start: int = i * frame_step
        time_signal[start : start + frame_length] += frames_time[i] * window_vec
        window_sum[start : start + frame_length] += win_sq

    non_zero: BackendArg = window_sum > 1e-10
    time_signal[non_zero] /= window_sum[non_zero]

    if center and expected_samples > frame_length:
        half_len: int = frame_length // 2
        return time_signal[half_len:-half_len]
    return time_signal


def _apply_istft_batch(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    stft_np: BackendArg | None,
    win: BackendArg | None,
    config: STFTConfig,
    center: bool = True,
) -> BackendArg:
    """Compute batched inverse Short-Time Fourier Transform.

    Args:
        np_mod: Optional execution backend module.
        stft_np: Input complex STFT tensor of shape (..., num_frames, fft_bins).
        win: Optional 1-D synthesis window array.
        config: STFT parameter configuration.
        center: Boolean flag indicating if centering padding was used.

    Returns:
        BackendArg: Reconstructed real-valued audio waveform array.
    """
    mod: BackendModuleProtocol = _resolve_module(np_mod)
    if stft_np is None or adapt_backend_module(stft_np).get_symbol("size", 0) == 0:
        return mod.zeros((0,), dtype=mod.float32)

    stft_arr: EagerTensorProtocol = mod.asarray(stft_np, dtype=mod.complex64)
    orig_2d: bool = stft_arr.ndim == 2

    if orig_2d:
        stft_arr = stft_arr[None, ...]

    batch_size: int = stft_arr.shape[0]
    frame_length: int = config.frame_length
    frame_step: int = config.frame_step
    fft_length: int = config.fft_length if config.fft_length is not None else frame_length
    params: tuple[int, int, int] = (frame_length, frame_step, fft_length)

    reconstructed_list: list[BackendArg] = []
    for b in range(batch_size):
        rec: BackendArg = _run_scipy_istft(stft_arr[b], win, params, center, mod)
        reconstructed_list.append(rec)

    reconstructed: EagerTensorProtocol = mod.asarray(reconstructed_list, dtype=mod.float32)
    if orig_2d:
        return reconstructed[0]
    return reconstructed


def istft_eager(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    stft_tensor: BackendArg | None,
    config: STFTConfig,
    center: bool = True,
) -> BackendArg:
    """Eager computation entrypoint for the Inverse Short-Time Fourier Transform.

    Args:
        backend_module: Optional execution backend module.
        stft_tensor: Complex STFT input tensor.
        config: STFT configuration parameters.
        center: Boolean flag indicating if centering padding was applied.

    Returns:
        BackendArg: Reconstructed audio waveform.
    """
    mod: BackendModuleProtocol = _resolve_module(backend_module)
    win: BackendArg = _get_window(mod, config.window_fn or "hann", config.frame_length)
    return _apply_istft_batch(mod, stft_tensor, win, config, center=center)
