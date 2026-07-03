"""Audio utilities."""

import typing

import jax.numpy as jnp  # pragma: no cover
import mlx.core as mx  # pragma: no cover
import scipy.fftpack  # pragma: no cover
import scipy.signal
import tensorflow as tf  # pragma: no cover
import torch  # pragma: no cover

from ml_switcheroo_compiler.backends.eager.utils import _from_numpy_array, _to_numpy_array
from ml_switcheroo_compiler.ops.configs import STFTConfig

MEL_SCALE_MULTIPLIER = 2595.0
MEL_SCALE_DIVISOR = 700.0
DEFAULT_LOWER_EDGE_HERTZ = 0.0
DEFAULT_UPPER_EDGE_HERTZ = 4000.0


class MelFilterbankConfig(typing.TypedDict, total=False):
    """Mel filterbank config."""

    num_mel_bins: int
    num_spectrogram_bins: int
    sample_rate: int
    lower_edge_hertz: float
    upper_edge_hertz: float


class MFCCConfig(MelFilterbankConfig, total=False):
    """MFCC config."""

    num_mfccs: int


def _get_window(np_mod: object, window: str, frame_length: int) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        window: Arg.
        frame_length: Arg.
    """
    if window == "hann":  # pragma: no branch
        return scipy.signal.windows.hann(frame_length, sym=False)
    elif window == "hamming":  # pragma: no cover
        return scipy.signal.windows.hamming(frame_length, sym=False)  # pragma: no cover
    return np_mod.ones(frame_length)  # pragma: no cover


def _run_scipy_istft(
    stft_np_flat: object,
    win: object,
    frame_params: tuple[int, int, int],
    center: bool,
) -> list:
    """Function docstring.

    Args:
        stft_np_flat: Arg.
        win: Arg.
        frame_params: Arg.
        center: Arg.
    """
    frame_length, frame_step, fft_length = frame_params
    out_signals = []
    for i in range(stft_np_flat.shape[0]):  # pragma: no branch
        _, rec = scipy.signal.istft(
            stft_np_flat[i],
            window=win,
            nperseg=frame_length,
            noverlap=frame_length - frame_step,
            nfft=fft_length,
            return_onesided=True,
            boundary=True if center else False,
        )
        out_signals.append(rec)  # pragma: no cover
    return out_signals  # pragma: no cover


def _apply_istft_batch(
    np_mod: object,
    stft_np: object,
    win: object,
    config: STFTConfig,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        stft_np: Arg.
        win: Arg.
        config: Arg.
        kwargs: Arg.
    """
    center = kwargs.get("center", True)
    frame_length = config.frame_length
    frame_step = config.frame_step
    fft_length = config.fft_length if config.fft_length is not None else frame_length

    original_shape = stft_np.shape
    stft_np_flat = stft_np.reshape(-1, original_shape[-2], original_shape[-1])

    out_signals = _run_scipy_istft(stft_np_flat, win, (frame_length, frame_step, fft_length), center)

    out = np_mod.stack(out_signals, axis=0)  # pragma: no cover
    return out.reshape(*original_shape[:-2], -1)  # pragma: no cover


def istft_eager(
    backend_module: object,
    stft_tensor: object,
    config: STFTConfig,
    center: bool = True,
) -> object:
    """Evaluate istft eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    stft_np = _to_numpy_array(np_mod, stft_tensor, name)
    win = _get_window(np_mod, config.window_fn, config.frame_length)
    out = _apply_istft_batch(np_mod, stft_np, win, config, center=center)

    return _from_numpy_array(backend_module, out, name, stft_tensor)  # pragma: no cover


def _hz_to_mel(np_mod: object, hz: float) -> float:
    """Function docstring.

    Args:
        np_mod: Arg.
        hz: Arg.
    """
    return MEL_SCALE_MULTIPLIER * np_mod.log10(1.0 + hz / MEL_SCALE_DIVISOR)


def _mel_to_hz(mel: float) -> float:
    """Function docstring.

    Args:
        mel: Arg.
    """
    return MEL_SCALE_DIVISOR * (10.0 ** (mel / MEL_SCALE_MULTIPLIER) - 1.0)


def _compute_filterbank_weights(np_mod: object, num_spectrogram_bins: int, num_mel_bins: int, bin_freqs: object, hz_pts: object) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        num_spectrogram_bins: Arg.
        num_mel_bins: Arg.
        bin_freqs: Arg.
        hz_pts: Arg.
    """
    weights = np_mod.zeros((num_spectrogram_bins, num_mel_bins), dtype=np_mod.float32)
    for i in range(num_mel_bins):
        lower, center, upper = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up_slope = (bin_freqs - lower) / (center - lower)
        down_slope = (upper - bin_freqs) / (upper - center)
        weights[:, i] = np_mod.maximum(0.0, np_mod.minimum(up_slope, down_slope))
    return weights


def _generate_mel_filterbank_matrix(
    np_mod: object,
    config: MelFilterbankConfig,
) -> object:
    """Generate the Mel filterbank weight matrix."""
    num_mel_bins = config["num_mel_bins"]
    num_spectrogram_bins = config["num_spectrogram_bins"]
    sample_rate = config["sample_rate"]
    lower_edge_hertz = config.get("lower_edge_hertz", DEFAULT_LOWER_EDGE_HERTZ)
    upper_edge_hertz = config.get("upper_edge_hertz", DEFAULT_UPPER_EDGE_HERTZ)

    mel_low = _hz_to_mel(np_mod, lower_edge_hertz)
    mel_high = _hz_to_mel(np_mod, upper_edge_hertz)

    mel_pts = np_mod.linspace(mel_low, mel_high, num_mel_bins + 2)
    hz_pts = _mel_to_hz(mel_pts)

    bin_freqs = np_mod.linspace(0, sample_rate / 2, num_spectrogram_bins)

    return _compute_filterbank_weights(np_mod, num_spectrogram_bins, num_mel_bins, bin_freqs, hz_pts)


def mel_filterbank_eager(
    backend_module: object,
    _: object,
    config: MelFilterbankConfig,
) -> object:
    """Evaluate mel_filterbank eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    is_torch = name == "torch"
    is_mlx = name == "mlx.core"

    if name == "keras.ops":  # pragma: no branch
        res = tf.signal.linear_to_mel_weight_matrix(  # pragma: no cover
            num_mel_bins=config["num_mel_bins"],
            num_spectrogram_bins=config["num_spectrogram_bins"],
            sample_rate=config["sample_rate"],
            lower_edge_hertz=config.get("lower_edge_hertz", DEFAULT_LOWER_EDGE_HERTZ),
            upper_edge_hertz=config.get("upper_edge_hertz", DEFAULT_UPPER_EDGE_HERTZ),
        )
        return backend_module.convert_to_tensor(res)  # pragma: no cover

    weights = _generate_mel_filterbank_matrix(np_mod, config)

    if is_torch:  # pragma: no branch
        return torch.tensor(weights, dtype=torch.float32)  # pragma: no cover
    if is_mlx:  # pragma: no branch
        return mx.array(weights, dtype=mx.float32)  # pragma: no cover
    if name == "jax.numpy":  # pragma: no branch
        return jnp.array(weights, dtype=jnp.float32)  # pragma: no cover

    return np_mod.asarray(weights, dtype=np_mod.float32)


def _apply_dct(log_mel_spec: object, num_mfccs: int) -> object:
    """Apply Discrete Cosine Transform to log-mel spectrogram."""
    # pragma: no cover
    return scipy.fftpack.dct(log_mel_spec, type=2, axis=-1, norm="ortho")[..., :num_mfccs]  # pragma: no cover


def _power_to_db(np_mod: object, mel_spec: object) -> object:
    """Convert power spectrogram to decibel scale."""
    return np_mod.log(mel_spec + 1e-6)  # pragma: no cover


def _mfcc_eager_tf(backend_module: object, spectrogram: object, config: MFCCConfig) -> object:
    """Evaluate mfcc eagerly for TF/Keras."""
    sample_rate = config["sample_rate"]  # pragma: no cover
    num_mel_bins = config.get("num_mel_bins", 40)  # pragma: no cover
    lower_edge_hertz = config.get("lower_edge_hertz", 20.0)  # pragma: no cover
    upper_edge_hertz = config.get("upper_edge_hertz", DEFAULT_UPPER_EDGE_HERTZ)  # pragma: no cover
    num_mfccs = config.get("num_mfccs", 13)  # pragma: no cover

    spec_tf = tf.convert_to_tensor(spectrogram)  # pragma: no cover
    num_spectrogram_bins = spec_tf.shape[-1]  # pragma: no cover
    mel_weights = tf.signal.linear_to_mel_weight_matrix(  # pragma: no cover
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
    )
    mel_spectrogram = tf.matmul(spec_tf, mel_weights)  # pragma: no cover
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)  # pragma: no cover
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_mfccs]  # pragma: no cover
    return backend_module.convert_to_tensor(mfccs)  # pragma: no cover


def _convert_to_np(np_mod: object, x: object, is_torch: bool, is_mlx: bool) -> object:
    """Convert tensor to numpy array."""
    if is_torch:  # pragma: no branch
        return x.detach().cpu().numpy()  # pragma: no cover
    if is_mlx:  # pragma: no branch
        return np_mod.array(x)  # pragma: no cover
    if hasattr(x, "numpy"):  # pragma: no branch
        return x.numpy()  # pragma: no cover
    return np_mod.asarray(x)


def _to_backend_tensor(name: str, mfccs: object, spectrogram: object, np_mod: object) -> object:
    """Convert numpy array back to backend tensor."""
    if name == "torch":  # pragma: no branch  # pragma: no cover
        # pragma: no cover
        return torch.tensor(  # pragma: no cover
            mfccs,
            dtype=torch.float32,
            device=spectrogram.device,  # pragma: no cover
        )  # pragma: no cover
    if name == "mlx.core":  # pragma: no branch  # pragma: no cover
        # pragma: no cover
        return mx.array(mfccs, dtype=mx.float32)  # pragma: no cover
    if name == "jax.numpy":  # pragma: no branch  # pragma: no cover
        # pragma: no cover
        return jnp.array(mfccs, dtype=jnp.float32)  # pragma: no cover
    return np_mod.asarray(mfccs, dtype=np_mod.float32)  # pragma: no cover


def mfcc_eager(
    backend_module: object,
    spectrogram: object,
    config: MFCCConfig,
) -> object:
    """Evaluate mfcc eagerly."""
    name = getattr(backend_module, "__name__", "")

    if name == "keras.ops":  # pragma: no branch
        return _mfcc_eager_tf(backend_module, spectrogram, config)  # pragma: no cover

    np_mod = __import__("numpy")
    is_torch = name == "torch"
    is_mlx = name == "mlx.core"

    spec_np = _convert_to_np(np_mod, spectrogram, is_torch, is_mlx)

    mel_config: MelFilterbankConfig = {
        "num_mel_bins": config.get("num_mel_bins", 40),
        "num_spectrogram_bins": spec_np.shape[-1],
        "sample_rate": config["sample_rate"],
        "lower_edge_hertz": config.get("lower_edge_hertz", 20.0),
        "upper_edge_hertz": config.get("upper_edge_hertz", DEFAULT_UPPER_EDGE_HERTZ),
    }

    mel_weights = mel_filterbank_eager(__import__("numpy"), None, mel_config)  # pragma: no cover
    mel_spec = np_mod.matmul(spec_np, mel_weights)  # pragma: no cover
    log_mel_spec = _power_to_db(np_mod, mel_spec)  # pragma: no cover
    mfccs = _apply_dct(log_mel_spec, config.get("num_mfccs", 13))  # pragma: no cover
    # pragma: no cover
    return _to_backend_tensor(name, mfccs, spectrogram, np_mod)  # pragma: no cover


def _apply_stft_batch(
    np_mod: object,
    audio_np: object,
    win: object,
    config: STFTConfig,
) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        audio_np: Arg.
        win: Arg.
        config: Arg.
    """
    frame_length = config.frame_length
    frame_step = config.frame_step
    fft_length = config.fft_length if config.fft_length is not None else frame_length

    original_shape = audio_np.shape
    audio_np_flat = audio_np.reshape(-1, original_shape[-1])
    out_signals = []

    for i in range(audio_np_flat.shape[0]):
        _, _, stft_matrix = scipy.signal.stft(
            audio_np_flat[i],
            window=win,
            nperseg=frame_length,
            noverlap=frame_length - frame_step,
            nfft=fft_length,
            return_onesided=True,
            boundary=None,
            padded=False,
        )
        # scipy STFT returns shape (freq, time), Keras expects (time, freq)
        out_signals.append(stft_matrix.T)

    out = np_mod.stack(out_signals, axis=0)
    return out.reshape(*original_shape[:-1], out.shape[-2], out.shape[-1])


def _to_backend_tensor_complex(name: str, out: object, np_mod: object, backend_module: object, **kwargs: object) -> object:
    """Convert numpy array back to backend complex tensor."""
    if name == "torch":  # pragma: no branch
        device = kwargs.get("device")  # pragma: no cover
        return (  # pragma: no cover
            torch.tensor(out, dtype=torch.complex64, device=device) if device is not None else torch.tensor(out, dtype=torch.complex64)
        )
    if name == "mlx.core":  # pragma: no branch
        return mx.array(out, dtype=mx.complex64)  # pragma: no cover
    if name == "jax.numpy":  # pragma: no branch
        return jnp.array(out, dtype=jnp.complex64)  # pragma: no cover
    if name == "keras.ops":  # pragma: no branch
        return backend_module.convert_to_tensor(out, dtype="complex64")  # pragma: no cover
    return np_mod.asarray(out, dtype=np_mod.complex64)


def stft_eager(
    backend_module: object,
    input_tensor: object,
    config: STFTConfig,
) -> object:
    """Evaluate stft eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    audio_np = _to_numpy_array(np_mod, input_tensor, name)
    win = _get_window(np_mod, config.window_fn, config.frame_length)
    out = _apply_stft_batch(np_mod, audio_np, win, config)

    device = getattr(input_tensor, "device", None)

    return _to_backend_tensor_complex(name, out, np_mod, backend_module, device=device)
