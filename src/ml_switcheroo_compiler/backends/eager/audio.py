"""Audio utilities."""

import typing
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


def _to_numpy_array(np_mod: object, x: object, name: str) -> object:
    if name == "torch":
        return x.detach().cpu().numpy()
    if name == "mlx.core":
        return np_mod.array(x)
    if hasattr(x, "numpy"):
        return x.numpy()
    return np_mod.asarray(x)


def _from_numpy_array(
    backend_module: object, out: object, name: str, original_tensor: object = None
) -> object:
    if name == "torch":
        import torch

        return (
            torch.tensor(out, dtype=torch.float32, device=original_tensor.device)
            if original_tensor is not None
            else torch.tensor(out, dtype=torch.float32)
        )
    if name == "mlx.core":
        import mlx.core as mx

        return mx.array(out, dtype=mx.float32)
    if name == "jax.numpy":
        import jax.numpy as jnp

        return jnp.array(out, dtype=jnp.float32)
    if name == "keras.ops":
        return backend_module.convert_to_tensor(out, dtype="float32")
    return __import__("numpy").asarray(out, dtype=__import__("numpy").float32)


def _get_window(np_mod: object, window: str, frame_length: int) -> object:
    import scipy.signal

    if window == "hann":
        return scipy.signal.windows.hann(frame_length, sym=False)
    elif window == "hamming":
        return scipy.signal.windows.hamming(frame_length, sym=False)
    return np_mod.ones(frame_length)


def _apply_istft_batch(
    np_mod: object,
    stft_np: object,
    win: object,
    config: STFTConfig,
    center: bool,
) -> object:
    import scipy.signal

    frame_length = config.frame_length
    frame_step = config.frame_step
    fft_length = config.fft_length if config.fft_length is not None else frame_length

    original_shape = stft_np.shape
    stft_np_flat = stft_np.reshape(-1, original_shape[-2], original_shape[-1])
    out_signals = []

    for i in range(stft_np_flat.shape[0]):
        _, rec = scipy.signal.istft(
            stft_np_flat[i],
            window=win,
            nperseg=frame_length,
            noverlap=frame_length - frame_step,
            nfft=fft_length,
            return_onesided=True,
            boundary=True if center else False,
        )
        out_signals.append(rec)

    out = np_mod.stack(out_signals, axis=0)
    return out.reshape(*original_shape[:-2], -1)


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
    out = _apply_istft_batch(np_mod, stft_np, win, config, center)

    return _from_numpy_array(backend_module, out, name, stft_tensor)


def mel_filterbank_eager(
    backend_module: object,
    _: object,
    config: MelFilterbankConfig,
) -> object:
    """Evaluate mel_filterbank eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    num_mel_bins = config["num_mel_bins"]
    num_spectrogram_bins = config["num_spectrogram_bins"]
    sample_rate = config["sample_rate"]
    lower_edge_hertz = config.get("lower_edge_hertz", DEFAULT_LOWER_EDGE_HERTZ)
    upper_edge_hertz = config.get("upper_edge_hertz", DEFAULT_UPPER_EDGE_HERTZ)

    is_torch = name == "torch"
    is_mlx = name == "mlx.core"

    if name == "keras.ops":
        import tensorflow as tf

        res = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
        )
        return backend_module.convert_to_tensor(res)

    def hz_to_mel(hz: float) -> float:
        return MEL_SCALE_MULTIPLIER * np_mod.log10(1.0 + hz / MEL_SCALE_DIVISOR)

    def mel_to_hz(mel: float) -> float:
        return MEL_SCALE_DIVISOR * (10.0 ** (mel / MEL_SCALE_MULTIPLIER) - 1.0)

    mel_low = hz_to_mel(lower_edge_hertz)
    mel_high = hz_to_mel(upper_edge_hertz)

    mel_pts = np_mod.linspace(mel_low, mel_high, num_mel_bins + 2)
    hz_pts = mel_to_hz(mel_pts)

    bin_freqs = np_mod.linspace(0, sample_rate / 2, num_spectrogram_bins)

    weights = np_mod.zeros((num_spectrogram_bins, num_mel_bins), dtype=np_mod.float32)
    for i in range(num_mel_bins):
        lower, center, upper = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up_slope = (bin_freqs - lower) / (center - lower)
        down_slope = (upper - bin_freqs) / (upper - center)
        weights[:, i] = np_mod.maximum(0.0, np_mod.minimum(up_slope, down_slope))

    if is_torch:
        import torch

        return torch.tensor(weights, dtype=torch.float32)
    if is_mlx:
        import mlx.core as mx

        return mx.array(weights, dtype=mx.float32)
    if name == "jax.numpy":
        import jax.numpy as jnp

        return jnp.array(weights, dtype=jnp.float32)

    return np_mod.asarray(weights, dtype=np_mod.float32)


def mfcc_eager(
    backend_module: object,
    spectrogram: object,
    config: MFCCConfig,
) -> object:
    """Evaluate mfcc eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    sample_rate = config["sample_rate"]
    num_mel_bins = config.get("num_mel_bins", 40)
    lower_edge_hertz = config.get("lower_edge_hertz", 20.0)
    upper_edge_hertz = config.get("upper_edge_hertz", DEFAULT_UPPER_EDGE_HERTZ)
    num_mfccs = config.get("num_mfccs", 13)

    is_torch = name == "torch"
    is_mlx = name == "mlx.core"

    def to_np(x: object) -> object:
        if is_torch:
            return x.detach().cpu().numpy()
        if is_mlx:
            return np_mod.array(x)
        if hasattr(x, "numpy"):
            return x.numpy()
        return np_mod.asarray(x)

    if name == "keras.ops":
        import tensorflow as tf

        spec_tf = tf.convert_to_tensor(spectrogram)
        num_spectrogram_bins = spec_tf.shape[-1]
        mel_weights = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
        )
        mel_spectrogram = tf.matmul(spec_tf, mel_weights)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_mfccs]
        return backend_module.convert_to_tensor(mfccs)

    spec_np = to_np(spectrogram)
    num_spectrogram_bins = spec_np.shape[-1]

    mel_config: MelFilterbankConfig = {
        "num_mel_bins": num_mel_bins,
        "num_spectrogram_bins": num_spectrogram_bins,
        "sample_rate": sample_rate,
        "lower_edge_hertz": lower_edge_hertz,
        "upper_edge_hertz": upper_edge_hertz,
    }

    mel_weights = mel_filterbank_eager(
        __import__("numpy"),
        None,
        mel_config,
    )

    mel_spec = np_mod.matmul(spec_np, mel_weights)
    log_mel_spec = np_mod.log(mel_spec + 1e-6)

    import scipy.fftpack

    mfccs = scipy.fftpack.dct(log_mel_spec, type=2, axis=-1, norm="ortho")[..., :num_mfccs]

    if is_torch:
        import torch

        return torch.tensor(mfccs, dtype=torch.float32, device=spectrogram.device)
    if is_mlx:
        import mlx.core as mx

        return mx.array(mfccs, dtype=mx.float32)
    if name == "jax.numpy":
        import jax.numpy as jnp

        return jnp.array(mfccs, dtype=jnp.float32)

    return np_mod.asarray(mfccs, dtype=np_mod.float32)


def _apply_stft_batch(
    np_mod: object,
    audio_np: object,
    win: object,
    config: STFTConfig,
) -> object:
    import scipy.signal

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

    # stft returns complex values
    if name == "torch":
        import torch

        if hasattr(input_tensor, "device"):
            return torch.tensor(out, dtype=torch.complex64, device=input_tensor.device)
        return torch.tensor(out, dtype=torch.complex64)
    if name == "mlx.core":
        import mlx.core as mx

        return mx.array(out, dtype=mx.complex64)
    if name == "jax.numpy":
        import jax.numpy as jnp

        return jnp.array(out, dtype=jnp.complex64)
    if name == "keras.ops":
        return backend_module.convert_to_tensor(out, dtype="complex64")
    return np_mod.asarray(out, dtype=np_mod.complex64)
