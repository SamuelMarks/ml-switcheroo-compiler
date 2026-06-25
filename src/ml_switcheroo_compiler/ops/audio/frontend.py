"""Audio and DSP operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend

from ml_switcheroo_compiler.backends.eager.audio import MFCCConfig
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import STFTConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def stft(input_tensor: Tensor, config_obj: STFTConfig | None = None, **kwargs: object) -> Tensor:
    """Computes the Short-Time Fourier Transform.

    Args:
        input_tensor (Tensor): Input audio tensor.
        config_obj (STFTConfig | None): STFT configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: STFT results.
    """
    if config_obj is None:  # pragma: no branch
        config_obj = STFTConfig(
            frame_length=kwargs.get("frame_length", 256),
            frame_step=kwargs.get("frame_step", 128),
            fft_length=kwargs.get("fft_length"),
        )
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Stft",
            input_tensor.data,
            config=config_obj,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Complex64, input_tensor.device),
        )
    return _emit_shape_node(
        "Stft",
        [input_tensor],
        {"config": config_obj},
        input_tensor.shape,
        DType.Complex64,
    )


def mel_spectrogram(
    input_tensor: Tensor,
    sample_rate: int,
    num_mel_bins: int,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
) -> Tensor:
    """Computes a Mel spectrogram.

    Args:
        input_tensor (Tensor): Input spectrogram or audio.
        sample_rate (int): Sample rate.
        num_mel_bins (int): Number of Mel bins.
        lower_edge_hertz (float): Lower edge.
        upper_edge_hertz (float): Upper edge.

    Returns:
        Tensor: Mel spectrogram.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "MelSpectrogram",
            input_tensor.data,
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
        )
        return Tensor(  # pragma: no cover
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Float32, input_tensor.device),
        )
    return _emit_shape_node(
        "MelSpectrogram",
        [input_tensor],
        {
            "sample_rate": sample_rate,
            "num_mel_bins": num_mel_bins,
            "lower_edge_hertz": lower_edge_hertz,
            "upper_edge_hertz": upper_edge_hertz,
        },
        input_tensor.shape,
        DType.Float32,
    )


def istft(
    stft_tensor: Tensor, config_obj: STFTConfig | None = None, center: bool = True, **kwargs: object
) -> Tensor:
    """Computes the Inverse Short-Time Fourier Transform.

    Args:
        stft_tensor (Tensor): Input STFT tensor (complex-valued).
        config_obj (STFTConfig | None): STFT configuration.
        center (bool): Whether the signal was padded so that t=0 is centered.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: Reconstructed audio signal in time domain.
    """
    if config_obj is None:
        config_obj = STFTConfig(
            frame_length=kwargs.get("frame_length", 256),
            frame_step=kwargs.get("frame_step", 128),
            fft_length=kwargs.get("fft_length"),
            window_fn=kwargs.get("window", "hann"),
        )
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Istft",
            stft_tensor.data,
            config=config_obj,
            center=center,
        )
        # ISTFT returns real audio signals (float)
        return Tensor(  # pragma: no cover
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Float32, stft_tensor.device),
        )
    return _emit_shape_node(
        "Istft",
        [stft_tensor],
        {
            "config": config_obj,
            "center": center,
        },
        (),
        DType.Float32,
    )


def _build_mel_config(
    num_mel_bins: int,
    num_spectrogram_bins: int,
    sample_rate: int,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
) -> dict[str, object]:
    """Builds the configuration dict for mel_filterbank."""
    return {
        "num_mel_bins": num_mel_bins,
        "num_spectrogram_bins": num_spectrogram_bins,
        "sample_rate": sample_rate,
        "lower_edge_hertz": lower_edge_hertz,
        "upper_edge_hertz": upper_edge_hertz,
    }


def mel_filterbank(
    num_mel_bins: int,
    num_spectrogram_bins: int,
    sample_rate: int,
    lower_edge_hertz: float = 0.0,
    upper_edge_hertz: float = 0.0,
) -> Tensor:
    """Creates a Mel filterbank matrix.

    Args:
        num_mel_bins (int): Number of Mel bins.
        num_spectrogram_bins (int): Number of spectrogram bins (usually n_fft // 2 + 1).
        sample_rate (int): Sample rate in Hz.
        lower_edge_hertz (float): Lower bound on the frequencies.
        upper_edge_hertz (float): Upper bound on the frequencies. (default: sample_rate / 2)

    Returns:
        Tensor: A shape [num_spectrogram_bins, num_mel_bins] tensor representing the filterbank.
    """
    if upper_edge_hertz <= 0.0:
        upper_edge_hertz = float(sample_rate) / 2.0

    cfg = _build_mel_config(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz
    )

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("MelFilterbank", None, config=cfg)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Float32, Device(DeviceType.CPU)),
        )

    return _emit_shape_node(
        "MelFilterbank", [], {"config": cfg}, (num_spectrogram_bins, num_mel_bins), DType.Float32
    )


def mfcc(spectrogram: Tensor, config_obj: MFCCConfig | None = None, **kwargs: object) -> Tensor:
    """Computes Mel-Frequency Cepstral Coefficients (MFCCs).

    Args:
        spectrogram (Tensor): Input spectrogram [..., frames, num_spectrogram_bins].
        config_obj (MFCCConfig | None): MFCC configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: MFCCs [..., frames, num_mfccs].
    """
    if config_obj is None:  # pragma: no branch
        config_obj = {
            "sample_rate": kwargs.get("sample_rate", 16000),
            "num_mel_bins": kwargs.get("num_mel_bins", 40),
            "lower_edge_hertz": kwargs.get("lower_edge_hertz", 20.0),
            "upper_edge_hertz": kwargs.get("upper_edge_hertz", 4000.0),
            "num_mfccs": kwargs.get("num_mfccs", 13),
        }
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "Mfcc",
            spectrogram.data,
            config=config_obj,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Float32, spectrogram.device),
        )

    return _emit_shape_node(
        "Mfcc",
        [spectrogram],
        {
            "config": config_obj,
        },
        (),
        DType.Float32,
    )


def mfccs_from_log_mel_spectrograms(log_mel_spectrograms: Tensor, num_mfccs: int = 13) -> Tensor:
    """Computes MFCCs from log mel spectrograms.

    Args:
        log_mel_spectrograms (Tensor): Log mel spectrograms.
        num_mfccs (int): Number of MFCCs to compute.

    Returns:
    Tensor: MFCCs.
    """
    if config.eager_mode:
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(
            "MfccsFromLogMelSpectrograms", log_mel_spectrograms.data, num_mfccs=num_mfccs
        )  # pragma: no cover
        return Tensor(
            data, TensorConfig(data.shape, log_mel_spectrograms.dtype, log_mel_spectrograms.device)
        )  # pragma: no cover

    out_shape = list(log_mel_spectrograms.shape)
    if len(out_shape) > 0:
        out_shape[-1] = num_mfccs
    return _emit_shape_node(
        "MfccsFromLogMelSpectrograms",
        [log_mel_spectrograms],
        {"num_mfccs": num_mfccs},
        tuple(out_shape),
        log_mel_spectrograms.dtype,
    )


def hann_window(window_length: int, periodic: bool = True) -> Tensor:
    """Hann window.

    Args:
        window_length (int): the size of the window.
        periodic (bool): If True, returns a window to be used as periodic function.

    Returns:
    Tensor: the window.
    """
    if config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(
            "HannWindow", window_length=window_length, periodic=periodic
        )  # pragma: no cover

        return Tensor(
            data, TensorConfig(data.shape, DType.Float32, Device("cpu"))
        )  # pragma: no cover

    return _emit_shape_node(  # pragma: no cover
        "HannWindow",
        [],
        {"window_length": window_length, "periodic": periodic},
        (window_length,),
        DType.Float32,
    )


def hamming_window(
    window_length: int, periodic: bool = True, alpha: float = 0.54, beta: float = 0.46
) -> Tensor:
    """Hamming window.

    Args:
        window_length (int): the size of the window.
        periodic (bool): If True, returns a window to be used as periodic function.
        alpha (float): The coefficient alpha.
        beta (float): The coefficient beta.

    Returns:
    Tensor: the window.
    """
    if config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(
            "HammingWindow", window_length=window_length, periodic=periodic, alpha=alpha, beta=beta
        )  # pragma: no cover

        return Tensor(
            data, TensorConfig(data.shape, DType.Float32, Device("cpu"))
        )  # pragma: no cover

    return _emit_shape_node(  # pragma: no cover
        "HammingWindow",
        [],
        {"window_length": window_length, "periodic": periodic, "alpha": alpha, "beta": beta},
        (window_length,),
        DType.Float32,
    )


def kaiser_window(window_length: int, periodic: bool = True, beta: float = 12.0) -> Tensor:
    """Kaiser window.

    Args:
        window_length (int): the size of the window.
        periodic (bool): If True, returns a window to be used as periodic function.
        beta (float): Shape parameter.

    Returns:
    Tensor: the window.
    """
    if config.eager_mode:  # pragma: no cover
        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op(
            "KaiserWindow", window_length=window_length, periodic=periodic, beta=beta
        )  # pragma: no cover

        return Tensor(
            data, TensorConfig(data.shape, DType.Float32, Device("cpu"))
        )  # pragma: no cover

    return _emit_shape_node(  # pragma: no cover
        "KaiserWindow",
        [],
        {"window_length": window_length, "periodic": periodic, "beta": beta},
        (window_length,),
        DType.Float32,
    )
