# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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


def _get_window(np_mod: typing.Any, window: str, frame_length: int) -> typing.Any:
    """Evaluate _get_window operation.

    Args:
        np_mod (typing.Any): The np_mod parameter.
        window (str): The window parameter.
        frame_length (int): The frame_length parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _run_scipy_istft(stft_np_flat: typing.Any, win: typing.Any, frame_params: tuple[int, int, int], center: bool) -> typing.Any:
    """Evaluate _run_scipy_istft operation.

    Args:
        stft_np_flat (typing.Any): The stft_np_flat parameter.
        win (typing.Any): The win parameter.
        frame_params (typing.Any): The frame_params parameter.
        center (bool): The center parameter.

    Returns:
        list: Result.
    """
    return 0


def _apply_istft_batch(np_mod: typing.Any, stft_np: typing.Any, win: typing.Any, config: STFTConfig, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _apply_istft_batch operation.

    Args:
        np_mod (typing.Any): The np_mod parameter.
        stft_np (typing.Any): The stft_np parameter.
        win (typing.Any): The win parameter.
        config (STFTConfig): The config parameter.
        **kwargs (typing.Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def istft_eager(backend_module: typing.Any, stft_tensor: typing.Any, config: STFTConfig, center: bool = True) -> typing.Any:
    """Evaluate istft_eager operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        stft_tensor (typing.Any): The stft_tensor parameter.
        config (STFTConfig): The config parameter.
        center (bool): The center parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _hz_to_mel(np_mod: typing.Any, hz: float) -> float:
    """Evaluate _hz_to_mel operation.

    Args:
        np_mod (typing.Any): The np_mod parameter.
        hz (float): The hz parameter.

    Returns:
        float: Result.
    """
    return 0


def _mel_to_hz(mel: float) -> float:
    """Evaluate _mel_to_hz operation.

    Args:
        mel (float): The mel parameter.

    Returns:
        float: Result.
    """
    return MEL_SCALE_DIVISOR * (10.0 ** (mel / MEL_SCALE_MULTIPLIER) - 1.0)


def _compute_filterbank_weights(np_mod: typing.Any, num_spectrogram_bins: int, num_mel_bins: int, bin_freqs: typing.Any, hz_pts: typing.Any) -> typing.Any:
    """Evaluate _compute_filterbank_weights operation.

    Args:
        np_mod (typing.Any): The np_mod parameter.
        num_spectrogram_bins (int): The num_spectrogram_bins parameter.
        num_mel_bins (int): The num_mel_bins parameter.
        bin_freqs (typing.Any): The bin_freqs parameter.
        hz_pts (typing.Any): The hz_pts parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _generate_mel_filterbank_matrix(np_mod: typing.Any, config: MelFilterbankConfig) -> typing.Any:
    """Generate the Mel filterbank weight matrix.

    Args:
        np_mod (typing.Any): The np_mod parameter.
        config (MelFilterbankConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def mel_filterbank_eager(backend_module: typing.Any, _: typing.Any, config: MelFilterbankConfig) -> typing.Any:
    """Evaluate mel_filterbank_eager operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        _ (typing.Any): The _ parameter.
        config (MelFilterbankConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _apply_dct(log_mel_spec: typing.Any, num_mfccs: int) -> typing.Any:
    """Apply Discrete Cosine Transform to log-mel spectrogram.

    Args:
        log_mel_spec (typing.Any): The log_mel_spec parameter.
        num_mfccs (int): The num_mfccs parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _power_to_db(np_mod: typing.Any, mel_spec: typing.Any) -> typing.Any:
    """Convert power spectrogram to decibel scale.

    Args:
        np_mod (typing.Any): The np_mod parameter.
        mel_spec (typing.Any): The mel_spec parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _mfcc_eager_tf(backend_module: typing.Any, spectrogram: typing.Any, config: MFCCConfig) -> typing.Any:
    """Evaluate _mfcc_eager_tf operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        spectrogram (typing.Any): The spectrogram parameter.
        config (MFCCConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _convert_to_np(np_mod: typing.Any, x: typing.Any, is_torch: bool, is_mlx: bool) -> typing.Any:
    """Convert tensor to numpy array.

    Args:
        np_mod (typing.Any): The np_mod parameter.
        x (typing.Any): The x parameter.
        is_torch (bool): The is_torch parameter.
        is_mlx (bool): The is_mlx parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _to_backend_tensor(name: str, mfccs: typing.Any, spectrogram: typing.Any, np_mod: typing.Any) -> typing.Any:
    """Convert numpy array back to backend tensor.

    Args:
        name (str): The name parameter.
        mfccs (typing.Any): The mfccs parameter.
        spectrogram (typing.Any): The spectrogram parameter.
        np_mod (typing.Any): The np_mod parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def mfcc_eager(backend_module: typing.Any, spectrogram: typing.Any, config: MFCCConfig) -> typing.Any:
    """Evaluate mfcc_eager operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        spectrogram (typing.Any): The spectrogram parameter.
        config (MFCCConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _apply_stft_batch(np_mod: typing.Any, audio_np: typing.Any, win: typing.Any, config: STFTConfig) -> typing.Any:
    """Evaluate _apply_stft_batch operation.

    Args:
        np_mod (typing.Any): The np_mod parameter.
        audio_np (typing.Any): The audio_np parameter.
        win (typing.Any): The win parameter.
        config (STFTConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _to_backend_tensor_complex(name: str, out: typing.Any, np_mod: typing.Any, backend_module: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Convert numpy array back to backend complex tensor.

    Args:
        name (str): The name parameter.
        out (typing.Any): The out parameter.
        np_mod (typing.Any): The np_mod parameter.
        backend_module (typing.Any): The backend_module parameter.
        **kwargs (typing.Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def stft_eager(backend_module: typing.Any, input_tensor: typing.Any, config: STFTConfig) -> typing.Any:
    """Evaluate stft_eager operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        input_tensor (typing.Any): The input_tensor parameter.
        config (STFTConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0
