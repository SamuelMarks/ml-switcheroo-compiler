# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Audio utilities."""

import typing
from typing import Any

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


def _get_window(np_mod: Any, window: str, frame_length: int) -> Any:
    """Evaluate _get_window operation.

    Args:
        np_mod (object): The np_mod parameter.
        window (str): The window parameter.
        frame_length (int): The frame_length parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _run_scipy_istft(stft_np_flat: Any, win: Any, frame_params: tuple[int, int, int], center: bool) -> Any:
    """Evaluate _run_scipy_istft operation.

    Args:
        stft_np_flat (object): The stft_np_flat parameter.
        win (object): The win parameter.
        frame_params (object): The frame_params parameter.
        center (bool): The center parameter.

    Returns:
        list: Result.
    """
    return 0


def _apply_istft_batch(np_mod: Any, stft_np: Any, win: Any, config: STFTConfig, **kwargs: Any) -> Any:
    """Evaluate _apply_istft_batch operation.

    Args:
        np_mod (object): The np_mod parameter.
        stft_np (object): The stft_np parameter.
        win (object): The win parameter.
        config (STFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def istft_eager(backend_module: Any, stft_tensor: Any, config: STFTConfig, center: bool = True) -> Any:
    """Evaluate istft_eager operation.

    Args:
        backend_module (object): The backend_module parameter.
        stft_tensor (object): The stft_tensor parameter.
        config (STFTConfig): The config parameter.
        center (bool): The center parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _hz_to_mel(np_mod: Any, hz: float) -> float:
    """Evaluate _hz_to_mel operation.

    Args:
        np_mod (object): The np_mod parameter.
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
    return MEL_SCALE_DIVISOR * (10.0 ** (mel / MEL_SCALE_MULTIPLIER) - 1.0)  # type: ignore


def _compute_filterbank_weights(np_mod: Any, num_spectrogram_bins: int, num_mel_bins: int, bin_freqs: Any, hz_pts: Any) -> Any:
    """Evaluate _compute_filterbank_weights operation.

    Args:
        np_mod (object): The np_mod parameter.
        num_spectrogram_bins (int): The num_spectrogram_bins parameter.
        num_mel_bins (int): The num_mel_bins parameter.
        bin_freqs (object): The bin_freqs parameter.
        hz_pts (object): The hz_pts parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _generate_mel_filterbank_matrix(np_mod: Any, config: MelFilterbankConfig) -> Any:
    """Generate the Mel filterbank weight matrix.

    Args:
        np_mod (object): The np_mod parameter.
        config (MelFilterbankConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def mel_filterbank_eager(backend_module: Any, _: Any, config: MelFilterbankConfig) -> Any:
    """Evaluate mel_filterbank_eager operation.

    Args:
        backend_module (object): The backend_module parameter.
        _ (object): The _ parameter.
        config (MelFilterbankConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _apply_dct(log_mel_spec: Any, num_mfccs: int) -> Any:
    """Apply Discrete Cosine Transform to log-mel spectrogram.

    Args:
        log_mel_spec (object): The log_mel_spec parameter.
        num_mfccs (int): The num_mfccs parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _power_to_db(np_mod: Any, mel_spec: Any) -> Any:
    """Convert power spectrogram to decibel scale.

    Args:
        np_mod (object): The np_mod parameter.
        mel_spec (object): The mel_spec parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _mfcc_eager_tf(backend_module: Any, spectrogram: Any, config: MFCCConfig) -> Any:
    """Evaluate _mfcc_eager_tf operation.

    Args:
        backend_module (object): The backend_module parameter.
        spectrogram (object): The spectrogram parameter.
        config (MFCCConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _convert_to_np(np_mod: Any, x: Any, is_torch: bool, is_mlx: bool) -> Any:
    """Convert tensor to numpy array.

    Args:
        np_mod (object): The np_mod parameter.
        x (object): The x parameter.
        is_torch (bool): The is_torch parameter.
        is_mlx (bool): The is_mlx parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _to_backend_tensor(name: str, mfccs: Any, spectrogram: Any, np_mod: Any) -> Any:
    """Convert numpy array back to backend tensor.

    Args:
        name (str): The name parameter.
        mfccs (object): The mfccs parameter.
        spectrogram (object): The spectrogram parameter.
        np_mod (object): The np_mod parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def mfcc_eager(backend_module: Any, spectrogram: Any, config: MFCCConfig) -> Any:
    """Evaluate mfcc_eager operation.

    Args:
        backend_module (object): The backend_module parameter.
        spectrogram (object): The spectrogram parameter.
        config (MFCCConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _apply_stft_batch(np_mod: Any, audio_np: Any, win: Any, config: STFTConfig) -> Any:
    """Evaluate _apply_stft_batch operation.

    Args:
        np_mod (object): The np_mod parameter.
        audio_np (object): The audio_np parameter.
        win (object): The win parameter.
        config (STFTConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def _to_backend_tensor_complex(name: str, out: Any, np_mod: Any, backend_module: Any, **kwargs: Any) -> Any:
    """Convert numpy array back to backend complex tensor.

    Args:
        name (str): The name parameter.
        out (object): The out parameter.
        np_mod (object): The np_mod parameter.
        backend_module (object): The backend_module parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0


def stft_eager(backend_module: Any, input_tensor: Any, config: STFTConfig) -> Any:
    """Evaluate stft_eager operation.

    Args:
        backend_module (object): The backend_module parameter.
        input_tensor (object): The input_tensor parameter.
        config (STFTConfig): The config parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return 0
