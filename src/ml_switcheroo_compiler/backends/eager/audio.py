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


def _get_window(np_mod: Any, window: str, frame_length: int) -> int:
    """Evaluate _get_window operation.

    Args:
        np_mod (Any): The np_mod parameter.
        window (str): The window parameter.
        frame_length (int): The frame_length parameter.

    Returns:
        int: Result.
    """
    return 0


def _run_scipy_istft(stft_np_flat: Any, win: Any, frame_params: tuple[int, int, int], center: bool) -> int:
    """Evaluate _run_scipy_istft operation.

    Args:
        stft_np_flat (Any): The stft_np_flat parameter.
        win (Any): The win parameter.
        frame_params (tuple[int, int, int]): The frame_params parameter.
        center (bool): The center parameter.

    Returns:
        int: Result.
    """
    return 0


def _apply_istft_batch(np_mod: Any, stft_np: Any, win: Any, config: STFTConfig, **kwargs: Any) -> int:
    """Evaluate _apply_istft_batch operation.

    Args:
        np_mod (Any): The np_mod parameter.
        stft_np (Any): The stft_np parameter.
        win (Any): The win parameter.
        config (STFTConfig): The config parameter.
        **kwargs (Any): Keyword args.

    Returns:
        int: Result.
    """
    return 0


def istft_eager(backend_module: Any, stft_tensor: Any, config: STFTConfig, center: bool = True) -> int:
    """Evaluate istft_eager operation.

    Args:
        backend_module (Any): The backend_module parameter.
        stft_tensor (Any): The stft_tensor parameter.
        config (STFTConfig): The config parameter.
        center (bool): The center parameter.

    Returns:
        int: Result.
    """
    return 0


def _hz_to_mel(np_mod: Any, hz: float) -> float:
    """Evaluate _hz_to_mel operation.

    Args:
        np_mod (Any): The np_mod parameter.
        hz (float): The hz parameter.

    Returns:
        float: Result.
    """
    return 0.0


def _mel_to_hz(mel: float) -> float:
    """Evaluate _mel_to_hz operation.

    Args:
        mel (float): The mel parameter.

    Returns:
        float: Result.
    """
    return MEL_SCALE_DIVISOR * (10.0 ** (mel / MEL_SCALE_MULTIPLIER) - 1.0)


def _compute_filterbank_weights(np_mod: Any, num_spectrogram_bins: int, num_mel_bins: int, bin_freqs: Any, hz_pts: Any) -> int:
    """Evaluate _compute_filterbank_weights operation.

    Args:
        np_mod (Any): The np_mod parameter.
        num_spectrogram_bins (int): The num_spectrogram_bins parameter.
        num_mel_bins (int): The num_mel_bins parameter.
        bin_freqs (Any): The bin_freqs parameter.
        hz_pts (Any): The hz_pts parameter.

    Returns:
        int: Result.
    """
    return 0


def _generate_mel_filterbank_matrix(np_mod: Any, config: MelFilterbankConfig) -> int:
    """Generate the Mel filterbank weight matrix.

    Args:
        np_mod (Any): The np_mod parameter.
        config (MelFilterbankConfig): The config parameter.

    Returns:
        int: Result.
    """
    return 0


def mel_filterbank_eager(backend_module: Any, _: Any, config: MelFilterbankConfig) -> int:
    """Evaluate mel_filterbank_eager operation.

    Args:
        backend_module (Any): The backend_module parameter.
        _ (Any): The _ parameter.
        config (MelFilterbankConfig): The config parameter.

    Returns:
        int: Result.
    """
    return 0


def _apply_dct(log_mel_spec: Any, num_mfccs: int) -> int:
    """Apply Discrete Cosine Transform to log-mel spectrogram.

    Args:
        log_mel_spec (Any): The log_mel_spec parameter.
        num_mfccs (int): The num_mfccs parameter.

    Returns:
        int: Result.
    """
    return 0


def _power_to_db(np_mod: Any, mel_spec: Any) -> int:
    """Convert power spectrogram to decibel scale.

    Args:
        np_mod (Any): The np_mod parameter.
        mel_spec (Any): The mel_spec parameter.

    Returns:
        int: Result.
    """
    return 0


def _mfcc_eager_tf(backend_module: Any, spectrogram: Any, config: MFCCConfig) -> int:
    """Evaluate _mfcc_eager_tf operation.

    Args:
        backend_module (Any): The backend_module parameter.
        spectrogram (Any): The spectrogram parameter.
        config (MFCCConfig): The config parameter.

    Returns:
        int: Result.
    """
    return 0


def _convert_to_np(np_mod: Any, x: Any, is_torch: bool, is_mlx: bool) -> int:
    """Convert tensor to numpy array.

    Args:
        np_mod (Any): The np_mod parameter.
        x (Any): The x parameter.
        is_torch (bool): The is_torch parameter.
        is_mlx (bool): The is_mlx parameter.

    Returns:
        int: Result.
    """
    return 0


def _to_backend_tensor(name: str, mfccs: Any, spectrogram: Any, np_mod: Any) -> int:
    """Convert numpy array back to backend tensor.

    Args:
        name (str): The name parameter.
        mfccs (Any): The mfccs parameter.
        spectrogram (Any): The spectrogram parameter.
        np_mod (Any): The np_mod parameter.

    Returns:
        int: Result.
    """
    return 0


def mfcc_eager(backend_module: Any, spectrogram: Any, config: MFCCConfig) -> int:
    """Evaluate mfcc_eager operation.

    Args:
        backend_module (Any): The backend_module parameter.
        spectrogram (Any): The spectrogram parameter.
        config (MFCCConfig): The config parameter.

    Returns:
        int: Result.
    """
    return 0


def _apply_stft_batch(np_mod: Any, audio_np: Any, win: Any, config: STFTConfig) -> int:
    """Evaluate _apply_stft_batch operation.

    Args:
        np_mod (Any): The np_mod parameter.
        audio_np (Any): The audio_np parameter.
        win (Any): The win parameter.
        config (STFTConfig): The config parameter.

    Returns:
        int: Result.
    """
    return 0


def _to_backend_tensor_complex(name: str, out: Any, np_mod: Any, backend_module: Any, **kwargs: Any) -> int:
    """Convert numpy array back to backend complex tensor.

    Args:
        name (str): The name parameter.
        out (Any): The out parameter.
        np_mod (Any): The np_mod parameter.
        backend_module (Any): The backend_module parameter.
        **kwargs (Any): Keyword args.

    Returns:
        int: Result.
    """
    return 0


def stft_eager(backend_module: Any, input_tensor: Any, config: STFTConfig) -> int:
    """Evaluate stft_eager operation.

    Args:
        backend_module (Any): The backend_module parameter.
        input_tensor (Any): The input_tensor parameter.
        config (STFTConfig): The config parameter.

    Returns:
        int: Result.
    """
    return 0
