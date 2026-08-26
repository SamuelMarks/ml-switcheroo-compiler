# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Shared utilities for extracting audio node attributes."""

from typing import Any

from ml_switcheroo_compiler.ir.core import IRNode


def extract_stft_attributes(node: IRNode) -> tuple[Any, Any, Any, str, bool, str]:
    """Extract standard STFT attributes from a node.

    Args:
        node: The logical node.

    Returns:
        tuple: frame_length, frame_step, fft_length, window, center, fft_len_str
    """
    frame_length = node.attributes.get("frame_length")
    frame_step = node.attributes.get("frame_step")
    fft_length = node.attributes.get("fft_length", None)
    window = str(node.attributes.get("window", "hann"))
    center = bool(node.attributes.get("center", True))
    fft_len_str = "None" if fft_length is None else str(fft_length)
    return frame_length, frame_step, fft_length, window, center, fft_len_str


def extract_mel_attributes(node: IRNode) -> tuple[int, Any, Any, float, float, int]:
    """Extract standard Mel filterbank attributes from a node.

    Args:
        node: The logical node.

    Returns:
        tuple: num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, num_mfccs
    """
    num_mel_bins = int(node.attributes.get("num_mel_bins", 40))
    num_spectrogram_bins = node.attributes.get("num_spectrogram_bins")
    sample_rate = node.attributes.get("sample_rate")
    lower_edge_hertz = float(node.attributes.get("lower_edge_hertz", 20.0))
    upper_edge_hertz = float(node.attributes.get("upper_edge_hertz", 4000.0))
    num_mfccs = int(node.attributes.get("num_mfccs", 13))
    return (
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
        num_mfccs,
    )
