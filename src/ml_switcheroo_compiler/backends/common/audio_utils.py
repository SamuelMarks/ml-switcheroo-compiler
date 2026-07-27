# ruff: noqa: E501
"""Shared utilities for extracting audio node attributes."""


def extract_stft_attributes(node: object) -> tuple:
    """Extract standard STFT attributes from a node.

    Args:
        node: The logical node.

    Returns:
        tuple: frame_length, frame_step, fft_length, window, center, fft_len_str
    """
    frame_length = node.attributes.get("frame_length")
    frame_step = node.attributes.get("frame_step")
    fft_length = node.attributes.get("fft_length", None)
    window = node.attributes.get("window", "hann")
    center = node.attributes.get("center", True)
    fft_len_str = "None" if fft_length is None else str(fft_length)
    return frame_length, frame_step, fft_length, window, center, fft_len_str


def extract_mel_attributes(node: object) -> tuple:
    """Extract standard Mel filterbank attributes from a node.

    Args:
        node: The logical node.

    Returns:
        tuple: num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, num_mfccs
    """
    num_mel_bins = node.attributes.get("num_mel_bins", 40)
    num_spectrogram_bins = node.attributes.get("num_spectrogram_bins")
    sample_rate = node.attributes.get("sample_rate")
    lower_edge_hertz = node.attributes.get("lower_edge_hertz", 20.0)
    upper_edge_hertz = node.attributes.get("upper_edge_hertz", 4000.0)
    num_mfccs = node.attributes.get("num_mfccs", 13)
    return (
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
        num_mfccs,
    )
