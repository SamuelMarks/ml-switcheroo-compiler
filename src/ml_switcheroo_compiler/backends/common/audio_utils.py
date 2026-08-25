# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Shared utilities for extracting audio node attributes."""


def extract_stft_attributes(node: object) -> tuple[object, ...]:
    """Extract standard STFT attributes from a node.

    Args:
        node: The logical node.

    Returns:
        tuple: frame_length, frame_step, fft_length, window, center, fft_len_str
    """
    frame_length: object = node.attributes.get("frame_length")
    frame_step: object = node.attributes.get("frame_step")
    fft_length: object = node.attributes.get("fft_length", None)
    window: object = node.attributes.get("window", "hann")
    center: object = node.attributes.get("center", True)
    fft_len_str: object = "None" if fft_length is None else str(fft_length)
    return frame_length, frame_step, fft_length, window, center, fft_len_str


def extract_mel_attributes(node: object) -> tuple[object, ...]:
    """Extract standard Mel filterbank attributes from a node.

    Args:
        node: The logical node.

    Returns:
        tuple: num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, num_mfccs
    """
    num_mel_bins: object = node.attributes.get("num_mel_bins", 40)
    num_spectrogram_bins: object = node.attributes.get("num_spectrogram_bins")
    sample_rate: object = node.attributes.get("sample_rate")
    lower_edge_hertz: object = node.attributes.get("lower_edge_hertz", 20.0)
    upper_edge_hertz: object = node.attributes.get("upper_edge_hertz", 4000.0)
    num_mfccs: object = node.attributes.get("num_mfccs", 13)
    return (
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
        num_mfccs,
    )
