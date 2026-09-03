# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for generator_utils.py."""

from typing import Optional


def _extract_audio_stft_attributes(node: object) -> tuple[int, int, Optional[object], str, bool]:
    """Extract STFT attributes.

    Args:
        node: The node parameter.

    Returns:
        tuple[int, int, Optional[object], str, bool]: Result.
    """
    frame_length: int = getattr(node, "attributes", {}).get("frame_length", 2048)
    frame_step: int = getattr(node, "attributes", {}).get("frame_step", 512)
    fft_length = getattr(node, "attributes", {}).get("fft_length", None)
    window_fn: str = getattr(node, "attributes", {}).get("window_fn", "hann")
    pad_end: bool = getattr(node, "attributes", {}).get("pad_end", False)
    return frame_length, frame_step, fft_length, window_fn, pad_end


def _extract_resize_attributes(node: object) -> tuple[object, str, bool, bool, Optional[object]]:
    """Extract resize attributes.

    Args:
        node: The node parameter.

    Returns:
        tuple[object, str, bool, bool, Optional[object]]: Result.
    """
    size = getattr(node, "attributes", {}).get("size")
    interpolation: str = getattr(node, "attributes", {}).get("interpolation", "bilinear")
    align_corners: bool = getattr(node, "attributes", {}).get("align_corners", False)
    antialias: bool = getattr(node, "attributes", {}).get("antialias", False)
    data_format = getattr(node, "attributes", {}).get("data_format", None)
    return size, interpolation, align_corners, antialias, data_format


def _extract_vision_transform_attributes(node: object) -> tuple[str, float, Optional[object]]:
    """Extract vision transform attributes.

    Args:
        node: The node parameter.

    Returns:
        tuple[str, float, Optional[object]]: Result.
    """
    interpolation: str = getattr(node, "attributes", {}).get("interpolation", "bilinear")
    fill_value: float = getattr(node, "attributes", {}).get("fill_value", 0.0)
    data_format = getattr(node, "attributes", {}).get("data_format", None)
    return interpolation, fill_value, data_format


def _extract_filter_attributes(node: object) -> tuple[object, Optional[object], str, Optional[object]]:
    """Extract filter attributes.

    Args:
        node: The node parameter.

    Returns:
        tuple[object, Optional[object], str, Optional[object]]: Result.
    """
    kernel_size = getattr(node, "attributes", {}).get("kernel_size")
    sigma = getattr(node, "attributes", {}).get("sigma", None)
    padding: str = getattr(node, "attributes", {}).get("padding", "same")
    data_format = getattr(node, "attributes", {}).get("data_format", None)
    return kernel_size, sigma, padding, data_format


def _extract_extract_boxes_attributes(node: object) -> tuple[object, str, float, Optional[object]]:
    """Extract bounding box extraction attributes.

    Args:
        node: The node parameter.

    Returns:
        tuple[object, str, float, Optional[object]]: Result.
    """
    crop_size = getattr(node, "attributes", {}).get("crop_size")
    interpolation: str = getattr(node, "attributes", {}).get("interpolation", "bilinear")
    extrapolation_value: float = getattr(node, "attributes", {}).get("extrapolation_value", 0.0)
    data_format = getattr(node, "attributes", {}).get("data_format", None)
    return crop_size, interpolation, extrapolation_value, data_format


def _extract_stft_attributes(node: object) -> tuple[int, Optional[object], Optional[object], Optional[object], bool, str, bool, bool]:
    """Extract generic STFT attributes.

    Args:
        node: The node parameter.

    Returns:
        tuple[int, Optional[object], Optional[object], Optional[object], bool, str, bool, bool]: Result.
    """
    n_fft: int = getattr(node, "attributes", {}).get("n_fft", 2048)
    hop_length = getattr(node, "attributes", {}).get("hop_length", None)
    win_length = getattr(node, "attributes", {}).get("win_length", None)
    window = getattr(node, "attributes", {}).get("window", None)
    center: bool = getattr(node, "attributes", {}).get("center", True)
    pad_mode: str = getattr(node, "attributes", {}).get("pad_mode", "reflect")
    normalized: bool = getattr(node, "attributes", {}).get("normalized", False)
    onesided: bool = getattr(node, "attributes", {}).get("onesided", True)
    return (
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
    )
