# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for generator_utils.py."""

from typing import Any


def _extract_audio_stft_attributes(node: Any) -> tuple[int, int, Any, str, bool]:
    """Extract STFT attributes.

    Args:
        node (Any): The node parameter.

    Returns:
        tuple[int, int, Any, str, bool]: Result.
    """
    frame_length: int = node.attributes.get("frame_length", 2048)
    frame_step: int = node.attributes.get("frame_step", 512)
    fft_length: Any = node.attributes.get("fft_length", None)
    window_fn: str = node.attributes.get("window_fn", "hann")
    pad_end: bool = node.attributes.get("pad_end", False)
    return frame_length, frame_step, fft_length, window_fn, pad_end


def _extract_resize_attributes(node: Any) -> tuple[Any, str, bool, bool, Any]:
    """Extract resize attributes.

    Args:
        node (Any): The node parameter.

    Returns:
        tuple[Any, str, bool, bool, Any]: Result.
    """
    size: Any = node.attributes.get("size")
    interpolation: str = node.attributes.get("interpolation", "bilinear")
    align_corners: bool = node.attributes.get("align_corners", False)
    antialias: bool = node.attributes.get("antialias", False)
    data_format: Any = node.attributes.get("data_format", None)
    return size, interpolation, align_corners, antialias, data_format


def _extract_vision_transform_attributes(node: Any) -> tuple[str, float, Any]:
    """Extract vision transform attributes.

    Args:
        node (Any): The node parameter.

    Returns:
        tuple[str, float, Any]: Result.
    """
    interpolation: str = node.attributes.get("interpolation", "bilinear")
    fill_value: float = node.attributes.get("fill_value", 0.0)
    data_format: Any = node.attributes.get("data_format", None)
    return interpolation, fill_value, data_format


def _extract_filter_attributes(node: Any) -> tuple[Any, Any, str, Any]:
    """Extract filter attributes.

    Args:
        node (Any): The node parameter.

    Returns:
        tuple[Any, Any, str, Any]: Result.
    """
    kernel_size: Any = node.attributes.get("kernel_size")
    sigma: Any = node.attributes.get("sigma", None)
    padding: str = node.attributes.get("padding", "same")
    data_format: Any = node.attributes.get("data_format", None)
    return kernel_size, sigma, padding, data_format


def _extract_extract_boxes_attributes(node: Any) -> tuple[Any, str, float, Any]:
    """Extract bounding box extraction attributes.

    Args:
        node (Any): The node parameter.

    Returns:
        tuple[Any, str, float, Any]: Result.
    """
    crop_size: Any = node.attributes.get("crop_size")
    interpolation: str = node.attributes.get("interpolation", "bilinear")
    extrapolation_value: float = node.attributes.get("extrapolation_value", 0.0)
    data_format: Any = node.attributes.get("data_format", None)
    return crop_size, interpolation, extrapolation_value, data_format


def _extract_stft_attributes(node: Any) -> tuple[int, Any, Any, Any, bool, str, bool, bool]:
    """Extract generic STFT attributes.

    Args:
        node (Any): The node parameter.

    Returns:
        tuple[int, Any, Any, Any, bool, str, bool, bool]: Result.
    """
    n_fft: int = node.attributes.get("n_fft", 2048)
    hop_length: Any = node.attributes.get("hop_length", None)
    win_length: Any = node.attributes.get("win_length", None)
    window: Any = node.attributes.get("window", None)
    center: bool = node.attributes.get("center", True)
    pad_mode: str = node.attributes.get("pad_mode", "reflect")
    normalized: bool = node.attributes.get("normalized", False)
    onesided: bool = node.attributes.get("onesided", True)
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
