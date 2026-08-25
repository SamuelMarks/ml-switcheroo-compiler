# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for generator_utils.py."""


def _extract_audio_stft_attributes(node: object) -> tuple[object, ...]:
    """Extract STFT attributes.

    Args:
        node (object): The node parameter.

    Returns:
        tuple: Result.
    """
    frame_length: object = node.attributes.get("frame_length", 2048)
    frame_step: object = node.attributes.get("frame_step", 512)
    fft_length: object = node.attributes.get("fft_length", None)
    window_fn: object = node.attributes.get("window_fn", "hann")
    pad_end: object = node.attributes.get("pad_end", False)
    return frame_length, frame_step, fft_length, window_fn, pad_end


def _extract_resize_attributes(node: object) -> tuple[object, ...]:
    """Extract resize attributes.

    Args:
        node (object): The node parameter.

    Returns:
        tuple: Result.
    """
    size: object = node.attributes.get("size")
    interpolation: object = node.attributes.get("interpolation", "bilinear")
    align_corners: object = node.attributes.get("align_corners", False)
    antialias: object = node.attributes.get("antialias", False)
    data_format: object = node.attributes.get("data_format", None)
    return size, interpolation, align_corners, antialias, data_format


def _extract_vision_transform_attributes(node: object) -> tuple[object, ...]:
    """Extract vision transform attributes.

    Args:
        node (object): The node parameter.

    Returns:
        tuple: Result.
    """
    interpolation: object = node.attributes.get("interpolation", "bilinear")
    fill_value: object = node.attributes.get("fill_value", 0.0)
    data_format: object = node.attributes.get("data_format", None)
    return interpolation, fill_value, data_format


def _extract_filter_attributes(node: object) -> tuple[object, ...]:
    """Extract filter attributes.

    Args:
        node (object): The node parameter.

    Returns:
        tuple: Result.
    """
    kernel_size: object = node.attributes.get("kernel_size")
    sigma: object = node.attributes.get("sigma", None)
    padding: object = node.attributes.get("padding", "same")
    data_format: object = node.attributes.get("data_format", None)
    return kernel_size, sigma, padding, data_format


def _extract_extract_boxes_attributes(node: object) -> tuple[object, ...]:
    """Extract bounding box extraction attributes.

    Args:
        node (object): The node parameter.

    Returns:
        tuple: Result.
    """
    crop_size: object = node.attributes.get("crop_size")
    interpolation: object = node.attributes.get("interpolation", "bilinear")
    extrapolation_value: object = node.attributes.get("extrapolation_value", 0.0)
    data_format: object = node.attributes.get("data_format", None)
    return crop_size, interpolation, extrapolation_value, data_format


def _extract_stft_attributes(node: object) -> tuple[object, ...]:
    """Extract generic STFT attributes.

    Args:
        node (object): The node parameter.

    Returns:
        tuple: Result.
    """
    n_fft: object = node.attributes.get("n_fft", 2048)
    hop_length: object = node.attributes.get("hop_length", None)
    win_length: object = node.attributes.get("win_length", None)
    window: object = node.attributes.get("window", None)
    center: object = node.attributes.get("center", True)
    pad_mode: object = node.attributes.get("pad_mode", "reflect")
    normalized: object = node.attributes.get("normalized", False)
    onesided: object = node.attributes.get("onesided", True)
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
