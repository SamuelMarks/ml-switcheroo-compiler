"""Module docstring."""


# ruff: noqa: E402, D100, D101
def _extract_audio_stft_attributes(node: object) -> tuple:
    """Extract STFT attributes."""
    frame_length = node.attributes.get("frame_length", 2048)
    frame_step = node.attributes.get("frame_step", 512)
    fft_length = node.attributes.get("fft_length", None)
    window_fn = node.attributes.get("window_fn", "hann")
    pad_end = node.attributes.get("pad_end", False)
    return frame_length, frame_step, fft_length, window_fn, pad_end


def _extract_resize_attributes(node: object) -> tuple:
    """Extract resize attributes."""
    size = node.attributes.get("size")
    interpolation = node.attributes.get("interpolation", "bilinear")
    align_corners = node.attributes.get("align_corners", False)
    antialias = node.attributes.get("antialias", False)
    data_format = node.attributes.get("data_format", None)
    return size, interpolation, align_corners, antialias, data_format


def _extract_vision_transform_attributes(node: object) -> tuple:
    """Extract vision transform attributes."""
    interpolation = node.attributes.get("interpolation", "bilinear")
    fill_value = node.attributes.get("fill_value", 0.0)
    data_format = node.attributes.get("data_format", None)
    return interpolation, fill_value, data_format


def _extract_filter_attributes(node: object) -> tuple:
    """Extract filter attributes."""
    kernel_size = node.attributes.get("kernel_size")
    sigma = node.attributes.get("sigma", None)
    padding = node.attributes.get("padding", "same")
    data_format = node.attributes.get("data_format", None)
    return kernel_size, sigma, padding, data_format


def _extract_extract_boxes_attributes(node: object) -> tuple:
    """Extract bounding box extraction attributes."""
    crop_size = node.attributes.get("crop_size")
    interpolation = node.attributes.get("interpolation", "bilinear")
    extrapolation_value = node.attributes.get("extrapolation_value", 0.0)
    data_format = node.attributes.get("data_format", None)
    return crop_size, interpolation, extrapolation_value, data_format


def _extract_stft_attributes(node: object) -> tuple:
    """Extract generic STFT attributes."""
    n_fft = node.attributes.get("n_fft", 2048)
    hop_length = node.attributes.get("hop_length", None)
    win_length = node.attributes.get("win_length", None)
    window = node.attributes.get("window", None)
    center = node.attributes.get("center", True)
    pad_mode = node.attributes.get("pad_mode", "reflect")
    normalized = node.attributes.get("normalized", False)
    onesided = node.attributes.get("onesided", True)
    return n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided
