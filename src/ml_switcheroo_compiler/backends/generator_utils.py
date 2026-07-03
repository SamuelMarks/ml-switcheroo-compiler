"""Module docstring."""


def _extract_audio_stft_attributes(node: object) -> tuple:
    """Extract STFT attributes."""
    frame_length = node.attributes.get("frame_length", 2048)  # pragma: no cover
    frame_step = node.attributes.get("frame_step", 512)  # pragma: no cover
    fft_length = node.attributes.get("fft_length", None)  # pragma: no cover
    window_fn = node.attributes.get("window_fn", "hann")  # pragma: no cover
    pad_end = node.attributes.get("pad_end", False)  # pragma: no cover
    return frame_length, frame_step, fft_length, window_fn, pad_end  # pragma: no cover


def _extract_resize_attributes(node: object) -> tuple:
    """Extract resize attributes."""
    size = node.attributes.get("size")  # pragma: no cover
    interpolation = node.attributes.get("interpolation", "bilinear")  # pragma: no cover
    align_corners = node.attributes.get("align_corners", False)  # pragma: no cover
    antialias = node.attributes.get("antialias", False)  # pragma: no cover
    data_format = node.attributes.get("data_format", None)  # pragma: no cover
    return size, interpolation, align_corners, antialias, data_format  # pragma: no cover


def _extract_vision_transform_attributes(node: object) -> tuple:
    """Extract vision transform attributes."""
    interpolation = node.attributes.get("interpolation", "bilinear")  # pragma: no cover
    fill_value = node.attributes.get("fill_value", 0.0)  # pragma: no cover
    data_format = node.attributes.get("data_format", None)  # pragma: no cover
    return interpolation, fill_value, data_format  # pragma: no cover


def _extract_filter_attributes(node: object) -> tuple:
    """Extract filter attributes."""
    kernel_size = node.attributes.get("kernel_size")  # pragma: no cover
    sigma = node.attributes.get("sigma", None)  # pragma: no cover
    padding = node.attributes.get("padding", "same")  # pragma: no cover
    data_format = node.attributes.get("data_format", None)  # pragma: no cover
    return kernel_size, sigma, padding, data_format  # pragma: no cover


def _extract_extract_boxes_attributes(node: object) -> tuple:
    """Extract bounding box extraction attributes."""
    crop_size = node.attributes.get("crop_size")  # pragma: no cover
    interpolation = node.attributes.get("interpolation", "bilinear")  # pragma: no cover
    extrapolation_value = node.attributes.get("extrapolation_value", 0.0)  # pragma: no cover
    data_format = node.attributes.get("data_format", None)  # pragma: no cover
    return crop_size, interpolation, extrapolation_value, data_format  # pragma: no cover


def _extract_stft_attributes(node: object) -> tuple:
    """Extract generic STFT attributes."""
    n_fft = node.attributes.get("n_fft", 2048)  # pragma: no cover
    hop_length = node.attributes.get("hop_length", None)  # pragma: no cover
    win_length = node.attributes.get("win_length", None)  # pragma: no cover
    window = node.attributes.get("window", None)  # pragma: no cover
    center = node.attributes.get("center", True)  # pragma: no cover
    pad_mode = node.attributes.get("pad_mode", "reflect")  # pragma: no cover
    normalized = node.attributes.get("normalized", False)  # pragma: no cover
    onesided = node.attributes.get("onesided", True)  # pragma: no cover
    return (
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
    )  # pragma: no cover
