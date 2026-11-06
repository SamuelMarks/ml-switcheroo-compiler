"""Implementation of generator utils operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.generator_utils`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_float, ensure_int
from ml_switcheroo_compiler.ir.core import IRNode


@typing.runtime_checkable
class _NodeAttributesProtocol(typing.Protocol):
    """Protocol for nodes exposing `.attributes`."""

    attributes: dict[str, EagerValue]


def _node_attrs(node: IRNode | EagerValue) -> dict[str, EagerValue]:
    """Extract `.attributes` dictionary from `node`.

    Args:
        node (IRNode | EagerValue): Target IR node.

    Returns:
        dict[str, EagerValue]: Node attributes mapping.
    """
    return dict(node.attributes) if isinstance(node, _NodeAttributesProtocol) and node.attributes is not None else {}


def _extract_audio_stft_attributes(node: IRNode) -> tuple[int, int, int | None, str, bool]:
    """Extract STFT attributes from an IR node.

    Args:
        node (IRNode): The IR node containing STFT attributes.

    Returns:
        tuple[int, int, int | None, str, bool]: Frame length, frame step, fft length, window function name, and pad end flag.
    """
    attrs = _node_attrs(node)
    raw_frame_length = attrs.get("frame_length", 2048)
    raw_frame_step = attrs.get("frame_step", 512)
    raw_fft_length = attrs.get("fft_length", None)
    raw_window_fn = attrs.get("window_fn", "hann")
    raw_pad_end = attrs.get("pad_end", False)

    frame_length: int = int(ensure_int(raw_frame_length)) if raw_frame_length is not None else 2048
    frame_step: int = int(ensure_int(raw_frame_step)) if raw_frame_step is not None else 512
    fft_length: int | None = int(ensure_int(raw_fft_length)) if raw_fft_length is not None else None
    window_fn: str = str(raw_window_fn)
    pad_end: bool = bool(raw_pad_end)
    return frame_length, frame_step, fft_length, window_fn, pad_end


def _extract_resize_attributes(
    node: IRNode,
) -> tuple[int | tuple[int, int] | list[int] | None, str, bool, bool, str | None]:
    """Extract resize attributes from an IR node.

    Args:
        node (IRNode): The IR node containing resize attributes.

    Returns:
        tuple[int | tuple[int, int] | list[int] | None, str, bool, bool, str | None]:
            Target size, interpolation method, align corners flag, antialias flag, and data format.
    """
    attrs = _node_attrs(node)
    raw_size = attrs.get("size")
    size: int | tuple[int, int] | list[int] | None = raw_size if isinstance(raw_size, int) else ([ensure_int(d) for d in raw_size] if isinstance(raw_size, list) else ((ensure_int(raw_size[0]), ensure_int(raw_size[1])) if isinstance(raw_size, tuple) and len(raw_size) >= 2 else None))
    interpolation: str = str(attrs.get("interpolation", "bilinear"))
    align_corners: bool = bool(attrs.get("align_corners", False))
    antialias: bool = bool(attrs.get("antialias", False))
    raw_data_format = attrs.get("data_format", None)
    data_format: str | None = str(raw_data_format) if raw_data_format is not None else None
    return size, interpolation, align_corners, antialias, data_format


def _extract_vision_transform_attributes(node: IRNode) -> tuple[str, float, str | None]:
    """Extract vision transform attributes from an IR node.

    Args:
        node (IRNode): The IR node containing transform attributes.

    Returns:
        tuple[str, float, str | None]: Interpolation method, fill value, and data format.
    """
    attrs = _node_attrs(node)
    interpolation: str = str(attrs.get("interpolation", "bilinear"))
    fill_value: float = float(ensure_float(attrs.get("fill_value", 0.0)))
    raw_data_format = attrs.get("data_format", None)
    data_format: str | None = str(raw_data_format) if raw_data_format is not None else None
    return interpolation, fill_value, data_format


def _extract_filter_attributes(
    node: IRNode,
) -> tuple[int | tuple[int, int] | list[int] | None, float | tuple[float, float] | None, str, str | None]:
    """Extract filter attributes from an IR node.

    Args:
        node (IRNode): The IR node containing filter attributes.

    Returns:
        tuple[int | tuple[int, int] | list[int] | None, float | tuple[float, float] | None, str, str | None]:
            Kernel size, sigma, padding, and data format.
    """
    attrs = _node_attrs(node)
    raw_ks = attrs.get("kernel_size")
    kernel_size: int | tuple[int, int] | list[int] | None = raw_ks if isinstance(raw_ks, int) else ([ensure_int(d) for d in raw_ks] if isinstance(raw_ks, list) else ((ensure_int(raw_ks[0]), ensure_int(raw_ks[1])) if isinstance(raw_ks, tuple) and len(raw_ks) >= 2 else None))
    raw_sigma = attrs.get("sigma", None)
    sigma: float | tuple[float, float] | None = float(raw_sigma) if isinstance(raw_sigma, (int, float)) else ((ensure_float(raw_sigma[0]), ensure_float(raw_sigma[1])) if isinstance(raw_sigma, (tuple, list)) and len(raw_sigma) >= 2 else None)
    padding: str = str(attrs.get("padding", "same"))
    raw_data_format = attrs.get("data_format", None)
    data_format: str | None = str(raw_data_format) if raw_data_format is not None else None
    return kernel_size, sigma, padding, data_format


def _extract_extract_boxes_attributes(
    node: IRNode,
) -> tuple[int | tuple[int, int] | list[int] | None, str, float, str | None]:
    """Extract bounding box extraction attributes from an IR node.

    Args:
        node (IRNode): The IR node containing box extraction attributes.

    Returns:
        tuple[int | tuple[int, int] | list[int] | None, str, float, str | None]:
            Crop size, interpolation method, extrapolation value, and data format.
    """
    attrs = _node_attrs(node)
    raw_cs = attrs.get("crop_size")
    crop_size: int | tuple[int, int] | list[int] | None = raw_cs if isinstance(raw_cs, int) else ([ensure_int(d) for d in raw_cs] if isinstance(raw_cs, list) else ((ensure_int(raw_cs[0]), ensure_int(raw_cs[1])) if isinstance(raw_cs, tuple) and len(raw_cs) >= 2 else None))
    interpolation: str = str(attrs.get("interpolation", "bilinear"))
    extrapolation_value: float = float(ensure_float(attrs.get("extrapolation_value", 0.0)))
    raw_data_format = attrs.get("data_format", None)
    data_format: str | None = str(raw_data_format) if raw_data_format is not None else None
    return crop_size, interpolation, extrapolation_value, data_format


def _extract_stft_attributes(
    node: IRNode,
) -> tuple[int, int | None, int | None, str | None, bool, str, bool, bool]:
    """Extract generic STFT attributes from an IR node.

    Args:
        node (IRNode): The IR node containing generic STFT attributes.

    Returns:
        tuple[int, int | None, int | None, str | None, bool, str, bool, bool]:
            n_fft, hop_length, win_length, window, center, pad_mode, normalized, and onesided.
    """
    attrs = _node_attrs(node)
    raw_n_fft = attrs.get("n_fft", 2048)
    raw_hop_length = attrs.get("hop_length", None)
    raw_win_length = attrs.get("win_length", None)
    raw_window = attrs.get("window", None)
    raw_center = attrs.get("center", True)
    raw_pad_mode = attrs.get("pad_mode", "reflect")
    raw_normalized = attrs.get("normalized", False)
    raw_onesided = attrs.get("onesided", True)

    n_fft: int = int(ensure_int(raw_n_fft)) if raw_n_fft is not None else 2048
    hop_length: int | None = int(ensure_int(raw_hop_length)) if raw_hop_length is not None else None
    win_length: int | None = int(ensure_int(raw_win_length)) if raw_win_length is not None else None
    window: str | None = str(raw_window) if raw_window is not None else None
    center: bool = bool(raw_center)
    pad_mode: str = str(raw_pad_mode)
    normalized: bool = bool(raw_normalized)
    onesided: bool = bool(raw_onesided)

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
