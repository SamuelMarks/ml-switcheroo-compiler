"""Tests for test_generator_utils_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.backends.generator_utils as generator_utils
from ml_switcheroo_compiler.ir.core import IRNode


class _MockDimWithId:
    """Mock dimension exposing an id attribute."""

    def __init__(self, dim_id: str) -> None:
        """Initialize mock dimension.

        Args:
            dim_id (str): Identifier for dimension.
        """
        self.id = dim_id


class _MockShapeContainer:
    """Mock container exposing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock shape container.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape = shape


class _MockShapeMetaContainer:
    """Mock container exposing a shape_metadata attribute."""

    def __init__(self, shape_metadata: tuple[int, ...]) -> None:
        """Initialize mock shape metadata container.

        Args:
            shape_metadata (tuple[int, ...]): Shape metadata tuple.
        """
        self.shape_metadata = shape_metadata


class _MockGenerator:
    """Mock generator for variable AST visitor."""

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix for AST emissions.

        Returns:
            str: Fallback prefix string.
        """
        return "mock_backend"


def test_generator_utils_exhaustive() -> None:
    """Verify attribute extraction across audio, resize, filter, boxes, and STFT helpers."""
    # 1. _extract_audio_stft_attributes custom & defaults
    node_stft = IRNode(
        "node_audio_stft",
        "AudioSTFT",
        attributes={"frame_length": 1024, "frame_step": 256, "fft_length": 512, "window_fn": "hamming", "pad_end": True},
    )
    fl, fs, fft_l, win, pad = generator_utils._extract_audio_stft_attributes(node_stft)
    assert (fl, fs, fft_l, win, pad) == (1024, 256, 512, "hamming", True)

    node_stft_def = IRNode("node_audio_stft_def", "AudioSTFT")
    fl_d, fs_d, fft_l_d, win_d, pad_d = generator_utils._extract_audio_stft_attributes(node_stft_def)
    assert (fl_d, fs_d, fft_l_d, win_d, pad_d) == (2048, 512, None, "hann", False)

    # 2. _extract_resize_attributes custom & defaults
    node_resize = IRNode(
        "node_resize",
        "Resize",
        attributes={"size": (256, 256), "interpolation": "nearest", "align_corners": True, "antialias": True, "data_format": "NHWC"},
    )
    sz, interp, align, aa, fmt = generator_utils._extract_resize_attributes(node_resize)
    assert (sz, interp, align, aa, fmt) == ((256, 256), "nearest", True, True, "NHWC")

    node_resize_def = IRNode("node_resize_def", "Resize")
    sz_d, interp_d, align_d, aa_d, fmt_d = generator_utils._extract_resize_attributes(node_resize_def)
    assert (sz_d, interp_d, align_d, aa_d, fmt_d) == (None, "bilinear", False, False, None)

    # 3. _extract_vision_transform_attributes custom & defaults
    node_trans = IRNode(
        "node_trans",
        "AffineTransform",
        attributes={"interpolation": "bicubic", "fill_value": 1.5, "data_format": "NCHW"},
    )
    t_interp, t_fill, t_fmt = generator_utils._extract_vision_transform_attributes(node_trans)
    assert (t_interp, t_fill, t_fmt) == ("bicubic", 1.5, "NCHW")

    node_trans_def = IRNode("node_trans_def", "AffineTransform")
    t_interp_d, t_fill_d, t_fmt_d = generator_utils._extract_vision_transform_attributes(node_trans_def)
    assert (t_interp_d, t_fill_d, t_fmt_d) == ("bilinear", 0.0, None)

    # 4. _extract_filter_attributes custom & defaults
    node_filter = IRNode(
        "node_filter",
        "GaussianBlur",
        attributes={"kernel_size": (5, 5), "sigma": (1.0, 1.0), "padding": "valid", "data_format": "NCHW"},
    )
    k_sz, k_sig, k_pad, k_fmt = generator_utils._extract_filter_attributes(node_filter)
    assert (k_sz, k_sig, k_pad, k_fmt) == ((5, 5), (1.0, 1.0), "valid", "NCHW")

    node_filter_def = IRNode("node_filter_def", "GaussianBlur")
    k_sz_d, k_sig_d, k_pad_d, k_fmt_d = generator_utils._extract_filter_attributes(node_filter_def)
    assert (k_sz_d, k_sig_d, k_pad_d, k_fmt_d) == (None, None, "same", None)

    # 5. _extract_extract_boxes_attributes custom & defaults
    node_box = IRNode(
        "node_box",
        "ExtractBoxes",
        attributes={"crop_size": (64, 64), "interpolation": "nearest", "extrapolation_value": 0.5, "data_format": "NHWC"},
    )
    c_sz, c_interp, c_extrap, c_fmt = generator_utils._extract_extract_boxes_attributes(node_box)
    assert (c_sz, c_interp, c_extrap, c_fmt) == ((64, 64), "nearest", 0.5, "NHWC")

    node_box_def = IRNode("node_box_def", "ExtractBoxes")
    c_sz_d, c_interp_d, c_extrap_d, c_fmt_d = generator_utils._extract_extract_boxes_attributes(node_box_def)
    assert (c_sz_d, c_interp_d, c_extrap_d, c_fmt_d) == (None, "bilinear", 0.0, None)

    # 6. _extract_stft_attributes custom & defaults
    node_stft_full = IRNode(
        "node_stft_full",
        "GenericSTFT",
        attributes={
            "n_fft": 1024,
            "hop_length": 128,
            "win_length": 512,
            "window": "blackman",
            "center": False,
            "pad_mode": "constant",
            "normalized": True,
            "onesided": False,
        },
    )
    n_fft, hop, win_l, win_s, ctr, p_mode, norm, oneside = generator_utils._extract_stft_attributes(node_stft_full)
    assert (n_fft, hop, win_l, win_s, ctr, p_mode, norm, oneside) == (
        1024,
        128,
        512,
        "blackman",
        False,
        "constant",
        True,
        False,
    )

    node_stft_gen_def = IRNode("node_stft_gen_def", "GenericSTFT")
    n_fft_d, hop_d, win_l_d, win_s_d, ctr_d, p_mode_d, norm_d, oneside_d = generator_utils._extract_stft_attributes(node_stft_gen_def)
    assert (n_fft_d, hop_d, win_l_d, win_s_d, ctr_d, p_mode_d, norm_d, oneside_d) == (
        2048,
        None,
        None,
        None,
        True,
        "reflect",
        False,
        True,
    )
