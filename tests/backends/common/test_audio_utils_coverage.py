"""Tests for test_audio_utils_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.backends.common.audio_utils as audio_utils
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


def test_audio_utils_exhaustive() -> None:
    """Verify standard STFT and Mel filterbank attribute extraction across all branches."""
    # 1. extract_stft_attributes with all attributes provided
    node_full = IRNode(
        "node_stft_full",
        "STFT",
        attributes={
            "frame_length": 1024,
            "frame_step": 256,
            "fft_length": 2048,
            "window": "hamming",
            "center": False,
        },
    )
    fl, fs, f_len, win, center, f_str = audio_utils.extract_stft_attributes(node_full)
    assert fl == 1024
    assert fs == 256
    assert f_len == 2048
    assert win == "hamming"
    assert center is False
    assert f_str == "2048"

    # 2. extract_stft_attributes with missing attributes (defaults)
    node_empty = IRNode("node_stft_empty", "STFT", attributes={})
    fl_def, fs_def, f_len_def, win_def, center_def, f_str_def = audio_utils.extract_stft_attributes(node_empty)
    assert fl_def is None
    assert fs_def is None
    assert f_len_def is None
    assert win_def == "hann"
    assert center_def is True
    assert f_str_def == "None"

    # 3. extract_mel_attributes with all attributes provided
    node_mel_full = IRNode(
        "node_mel_full",
        "MelSpectrogram",
        attributes={
            "num_mel_bins": 80,
            "num_spectrogram_bins": 513,
            "sample_rate": 16000,
            "lower_edge_hertz": 40.0,
            "upper_edge_hertz": 8000.0,
            "num_mfccs": 20,
        },
    )
    n_mel, n_spec, sr, lower, upper, mfcc = audio_utils.extract_mel_attributes(node_mel_full)
    assert n_mel == 80
    assert n_spec == 513
    assert sr == 16000
    assert lower == 40.0
    assert upper == 8000.0
    assert mfcc == 20

    # 4. extract_mel_attributes with missing attributes (defaults)
    node_mel_empty = IRNode("node_mel_empty", "MelSpectrogram", attributes={})
    n_mel_d, n_spec_d, sr_d, lower_d, upper_d, mfcc_d = audio_utils.extract_mel_attributes(node_mel_empty)
    assert n_mel_d == 40
    assert n_spec_d is None
    assert sr_d is None
    assert lower_d == 20.0
    assert upper_d == 4000.0
    assert mfcc_d == 13
