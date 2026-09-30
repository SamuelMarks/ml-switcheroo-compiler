# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Configuration classes for operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ConvConfig:
    """Configuration for convolution operations."""

    window_strides: Sequence[int]
    padding: Sequence[tuple[int, int]] | str
    lhs_dilation: Sequence[int] | None = None
    rhs_dilation: Sequence[int] | None = None
    dimension_numbers: tuple[int, ...] | str | None = None
    feature_group_count: int = 1
    batch_group_count: int = 1


@dataclass
class WindowConfig:
    """Configuration for windowed operations."""

    window_dimensions: Sequence[int]
    window_strides: Sequence[int] | None = None
    padding: Sequence[tuple[int, int]] | str | None = None
    base_dilation: Sequence[int] | None = None
    window_dilation: Sequence[int] | None = None


@dataclass
class InitializerConfig:
    """Configuration for initializers."""

    scale: float
    mode: str
    distribution: str
    in_axis: int | Sequence[int] = -2
    out_axis: int | Sequence[int] = -1
    batch_axis: int | Sequence[int] = ()
    dtype: str | type | None = None


@dataclass
class SpaceConfig:
    """Configuration for space operations."""

    num: int = 50
    endpoint: bool = True
    base: float = 10.0
    dtype: str | type | None = None
    axis: int = 0


@dataclass
class STFTConfig:
    """Configuration for STFT/ISTFT operations."""

    frame_length: int
    frame_step: int
    fft_length: int | None = None
    window_fn: str | None = "hann"
    pad_end: bool = False


@dataclass
class BBoxConfig:
    """Bounding box configuration."""

    crop_size: tuple[int, int]
    interpolation: str = "bilinear"
    extrapolation_value: float = 0.0
    data_format: str | None = None


@dataclass
class PerspectiveConfig:
    """Perspective transformation configuration."""

    interpolation: str = "bilinear"
    fill_value: float = 0.0
    data_format: str | None = None


@dataclass
class BlurConfig:
    """Gaussian blur configuration."""

    kernel_size: int | tuple[int, int]
    sigma: float | tuple[float, float]
    data_format: str | None = None


@dataclass
class ElasticConfig:
    """Elastic transformation configuration."""

    interpolation: str = "bilinear"
    fill_value: float = 0.0
    data_format: str | None = None


@dataclass
class ResizeOptions:
    """Image resize configuration."""

    size: int | tuple[int, int]
    interpolation: str = "bilinear"
    align_corners: bool = False
    half_pixel_centers: bool = False
    data_format: str | None = None


@dataclass
class TriangularSolveOptions:
    """Configuration for triangular solve operations."""

    trans: int = 0
    lower: bool = False
    unit_diagonal: bool = False
    overwrite_b: bool = False
    check_finite: bool = True


@dataclass
class ConvDimensionNumbers:
    """ConvDimensionNumbers class."""

    lhs_spec: Sequence[int]
    rhs_spec: Sequence[int]
    out_spec: Sequence[int]


@dataclass
class ConvGeneralDilatedDimensionNumbers:
    """ConvGeneralDilatedDimensionNumbers class."""

    lhs_spec: Sequence[int]
    rhs_spec: Sequence[int]
    out_spec: Sequence[int]


@dataclass
class DotDimensionNumbers:
    """DotDimensionNumbers class."""

    lhs_contracting_dimensions: Sequence[int]
    rhs_contracting_dimensions: Sequence[int]
    lhs_batch_dimensions: Sequence[int]
    rhs_batch_dimensions: Sequence[int]


@dataclass
class GatherDimensionNumbers:
    """GatherDimensionNumbers class."""

    offset_dims: Sequence[int]
    collapsed_slice_dims: Sequence[int]
    start_index_map: Sequence[int]


class GatherScatterMode:
    """GatherScatterMode class."""

    STRICT = "strict"
    PROMISE_IN_BOUNDS = "promise_in_bounds"
    CLIP = "clip"


class Precision:
    """Precision class."""

    DEFAULT = "default"
    HIGH = "high"
    HIGHEST = "highest"


class PrecisionLike:
    """PrecisionLike class."""

    value: str | None

    def __init__(self, value: str | None = None) -> None:
        """Initialize precision descriptor.

        Args:
            value (str | None): Optional precision name or setting.
        """
        self.value = value


class RandomAlgorithm:
    """RandomAlgorithm class."""

    DEFAULT = "default"
    THREEFRY = "threefry"
    RBGS = "rbgs"


class RoundingMethod:
    """RoundingMethod class."""

    AWAY_FROM_ZERO = "away_from_zero"
    TO_NEAREST_EVEN = "to_nearest_even"


@dataclass
class ScatterDimensionNumbers:
    """ScatterDimensionNumbers class."""

    update_window_dims: Sequence[int]
    inserted_window_dims: Sequence[int]
    scatter_dims_to_operand_dims: Sequence[int]
