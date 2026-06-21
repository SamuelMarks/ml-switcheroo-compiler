"""Configuration classes for operations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class ConvConfig:
    """Configuration for convolution operations."""

    window_strides: Sequence[int]
    padding: Union[Sequence[tuple[int, int]], str]
    lhs_dilation: Optional[Sequence[int]] = None
    rhs_dilation: Optional[Sequence[int]] = None
    dimension_numbers: Any = None
    feature_group_count: int = 1
    batch_group_count: int = 1


@dataclass
class WindowConfig:
    """Configuration for windowed operations."""

    window_dimensions: Sequence[int]
    window_strides: Optional[Sequence[int]] = None
    padding: Optional[Union[Sequence[tuple[int, int]], str]] = None
    base_dilation: Optional[Sequence[int]] = None
    window_dilation: Optional[Sequence[int]] = None


@dataclass
class InitializerConfig:
    """Configuration for initializers."""

    scale: float
    mode: str
    distribution: str
    in_axis: Union[int, Sequence[int]] = -2
    out_axis: Union[int, Sequence[int]] = -1
    batch_axis: Union[int, Sequence[int]] = ()
    dtype: Any = None


@dataclass
class SpaceConfig:
    """Configuration for space operations."""

    num: int = 50
    endpoint: bool = True
    base: float = 10.0
    dtype: Any = None
    axis: int = 0


@dataclass
class STFTConfig:
    """Configuration for STFT/ISTFT operations."""

    frame_length: int
    frame_step: int
    fft_length: Optional[int] = None
    window_fn: Optional[str] = "hann"
    pad_end: bool = False


@dataclass
class BBoxConfig:
    """Bounding box configuration."""

    crop_size: tuple[int, int]
    interpolation: str = "bilinear"
    extrapolation_value: float = 0.0
    data_format: Optional[str] = None


@dataclass
class PerspectiveConfig:
    """Perspective transformation configuration."""

    interpolation: str = "bilinear"
    fill_value: float = 0.0
    data_format: Optional[str] = None


@dataclass
class BlurConfig:
    """Gaussian blur configuration."""

    kernel_size: Union[int, tuple[int, int]]
    sigma: Union[float, tuple[float, float]]
    data_format: Optional[str] = None


@dataclass
class ElasticConfig:
    """Elastic transformation configuration."""

    interpolation: str = "bilinear"
    fill_value: float = 0.0
    data_format: Optional[str] = None


@dataclass
class ResizeOptions:
    """Image resize configuration."""

    size: Union[int, tuple[int, int]]
    interpolation: str = "bilinear"
    align_corners: bool = False
    half_pixel_centers: bool = False
    data_format: Optional[str] = None


@dataclass
class TriangularSolveOptions:
    """Configuration for triangular solve operations."""

    trans: int = 0
    lower: bool = False
    unit_diagonal: bool = False
    overwrite_b: bool = False
    check_finite: bool = True


class ConvDimensionNumbers:
    """ConvDimensionNumbers class."""

    pass


class ConvGeneralDilatedDimensionNumbers:
    """ConvGeneralDilatedDimensionNumbers class."""

    pass


class DotDimensionNumbers:
    """DotDimensionNumbers class."""

    pass


class GatherDimensionNumbers:
    """GatherDimensionNumbers class."""

    pass


class GatherScatterMode:
    """GatherScatterMode class."""

    pass


class Precision:
    """Precision class."""

    pass


class PrecisionLike:
    """PrecisionLike class."""

    pass


class RandomAlgorithm:
    """RandomAlgorithm class."""

    pass


class RoundingMethod:
    """RoundingMethod class."""

    pass


class ScatterDimensionNumbers:
    """ScatterDimensionNumbers class."""

    pass
