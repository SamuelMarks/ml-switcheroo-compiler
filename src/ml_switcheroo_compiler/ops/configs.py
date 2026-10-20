"""Configuration classes for operations."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.core.shape import AttrValue


class _PositionalConfigModel(pydantic.BaseModel):
    """Base operation configuration model supporting positional and keyword initialization."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    def __init__(self, *args: AttrValue, **kwargs: AttrValue) -> None:
        """Initialize configuration model from positional or keyword arguments."""
        super().__init__(**{**dict(zip(self.__class__.model_fields.keys(), args)), **kwargs})


class ConvConfig(_PositionalConfigModel):
    """Configuration for convolution operations."""

    window_strides: Sequence[int] = (1,)
    padding: Sequence[tuple[int, int]] | str | None = "VALID"
    lhs_dilation: Sequence[int] | None = None
    rhs_dilation: Sequence[int] | None = None
    dimension_numbers: AttrValue = None
    feature_group_count: int = 1
    batch_group_count: int = 1
    precision: AttrValue = None
    preferred_element_type: AttrValue = None


class WindowConfig(_PositionalConfigModel):
    """Configuration for windowed operations."""

    window_dimensions: Sequence[int] = (1,)
    window_strides: Sequence[int] | None = None
    padding: Sequence[tuple[int, int] | int] | str | None = None
    base_dilation: Sequence[int] | None = None
    window_dilation: Sequence[int] | None = None


class InitializerConfig(_PositionalConfigModel):
    """Configuration for initializers."""

    scale: float = 1.0
    mode: str = "fan_in"
    distribution: str = "truncated_normal"
    in_axis: int | Sequence[int] = -2
    out_axis: int | Sequence[int] = -1
    batch_axis: int | Sequence[int] = ()
    dtype: str | type | None = "float32"


class SpaceConfig(_PositionalConfigModel):
    """Configuration for space operations."""

    num: int = 50
    endpoint: bool = True
    base: float = 10.0
    dtype: str | type | None = None
    axis: int = 0


class STFTConfig(_PositionalConfigModel):
    """Configuration for STFT/ISTFT operations."""

    frame_length: int = 256
    frame_step: int = 128
    fft_length: int | None = None
    window_fn: str | None = "hann"
    pad_end: bool = False


class BBoxConfig(_PositionalConfigModel):
    """Bounding box configuration."""

    crop_size: tuple[int, int] = (1, 1)
    interpolation: str = "bilinear"
    extrapolation_value: float = 0.0
    data_format: str | None = None


class PerspectiveConfig(_PositionalConfigModel):
    """Perspective transformation configuration."""

    interpolation: str | AttrValue = "bilinear"
    fill_value: float | AttrValue = 0.0
    data_format: str | AttrValue | None = None


class BlurConfig(_PositionalConfigModel):
    """Gaussian blur configuration."""

    kernel_size: int | Sequence[int] | AttrValue
    sigma: float | Sequence[float] | AttrValue
    data_format: str | AttrValue | None = None

    def __init__(self, *args: AttrValue, **kwargs: AttrValue) -> None:
        """Initialize BlurConfig requiring kernel_size and sigma."""
        merged = {**dict(zip(self.__class__.model_fields.keys(), args)), **kwargs}
        if "kernel_size" not in merged or "sigma" not in merged:
            msg = "BlurConfig.__init__() missing required arguments: 'kernel_size' and 'sigma'"
            raise TypeError(msg)
        super().__init__(**merged)


class ElasticConfig(_PositionalConfigModel):
    """Elastic transformation configuration."""

    interpolation: str | AttrValue = "bilinear"
    fill_value: float | AttrValue = 0.0
    data_format: str | AttrValue | None = None


class ResizeOptions(_PositionalConfigModel):
    """Image resize configuration."""

    size: int | Sequence[int] = (1, 1)
    interpolation: str = "bilinear"
    align_corners: bool = False
    half_pixel_centers: bool = False
    data_format: str | None = None


class TriangularSolveOptions(_PositionalConfigModel):
    """Configuration for triangular solve operations."""

    trans: int = 0
    lower: bool = False
    unit_diagonal: bool = False
    overwrite_b: bool = False
    check_finite: bool = True


class ConvDimensionNumbers(_PositionalConfigModel):
    """ConvDimensionNumbers class."""

    lhs_spec: Sequence[int] | str = ()
    rhs_spec: Sequence[int] | str = ()
    out_spec: Sequence[int] | str = ()


class ConvGeneralDilatedDimensionNumbers(_PositionalConfigModel):
    """ConvGeneralDilatedDimensionNumbers class."""

    lhs_spec: Sequence[int] | str = ()
    rhs_spec: Sequence[int] | str = ()
    out_spec: Sequence[int] | str = ()


class DotDimensionNumbers(_PositionalConfigModel):
    """DotDimensionNumbers class."""

    lhs_contracting_dimensions: Sequence[int] = ()
    rhs_contracting_dimensions: Sequence[int] = ()
    lhs_batch_dimensions: Sequence[int] = ()
    rhs_batch_dimensions: Sequence[int] = ()


class GatherDimensionNumbers(_PositionalConfigModel):
    """GatherDimensionNumbers class."""

    offset_dims: Sequence[int] = ()
    collapsed_slice_dims: Sequence[int] = ()
    start_index_map: Sequence[int] = ()


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


class ScatterDimensionNumbers(_PositionalConfigModel):
    """ScatterDimensionNumbers class."""

    update_window_dims: Sequence[int] = ()
    inserted_window_dims: Sequence[int] = ()
    scatter_dims_to_operand_dims: Sequence[int] = ()
