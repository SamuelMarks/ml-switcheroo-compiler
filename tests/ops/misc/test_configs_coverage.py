"""Tests for test_configs_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.configs as configs_mod


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


def test_configs_exhaustive() -> None:
    """Instantiate and verify every configuration dataclass in ops/configs.py."""
    conv_cfg = configs_mod.ConvConfig(
        window_strides=[1, 1],
        padding="SAME",
        lhs_dilation=[1, 1],
        rhs_dilation=[1, 1],
        dimension_numbers="NCHW",
        feature_group_count=2,
        batch_group_count=1,
    )
    assert conv_cfg.feature_group_count == 2

    win_cfg = configs_mod.WindowConfig(
        window_dimensions=[3, 3],
        window_strides=[2, 2],
        padding=[(0, 0), (1, 1)],
        base_dilation=[1, 1],
        window_dilation=[1, 1],
    )
    assert win_cfg.window_dimensions == [3, 3]

    init_cfg = configs_mod.InitializerConfig(
        scale=2.0,
        mode="fan_out",
        distribution="normal",
        in_axis=0,
        out_axis=1,
        batch_axis=[2],
        dtype="float32",
    )
    assert init_cfg.scale == 2.0

    space_cfg = configs_mod.SpaceConfig(num=25, endpoint=False, base=2.0, dtype="int32", axis=1)
    assert space_cfg.num == 25

    stft_cfg = configs_mod.STFTConfig(frame_length=1024, frame_step=256, fft_length=2048, window_fn="boxcar", pad_end=True)
    assert stft_cfg.pad_end is True

    bbox_cfg = configs_mod.BBoxConfig(crop_size=(128, 128), interpolation="nearest", extrapolation_value=1.0, data_format="NCHW")
    assert bbox_cfg.crop_size == (128, 128)

    persp_cfg = configs_mod.PerspectiveConfig(interpolation="bicubic", fill_value=255.0, data_format="NHWC")
    assert persp_cfg.fill_value == 255.0

    blur_cfg = configs_mod.BlurConfig(kernel_size=(5, 5), sigma=(1.5, 1.5), data_format="NCHW")
    assert blur_cfg.kernel_size == (5, 5)

    elastic_cfg = configs_mod.ElasticConfig(interpolation="bilinear", fill_value=0.0, data_format="NHWC")
    assert elastic_cfg.interpolation == "bilinear"

    resize_opts = configs_mod.ResizeOptions(size=(224, 224), interpolation="bilinear", align_corners=True, half_pixel_centers=True, data_format="NCHW")
    assert resize_opts.half_pixel_centers is True

    solve_opts = configs_mod.TriangularSolveOptions(trans=1, lower=True, unit_diagonal=True, overwrite_b=True, check_finite=False)
    assert solve_opts.lower is True

    p_like_empty = configs_mod.PrecisionLike()
    assert p_like_empty.value is None

    p_like_val = configs_mod.PrecisionLike("highest")
    assert p_like_val.value == "highest"

    assert configs_mod.GatherScatterMode.STRICT == "strict"
    assert configs_mod.GatherScatterMode.PROMISE_IN_BOUNDS == "promise_in_bounds"
    assert configs_mod.GatherScatterMode.CLIP == "clip"

    assert configs_mod.Precision.DEFAULT == "default"
    assert configs_mod.Precision.HIGH == "high"
    assert configs_mod.Precision.HIGHEST == "highest"

    assert configs_mod.RandomAlgorithm.DEFAULT == "default"
    assert configs_mod.RandomAlgorithm.THREEFRY == "threefry"
    assert configs_mod.RandomAlgorithm.RBGS == "rbgs"

    assert configs_mod.RoundingMethod.AWAY_FROM_ZERO == "away_from_zero"
    assert configs_mod.RoundingMethod.TO_NEAREST_EVEN == "to_nearest_even"
