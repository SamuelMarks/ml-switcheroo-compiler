"""Tests for test_color."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.color import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    augmix,
    auto_contrast,
    degeneration,
    equalization,
    hsv_to_rgb,
    invert,
    posterize,
    rand_augment,
    random_color_jitter,
    random_erasing,
    rgb_to_grayscale,
    rgb_to_hsv,
    rgb_to_yiq,
    rgb_to_yuv,
    solarize,
    yiq_to_rgb,
    yuv_to_rgb,
)

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def _test_vision_color_funcs_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_vision_color_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    mocker = _state["mocker"]
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.vision.color.get_op")
    mocker.patch("ml_switcheroo_compiler.ops.vision.color._emit_shape_node", return_value="node")
    mock_op = mocker.MagicMock()
    setattr(mock_get_op, "return_value", mocker.MagicMock(return_value=mock_op))
    assert rgb_to_hsv(t) == mock_op()
    assert hsv_to_rgb(t) == mock_op()
    assert adjust_hue(t, 0.5) == mock_op()
    assert adjust_saturation(t, 0.5) == "node"
    assert adjust_contrast(t, 0.5) == "node"
    assert adjust_brightness(t, 0.5) == mock_op()
    assert rgb_to_grayscale(t) == "node"
    assert random_color_jitter(t, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1) == "node"
    assert solarize(t) == mock_op()
    assert invert(t) == mock_op()
    assert posterize(t, 4) == mock_op()
    assert degeneration(t) == mock_op()
    assert augmix(t) == mock_op()
    assert auto_contrast(t) == mock_op()
    assert rand_augment(t) == mock_op()
    assert random_erasing(t) == mock_op()
    getattr(_state, "update")({"mock_get_op": mock_get_op, "mock_op": mock_op, "t": t})
    _state.update(locals())


def _test_vision_color_funcs_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_vision_color_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    mock_op, mocker, t = _state["mock_op"], _state["mocker"], _state["t"]
    assert cast("EagerCallableProtocol", equalization)(t) == cast("EagerCallableProtocol", mock_op)()
    assert cast("EagerCallableProtocol", rgb_to_yiq)(t) == "node"
    assert cast("EagerCallableProtocol", yiq_to_rgb)(t) == "node"
    assert cast("EagerCallableProtocol", rgb_to_yuv)(t) == "node"
    assert cast("EagerCallableProtocol", yuv_to_rgb)(t) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.color.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", rgb_to_hsv)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", hsv_to_rgb)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", adjust_hue)(t, 0.5), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", adjust_saturation)(t, 0.5), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", adjust_contrast)(t, 0.5), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", adjust_brightness)(t, 0.5), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", rgb_to_grayscale)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", random_color_jitter)(t, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", solarize)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", invert)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", posterize)(t, 4), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", degeneration)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", augmix)(t), "config")).shape == (2, 3)
    getattr(_state, "update")({"mock_backend": mock_backend})
    _state.update(locals())


def _test_vision_color_funcs_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_vision_color_funcs.

    Args:
        _state: Shared test state dictionary.
    """
    t = _state["t"]
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", auto_contrast)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", rand_augment)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", random_erasing)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", equalization)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", rgb_to_yiq)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", yiq_to_rgb)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", rgb_to_yuv)(t), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(cast("EagerCallableProtocol", yuv_to_rgb)(t), "config")).shape == (2, 3)


def test_vision_color_funcs(mocker: EagerTensorProtocol) -> None:
    """Test function test_vision_color_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
    _state: dict[str, EagerTensorProtocol] = {"mocker": mocker}
    _test_vision_color_funcs_part_1(_state)
    _test_vision_color_funcs_part_2(_state)
    _test_vision_color_funcs_part_3(_state)
    assert len(_state) > 0


def test_color_opdef_infer_shape() -> None:
    """Test function test_color_opdef_infer_shape."""
    from ml_switcheroo_compiler.ops.vision.color import AdjustBrightness, AdjustContrast, AdjustHue, AdjustSaturation, AugMix, AutoContrast, Equalization, Invert, Posterize, RgbToGrayscale, Solarize

    class DummyTensor:
        """Test class DummyTensor."""

        shape = (1, 3, 32, 32)

    class DummyTensorLast:
        """Test class DummyTensorLast."""

        shape = (1, 32, 32, 3)

    t = DummyTensor()
    t_last = DummyTensorLast()

    assert cast("EagerCallableProtocol", AdjustBrightness)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", AdjustContrast)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", AdjustHue)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", AdjustSaturation)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", AugMix)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", AutoContrast)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", Equalization)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", Invert)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", Posterize)().infer_shape(t) == (1, 3, 32, 32)
    assert cast("EagerCallableProtocol", Solarize)().infer_shape(t) == (1, 3, 32, 32)

    # RgbToGrayscale
    assert cast("EagerCallableProtocol", RgbToGrayscale)().infer_shape(t, data_format="channels_first") == (1, 1, 32, 32)
    assert cast("EagerCallableProtocol", RgbToGrayscale)().infer_shape(t_last, data_format="channels_last") == (1, 32, 32, 1)

    # Empty shape
    class EmptyTensor:
        """Test class EmptyTensor."""

        shape = ()

    assert cast("EagerCallableProtocol", RgbToGrayscale)().infer_shape(cast("EagerCallableProtocol", EmptyTensor)()) == ()
