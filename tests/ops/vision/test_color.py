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


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_vision_color_funcs(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.vision.color.get_op")
    mocker.patch("ml_switcheroo_compiler.ops.vision.color._emit_shape_node", return_value="node")
    mock_op = mocker.MagicMock()
    mock_get_op.return_value = mocker.MagicMock(return_value=mock_op)
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
    assert equalization(t) == mock_op()
    assert rgb_to_yiq(t) == "node"
    assert yiq_to_rgb(t) == "node"
    assert rgb_to_yuv(t) == "node"
    assert yuv_to_rgb(t) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.color.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))
    assert rgb_to_hsv(t).config.shape == (2, 3)
    assert hsv_to_rgb(t).config.shape == (2, 3)
    assert adjust_hue(t, 0.5).config.shape == (2, 3)
    assert adjust_saturation(t, 0.5).config.shape == (2, 3)
    assert adjust_contrast(t, 0.5).config.shape == (2, 3)
    assert adjust_brightness(t, 0.5).config.shape == (2, 3)
    assert rgb_to_grayscale(t).config.shape == (2, 3)
    assert random_color_jitter(t, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1).config.shape == (2, 3)
    assert solarize(t).config.shape == (2, 3)
    assert invert(t).config.shape == (2, 3)
    assert posterize(t, 4).config.shape == (2, 3)
    assert degeneration(t).config.shape == (2, 3)
    assert augmix(t).config.shape == (2, 3)
    assert auto_contrast(t).config.shape == (2, 3)
    assert rand_augment(t).config.shape == (2, 3)
    assert random_erasing(t).config.shape == (2, 3)
    assert equalization(t).config.shape == (2, 3)
    assert rgb_to_yiq(t).config.shape == (2, 3)
    assert yiq_to_rgb(t).config.shape == (2, 3)
    assert rgb_to_yuv(t).config.shape == (2, 3)
    assert yuv_to_rgb(t).config.shape == (2, 3)


def test_color_opdef_infer_shape():
    from ml_switcheroo_compiler.ops.vision.color import AdjustBrightness, AdjustContrast, AdjustHue, AdjustSaturation, AugMix, AutoContrast, Equalization, Invert, Posterize, RgbToGrayscale, Solarize

    class DummyTensor:
        shape = (1, 3, 32, 32)

    class DummyTensorLast:
        shape = (1, 32, 32, 3)

    t = DummyTensor()
    t_last = DummyTensorLast()

    assert AdjustBrightness().infer_shape(t) == (1, 3, 32, 32)
    assert AdjustContrast().infer_shape(t) == (1, 3, 32, 32)
    assert AdjustHue().infer_shape(t) == (1, 3, 32, 32)
    assert AdjustSaturation().infer_shape(t) == (1, 3, 32, 32)
    assert AugMix().infer_shape(t) == (1, 3, 32, 32)
    assert AutoContrast().infer_shape(t) == (1, 3, 32, 32)
    assert Equalization().infer_shape(t) == (1, 3, 32, 32)
    assert Invert().infer_shape(t) == (1, 3, 32, 32)
    assert Posterize().infer_shape(t) == (1, 3, 32, 32)
    assert Solarize().infer_shape(t) == (1, 3, 32, 32)

    # RgbToGrayscale
    assert RgbToGrayscale().infer_shape(t, data_format="channels_first") == (1, 1, 32, 32)
    assert RgbToGrayscale().infer_shape(t_last, data_format="channels_last") == (1, 32, 32, 1)

    # Empty shape
    class EmptyTensor:
        shape = ()

    assert RgbToGrayscale().infer_shape(EmptyTensor()) == ()
