"""Vision operations module."""

from ml_switcheroo_compiler.ops.vision.geometric import (
    affine_generator,
    affine_transform,
    crop,
    crop_and_resize,
    elastic_transform,
    extract_bounding_boxes,
    flip_left_right,
    flip_up_down,
    pad_to_bounding_box,
    perspective_transform,
    resize_bicubic,
    resize_bilinear,
    resize_lanczos3,
    resize_nearest,
)
from ml_switcheroo_compiler.ops.vision.color import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    hsv_to_rgb,
    rgb_to_hsv,
)
from ml_switcheroo_compiler.ops.vision.filtering import (
    gaussian_blur,
    iou,
    median_filter,
    non_max_suppression,
)

from ml_switcheroo_compiler.ops.vision import ops  # noqa: F401

__all__ = [
    "adjust_contrast",
    "adjust_hue",
    "adjust_saturation",
    "affine_transform",
    "affine_generator",
    "flip_left_right",
    "flip_up_down",
    "adjust_brightness",
    "crop",
    "crop_and_resize",
    "pad_to_bounding_box",
    "perspective_transform",
    "elastic_transform",
    "gaussian_blur",
    "median_filter",
    "extract_bounding_boxes",
    "iou",
    "non_max_suppression",
    "hsv_to_rgb",
    "resize_bilinear",
    "resize_nearest",
    "resize_bicubic",
    "resize_lanczos3",
    "rgb_to_hsv",
]
