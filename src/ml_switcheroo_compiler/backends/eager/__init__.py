# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

"""Eager backend utilities."""

from .audio import istft_eager, mel_filterbank_eager, mfcc_eager
from .core_group_ops import _group_mean, _group_norm, _group_variance
from .core_math_ops import (
    _allclose,
    _einsum,
    _erfinv,
    _fft,
    _fftn,
    _nan_to_num,
    _pmean,
    _psum,
    _rfft,
    _segment_sum,
    _true_divide,
)
from .optimizers import apply_adagrad, apply_adam, apply_ftrl, apply_rmsprop
from .random_ops import prng_key, rand, randint, randn, random_fold_in, random_split
from .signal import gaussian_blur_eager, median_filter_eager
from .vision_filtering import _extract_volume_patches, extract_bounding_boxes_eager, iou_eager, nms_eager
from .vision_transforms import elastic_transform_eager, perspective_transform_eager, resize_eager

__all__ = [
    "_allclose",
    "_einsum",
    "_erfinv",
    "_extract_volume_patches",
    "_fft",
    "_fftn",
    "_group_mean",
    "_group_norm",
    "_group_variance",
    "_nan_to_num",
    "_pmean",
    "_psum",
    "_rfft",
    "_segment_sum",
    "_true_divide",
    "elastic_transform_eager",
    "extract_bounding_boxes_eager",
    "gaussian_blur_eager",
    "iou_eager",
    "istft_eager",
    "median_filter_eager",
    "mel_filterbank_eager",
    "mfcc_eager",
    "nms_eager",
    "perspective_transform_eager",
    "resize_eager",
]
from .types_utils import generic_array as generic_array
from .types_utils import generic_asarray as generic_asarray
from .types_utils import generic_item as generic_item
from .types_utils import generic_zeros as generic_zeros

__all__ = [
    "_allclose",
    "_einsum",
    "_erfinv",
    "_extract_volume_patches",
    "_fft",
    "_fftn",
    "_group_mean",
    "_group_norm",
    "_group_variance",
    "_nan_to_num",
    "_pmean",
    "_psum",
    "_rfft",
    "_segment_sum",
    "_true_divide",
    "elastic_transform_eager",
    "extract_bounding_boxes_eager",
    "gaussian_blur_eager",
    "iou_eager",
    "istft_eager",
    "median_filter_eager",
    "mel_filterbank_eager",
    "mfcc_eager",
    "nms_eager",
    "perspective_transform_eager",
    "resize_eager",
]
