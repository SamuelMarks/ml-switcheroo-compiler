"""Extra vision ops for eager numpy execution."""

import numpy as np
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _np_perspective_transform(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import perspective_transform_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import PerspectiveConfig

        config_obj = PerspectiveConfig(
            interpolation=config_obj.get("interpolation", "bilinear"),
            fill_value=config_obj.get("fill_value", 0.0),
            data_format=config_obj.get("data_format", None),
        )

    return perspective_transform_eager(backend_module, images, start_points, end_points, config_obj)


@numpy_eager_registry.register("ElasticTransform")
def _np_elastic_transform(
    backend_module: object,
    images: object,
    displacement: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import elastic_transform_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import ElasticConfig

        config_obj = ElasticConfig(
            interpolation=config_obj.get("interpolation", "bilinear"),
            fill_value=config_obj.get("fill_value", 0.0),
            data_format=config_obj.get("data_format", None),
        )

    return elastic_transform_eager(backend_module, images, displacement, config_obj)


@numpy_eager_registry.register("GaussianBlur")
def _np_gaussian_blur(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.signal import gaussian_blur_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import BlurConfig

        config_obj = BlurConfig(
            kernel_size=config_obj.get("kernel_size", (3, 3)),
            sigma=config_obj.get("sigma", (1.0, 1.0)),
            data_format=config_obj.get("data_format", None),
        )

    return gaussian_blur_eager(backend_module, images, config_obj)


@numpy_eager_registry.register("MedianFilter")
def _np_median_filter(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import median_filter_eager

    return median_filter_eager(backend_module, images, **kwargs)


@numpy_eager_registry.register("ExtractBoundingBoxes")
def _np_extract_bounding_boxes(
    backend_module: object,
    images: object,
    boxes: object,
    box_indices: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_filtering import extract_bounding_boxes_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import BBoxConfig

        config_obj = BBoxConfig(
            crop_size=config_obj.get("crop_size", (0, 0)),
            interpolation=config_obj.get("interpolation", "bilinear"),
            extrapolation_value=config_obj.get("extrapolation_value", 0.0),
            data_format=config_obj.get("data_format", None),
        )

    return extract_bounding_boxes_eager(backend_module, images, boxes, box_indices, config_obj)


@numpy_eager_registry.register("IoU")
def _np_iou(
    backend_module: object,
    boxes1: object,
    boxes2: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import iou_eager

    return iou_eager(backend_module, boxes1, boxes2, **kwargs)


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(
    backend_module: object,
    boxes: object,
    scores: object,
    max_output_size: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import nms_eager

    return nms_eager(backend_module, boxes, scores, max_output_size=max_output_size, **kwargs)


@numpy_eager_registry.register("ResizeBicubic")
def _np_resize_bicubic(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import resize_eager

    return resize_eager(backend_module, images, interpolation="bicubic", **kwargs)


@numpy_eager_registry.register("ResizeLanczos3")
def _np_resize_lanczos3(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import resize_eager

    return resize_eager(backend_module, images, interpolation="lanczos3", **kwargs)


@numpy_eager_registry.register("Stft")
def _np_stft(
    np: object,
    input_tensor: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import stft_eager

    return stft_eager(np, input_tensor, **kwargs)


@numpy_eager_registry.register("Stft")
def _np_stft(
    np: object,
    input_tensor: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import stft_eager

    return stft_eager(np, input_tensor, **kwargs)


@numpy_eager_registry.register("Istft")
def _np_istft(
    backend_module: object,
    stft_tensor: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import istft_eager

    return istft_eager(backend_module, stft_tensor, **kwargs)


@numpy_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(
    backend_module: object,
    _: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager

    return mel_filterbank_eager(backend_module, None, kwargs.get("config", kwargs))


@numpy_eager_registry.register("Mfcc")
def _np_mfcc(
    backend_module: object,
    spectrogram: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import mfcc_eager

    return mfcc_eager(backend_module, spectrogram, kwargs.get("config", kwargs))


@numpy_eager_registry.register("PowerIteration")
def _np_power_iteration(
    backend_module: object, w: object, *args: object, **kwargs: object
) -> object:

    num_iters = kwargs.get("num_iters", 1)
    u = kwargs.get("u", None)
    if u is None:
        u = np.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype)
    for _ in range(num_iters):
        w_t = np.swapaxes(w, -1, -2)
        v = np.matmul(w_t, u)
        v = v / (np.linalg.norm(v, axis=-2, keepdims=True) + 1e-12)
        u = np.matmul(w, v)
        u = u / (np.linalg.norm(u, axis=-2, keepdims=True) + 1e-12)
    sigma = np.matmul(np.swapaxes(u, -1, -2), np.matmul(w, v))
    return np.squeeze(v, -1), np.squeeze(u, -1), np.squeeze(np.squeeze(sigma, -1), -1)


@numpy_eager_registry.register("StringToHash")
def _np_string_to_hash(
    backend_module: object, input_tensor: object, num_buckets: int, **kwargs: object
) -> object:
    import hashlib

    # We will use hashlib.md5 as a stable hash (or siphash if available, but md5 is built-in)
    # Numpy arrays of strings can be iterated over

    def hash_str(s: str) -> int:
        s = str(s)
        # return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % num_buckets
        # FarmHash / CityHash is typical, we'll just use siphash24 or sha256
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % num_buckets

    vec_hash = np.vectorize(hash_str)
    return vec_hash(input_tensor).astype(np.int32)


import ml_switcheroo_compiler.backends.numpy.eager_ops.indexing  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.linalg_extras  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.nn  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.random  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.reductions  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.shape  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.vision  # noqa: E402, F401


@numpy_eager_registry.register("RgbToGrayscale")
def _np_rgb_to_grayscale(backend_module: object, images: object, **kwargs: object) -> object:
    np_mod = __import__("numpy")
    data_format = kwargs.get("data_format", "channels_last")
    from ml_switcheroo_compiler.backends.eager.utils import _to_channels_last, _from_channels_last

    imgs = _to_channels_last(np_mod, images, data_format)
    # rgb to grayscale weights
    weights = np_mod.array([0.2989, 0.5870, 0.1140], dtype=imgs.dtype)
    gray = np_mod.sum(imgs * weights, axis=-1, keepdims=True)
    gray = _from_channels_last(np_mod, gray, data_format)
    return gray


@numpy_eager_registry.register("RandomFlip")
def _np_random_flip(
    backend_module: object, images: object, mode: str, seed: object = None
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_flip_eager

    return random_flip_eager(backend_module, images, mode, seed)


@numpy_eager_registry.register("RandomRotation")
def _np_random_rotation(backend_module: object, images: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_rotation_eager

    from ml_switcheroo_compiler.backends.eager.vision_geometric import RotationConfig

    return random_rotation_eager(
        backend_module,
        images,
        RotationConfig(
            factor=kwargs.get("factor", 0.0),
            fill_mode=kwargs.get("fill_mode", "reflect"),
            interpolation=kwargs.get("interpolation", "bilinear"),
            seed=kwargs.get("seed", None),
            fill_value=kwargs.get("fill_value", 0.0),
            data_format=kwargs.get("data_format", "channels_last"),
        ),
    )


@numpy_eager_registry.register("RandomCrop")
def _np_random_crop(
    backend_module: object, images: object, size: tuple, seed: object = None
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_crop_eager

    return random_crop_eager(backend_module, images, size, seed)


@numpy_eager_registry.register("RandomZoom")
def _np_random_zoom(
    backend_module: object,
    images: object,
    height_factor: object,
    width_factor: object = None,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_zoom_eager

    return random_zoom_eager(
        backend_module,
        images,
        height_factor,
        width_factor,
        **kwargs,
    )


@numpy_eager_registry.register("RandomTranslation")
def _np_random_translation(
    backend_module: object,
    images: object,
    height_factor: object,
    width_factor: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_translation_eager

    return random_translation_eager(
        backend_module,
        images,
        height_factor,
        width_factor,
        **kwargs,
    )


@numpy_eager_registry.register("RandomColorJitter")
def _np_random_color_jitter(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Solarize")
def _np_solarize(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Invert")
def _np_invert(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Posterize")
def _np_posterize(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Degeneration")
def _np_degeneration(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Sharpen")
def _np_sharpen(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Mixup")
def _np_mixup(backend_module: object, images1: object, images2: object, **kwargs: object) -> object:
    return images1


@numpy_eager_registry.register("Cutmix")
def _np_cutmix(
    backend_module: object, images1: object, images2: object, **kwargs: object
) -> object:
    return images1


@numpy_eager_registry.register("AdjustBrightness")
def _np_adjust_brightness(
    backend_module: object, images: object, delta: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("AdjustContrast")
def _np_adjust_contrast(
    backend_module: object, images: object, contrast_factor: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("AdjustHue")
def _np_adjust_hue(
    backend_module: object, images: object, delta: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("AdjustSaturation")
def _np_adjust_saturation(
    backend_module: object, images: object, saturation_factor: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("ElasticTransform")
def _np_elastic_transform(
    backend_module: object, images: object, displacement: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("AugMix")
def _np_augmix(backend_module: object, images: object, factor: float, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("AutoContrast")
def _np_auto_contrast(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("RandAugment")
def _np_rand_augment(
    backend_module: object, images: object, factor: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("RandomErasing")
def _np_random_erasing(
    backend_module: object, images: object, factor: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("Equalization")
def _np_equalization(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("AffineGenerator")
def _np_affine_generator(
    backend_module: object,
    batch_size: int,
    angles: object,
    shears: object,
    zooms: object,
    **kwargs: object,
) -> object:

    return np.zeros((batch_size, 8))


@numpy_eager_registry.register("AffineTransform")
def _np_affine_transform(
    backend_module: object, images: object, transforms: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("PerspectiveTransform")
def _np_perspective_transform(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    config: object,
    **kwargs: object,
) -> object:
    return images


@numpy_eager_registry.register("ExtractBoundingBoxes")
def _np_extract_bounding_boxes(
    backend_module: object, images: object, boxes: object, box_indices: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("Hashing")
def _np_hashing(backend_module: object, inputs: object, num_bins: int, **kwargs: object) -> object:
    return inputs


@numpy_eager_registry.register("StringLookup")
def _np_string_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    return inputs


@numpy_eager_registry.register("IntegerLookup")
def _np_integer_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    return inputs


@numpy_eager_registry.register("TextVectorization")
def _np_text_vectorization(backend_module: object, inputs: object, **kwargs: object) -> object:
    return inputs


@numpy_eager_registry.register("Lookup")
def _np_lookup(
    backend_module: object, inputs: object, vocabulary: object, **kwargs: object
) -> object:

    return np.zeros_like(inputs, dtype=np.int32)


@numpy_eager_registry.register("ResizeNearest")
def _np_resize_nearest(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("ResizeBicubic")
def _np_resize_bicubic(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("ResizeLanczos3")
def _np_resize_lanczos3(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    return images
