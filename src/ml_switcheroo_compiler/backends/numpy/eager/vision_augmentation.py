"""Shared vision utilities and ops."""

from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    RotationConfig,
    random_crop_eager,
    random_flip_eager,
    random_rotation_eager,
    random_translation_eager,
    random_zoom_eager,
)  # pragma: no cover
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AugMix")
def _np_augmix(backend_module: object, images: object, factor: float, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        factor: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("Cutmix")
def _np_cutmix(backend_module: object, images1: object, images2: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images1: Arg.
        images2: Arg.
        kwargs: Arg.
    """
    return images1


@numpy_eager_registry.register("Mixup")
def _np_mixup(backend_module: object, images1: object, images2: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images1: Arg.
        images2: Arg.
        kwargs: Arg.
    """
    return images1


@numpy_eager_registry.register("RandAugment")
def _np_rand_augment(backend_module: object, images: object, factor: float, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        factor: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("RandomColorJitter")
def _np_random_color_jitter(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("RandomCrop")
def _np_random_crop(backend_module: object, images: object, size: tuple, seed: object = None) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        size: Arg.
        seed: Arg.
    """
    return random_crop_eager(backend_module, images, size, seed)


@numpy_eager_registry.register("RandomErasing")
def _np_random_erasing(backend_module: object, images: object, factor: float, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        factor: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("RandomFlip")
def _np_random_flip(backend_module: object, images: object, mode: str, seed: object = None) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        mode: Arg.
        seed: Arg.
    """
    return random_flip_eager(backend_module, images, mode, seed)


@numpy_eager_registry.register("RandomRotation")
def _np_random_rotation(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
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


@numpy_eager_registry.register("RandomTranslation")
def _np_random_translation(
    backend_module: object,
    images: object,
    height_factor: object,
    width_factor: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        height_factor: Arg.
        width_factor: Arg.
        kwargs: Arg.
    """
    return random_translation_eager(  # pragma: no cover
        backend_module,
        images,
        height_factor,
        width_factor,
        **kwargs,
    )


@numpy_eager_registry.register("RandomZoom")
def _np_random_zoom(
    backend_module: object,
    images: object,
    height_factor: object,
    width_factor: object = None,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        height_factor: Arg.
        width_factor: Arg.
        kwargs: Arg.
    """
    return random_zoom_eager(
        backend_module,
        images,
        height_factor,
        width_factor,
        **kwargs,
    )


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_np_augmix",
    "_np_cutmix",
    "_np_mixup",
    "_np_rand_augment",
    "_np_random_color_jitter",
    "_np_random_crop",
    "_np_random_erasing",
    "_np_random_flip",
    "_np_random_rotation",
    "_np_random_translation",
    "_np_random_zoom",
    "numpy_eager_registry",
]
