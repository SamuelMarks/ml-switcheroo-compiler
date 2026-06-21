"""Vision operations for the numpy backend."""

import scipy.ndimage
from ml_switcheroo_compiler.backends.numpy.eager import numpy_eager_registry


@numpy_eager_registry.register("ResizeBilinear")
def resize_bilinear(
    np_mod: object, images: object, size: tuple[int, int], align_corners: bool = False
) -> object:
    """Resize images using bilinear interpolation.

    Args:
        np_mod: Numpy module.
        images: The images to resize.
        size: The target size (height, width).
        align_corners: Whether to align corners.

    Returns:
        The resized images.
    """
    h_factor = size[0] / images.shape[1]
    w_factor = size[1] / images.shape[2]
    return scipy.ndimage.zoom(images, (1, h_factor, w_factor, 1), order=1)


@numpy_eager_registry.register("ResizeNearest")
def resize_nearest(
    np_mod: object, images: object, size: tuple[int, int], align_corners: bool = False
) -> object:
    """Resize images using nearest-neighbor interpolation.

    Args:
        np_mod: Numpy module.
        images: The images to resize.
        size: The target size (height, width).
        align_corners: Whether to align corners.

    Returns:
        The resized images.
    """
    h_factor = size[0] / images.shape[1]
    w_factor = size[1] / images.shape[2]
    return scipy.ndimage.zoom(images, (1, h_factor, w_factor, 1), order=0)


@numpy_eager_registry.register("ResizeBicubic")
def resize_bicubic(
    np_mod: object, images: object, size: tuple[int, int], align_corners: bool = False
) -> object:
    """Resize images using bicubic interpolation.

    Args:
        np_mod: Numpy module.
        images: The images to resize.
        size: The target size (height, width).
        align_corners: Whether to align corners.

    Returns:
        The resized images.
    """
    h_factor = size[0] / images.shape[1]
    w_factor = size[1] / images.shape[2]
    return scipy.ndimage.zoom(images, (1, h_factor, w_factor, 1), order=3)


@numpy_eager_registry.register("ResizeLanczos3")
def resize_lanczos3(
    np_mod: object, images: object, size: tuple[int, int], align_corners: bool = False
) -> object:
    """Resize images using Lanczos3 interpolation.

    Args:
        np_mod: Numpy module.
        images: The images to resize.
        size: The target size (height, width).
        align_corners: Whether to align corners.

    Returns:
        The resized images.
    """
    h_factor = size[0] / images.shape[1]
    w_factor = size[1] / images.shape[2]
    return scipy.ndimage.zoom(images, (1, h_factor, w_factor, 1), order=3)
