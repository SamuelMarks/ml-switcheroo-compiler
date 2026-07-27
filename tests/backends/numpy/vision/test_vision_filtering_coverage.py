import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import _np_gaussian_blur
from ml_switcheroo_compiler.ops.vision.filtering import BlurConfig


def test_gaussian_blur_cov():
    config = BlurConfig(kernel_size=(3, 3), sigma=(1.0, 1.0))
    _np_gaussian_blur(np, np.ones((1, 3, 3, 1)), config=config)
