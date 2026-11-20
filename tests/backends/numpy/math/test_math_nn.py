"""Behavioral unit tests for math nn."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.nn_ops import layers as math_nn_module
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import scipy_ndimage_kernels as nd_kernels
from tests.testing_utils import require_callable

if typing.TYPE_CHECKING:
    pass


def test_math_nn_ops() -> None:
    """Test neural network convolution operations directly and through eager registry.

    Returns:
        None
    """
    sig1 = np.array([1.0, 2.0, 3.0])
    sig2 = np.array([0.5, 1.0])

    # Convolve
    conv_res = math_nn_module._np_convolve(np, sig1, sig2)
    assert len(conv_res) == 4
    reg_conv = numpy_eager_registry.get("Convolve")
    assert len(require_callable(reg_conv)(np, sig1, sig2)) == 4

    # ConvGeneralDilatedLocal
    dil_local = nd_kernels._np_convgeneraldilatedlocal(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])
    assert len(dil_local) == 3
    reg_dil_local = numpy_eager_registry.get("ConvGeneralDilatedLocal")
    assert len(require_callable(reg_dil_local)(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])) == 3

    # ConvGeneralDilatedPatches
    dil_patches = nd_kernels._np_convgeneraldilatedpatches(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])
    assert len(dil_patches) == 3
    reg_dil_patches = numpy_eager_registry.get("ConvGeneralDilatedPatches")
    assert len(require_callable(reg_dil_patches)(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])) == 3

    # ConvWithGeneralPadding
    gen_pad = nd_kernels._np_convwithgeneralpadding(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])
    assert len(gen_pad) == 3
    reg_gen_pad = numpy_eager_registry.get("ConvWithGeneralPadding")
    assert len(require_callable(reg_gen_pad)(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])) == 3

    # RawConv2D
    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    filt = np.array([[1.0, 0.0], [0.0, 1.0]])
    raw_conv = nd_kernels._np_rawconv2d(np, img, filt)
    assert raw_conv.shape == (1, 1)
    reg_raw_conv = numpy_eager_registry.get("RawConv2D")
    assert require_callable(reg_raw_conv)(np, img, filt).shape == (1, 1)
