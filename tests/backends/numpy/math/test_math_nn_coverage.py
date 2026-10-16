"""Tests for test_math_nn_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_nn as math_nn_module

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_math_nn_ops() -> None:
    """Test neural network convolution operations directly and through eager registry.

    Returns:
        None
    """
    sig1 = np.array([1.0, 2.0, 3.0])
    sig2 = np.array([0.5, 1.0])

    # Convolve
    conv_res = math_nn_module._np_convolve(np, sig1, sig2)
    assert cast("Callable[..., EagerTensorProtocol]", len)(conv_res) == 4
    reg_conv = numpy_eager_registry.get("Convolve")
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", reg_conv)(np, sig1, sig2)) == 4

    # ConvGeneralDilatedLocal
    dil_local = math_nn_module._np_convgeneraldilatedlocal(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])
    assert cast("Callable[..., EagerTensorProtocol]", len)(dil_local) == 3
    reg_dil_local = numpy_eager_registry.get("ConvGeneralDilatedLocal")
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", reg_dil_local)(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])) == 3

    # ConvGeneralDilatedPatches
    dil_patches = math_nn_module._np_convgeneraldilatedpatches(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])
    assert cast("Callable[..., EagerTensorProtocol]", len)(dil_patches) == 3
    reg_dil_patches = numpy_eager_registry.get("ConvGeneralDilatedPatches")
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", reg_dil_patches)(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])) == 3

    # ConvWithGeneralPadding
    gen_pad = math_nn_module._np_convwithgeneralpadding(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])
    assert cast("Callable[..., EagerTensorProtocol]", len)(gen_pad) == 3
    reg_gen_pad = numpy_eager_registry.get("ConvWithGeneralPadding")
    assert cast("Callable[..., EagerTensorProtocol]", len)(cast("Callable[..., EagerTensorProtocol]", reg_gen_pad)(np, [1.0, 2.0, 3.0, 4.0], [1.0, 1.0])) == 3

    # RawConv2D
    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    filt = np.array([[1.0, 0.0], [0.0, 1.0]])
    raw_conv = math_nn_module._np_rawconv2d(np, img, filt)
    assert cast("EagerTensorProtocol", getattr(raw_conv, "shape")) == (1, 1)
    reg_raw_conv = numpy_eager_registry.get("RawConv2D")
    assert cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", reg_raw_conv)(np, img, filt), "shape")) == (1, 1)
