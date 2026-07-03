"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.conv import _conv_general_dilated
from ml_switcheroo_compiler.ops.configs import ConvConfig


class DummyDimSpecs:
    """Class docstring."""

    def __init__(self) -> object:
        """Function docstring."""
        self.lhs_spec = (0, 1, 2)
        self.rhs_spec = (0, 1, 2)
        self.out_spec = (0, 1, 2)


def test_conv_general_dilated_extra() -> object:
    """Function docstring."""
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))

    # Cover None padding
    res = _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding=None))
    assert res is not None

    # Cover custom dimension numbers
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=DummyDimSpecs()),
    )
    assert res is not None

    # Cover dilation
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="VALID", lhs_dilation=[2], rhs_dilation=[2]),
    )
    assert res is not None

    # Cover feature group count > 1
    lhs = np.ones((1, 4, 5))
    rhs = np.ones((4, 2, 3))
    res = _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", feature_group_count=2))
    assert res is not None
