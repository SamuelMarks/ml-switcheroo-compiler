"""Module docstring."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.conv import _conv_general_dilated, _get_transpose
from ml_switcheroo_compiler.ops.configs import ConvConfig


class DummyDimSpecs:
    """Class docstring."""

    def __init__(self) -> object:
        """Function docstring."""
        self.lhs_spec = (0, 1, 2)
        self.rhs_spec = (0, 1, 2)
        self.out_spec = (0, 1, 2)


def test_conv_general_dilated_all_coverage() -> object:
    """Function docstring."""
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))

    # _get_transpose error
    with pytest.raises(TypeError):
        _get_transpose(123, "OIW")

    # None padding
    _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding=None, dimension_numbers=("NCW", "OIW", "NCW")),
    )

    # custom dimension numbers
    _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=DummyDimSpecs()),
    )

    # feature_group_count > 1
    lhs2 = np.ones((1, 4, 5))
    rhs2 = np.ones((4, 2, 3))
    _conv_general_dilated(
        lhs2,
        rhs2,
        ConvConfig(
            window_strides=(1,),
            padding="VALID",
            feature_group_count=2,
            dimension_numbers=("NCW", "OIW", "NCW"),
        ),
    )

    # dimension_numbers None
    _conv_general_dilated(lhs, rhs, ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=None))
