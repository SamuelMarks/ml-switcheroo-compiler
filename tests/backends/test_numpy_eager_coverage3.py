import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.reductions import _reduce_window
from ml_switcheroo_compiler.backends.numpy.eager.conv import _conv_general_dilated
from ml_switcheroo_compiler.ops.configs import ConvConfig, WindowConfig


def test_reduce_window_coverage():
    operand = np.array([1, 2, 3])

    # base_dilation coverage
    res = _reduce_window(
        operand,
        0,
        "sum",
        WindowConfig(
            window_dimensions=(2,),
            window_strides=(1,),
            base_dilation=(2,),
            window_dilation=(1,),
            padding=[(0, 0)],
        ),
    )
    assert res is not None

    # prod and min coverage
    res = _reduce_window(
        operand,
        0,
        "prod",
        WindowConfig(
            window_dimensions=(2,),
            window_strides=(1,),
            base_dilation=(1,),
            window_dilation=(1,),
            padding=[(0, 0)],
        ),
    )
    assert res is not None

    res = _reduce_window(
        operand,
        0,
        "min",
        WindowConfig(
            window_dimensions=(2,),
            window_strides=(1,),
            base_dilation=(1,),
            window_dilation=(1,),
            padding=[(0, 0)],
        ),
    )
    assert res is not None

    # Exception coverage
    try:
        _reduce_window(
            operand,
            0,
            "unknown",
            WindowConfig(
                window_dimensions=(2,),
                window_strides=(1,),
                base_dilation=(1,),
                window_dilation=(1,),
                padding=[(0, 0)],
            ),
        )
    except ValueError:
        pass


def test_conv_general_dilated_coverage():
    lhs = np.ones((1, 2, 5))
    rhs = np.ones((4, 2, 3))

    # dimension_numbers as tuple of strings
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="SAME", dimension_numbers=("NCW", "OIW", "NCW")),
    )
    assert res is not None
    assert res.shape == (1, 4, 5)

    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="VALID", dimension_numbers=("NCW", "OIW", "NCW")),
    )
    assert res is not None
    assert res.shape == (1, 4, 3)

    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(window_strides=(1,), padding="UNKNOWN", dimension_numbers=("NCW", "OIW", "NCW")),
    )
    assert res is not None

    # lhs_dilation and rhs_dilation
    res = _conv_general_dilated(
        lhs,
        rhs,
        ConvConfig(
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCW", "OIW", "NCW"),
            lhs_dilation=(2,),
            rhs_dilation=(2,),
        ),
    )
    assert res is not None

    # feature_group_count
    lhs_group = np.ones((1, 4, 5))
    rhs_group = np.ones((4, 2, 3))  # OI... format, out=4, in=2 (total in=4/group=2) -> groups=2
    res = _conv_general_dilated(
        lhs_group,
        rhs_group,
        ConvConfig(
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCW", "OIW", "NCW"),
            feature_group_count=2,
        ),
    )
    assert res is not None

    # 2D case to cover NCHW defaults
    lhs2 = np.ones((1, 2, 5, 5))
    rhs2 = np.ones((4, 2, 3, 3))

    class DimensionNumbers:
        lhs_spec = (0, 1, 2, 3)
        rhs_spec = (0, 1, 2, 3)
        out_spec = (0, 1, 2, 3)

    res = _conv_general_dilated(
        lhs2,
        rhs2,
        ConvConfig(window_strides=(1, 1), padding="SAME", dimension_numbers=DimensionNumbers()),
    )
    assert res is not None


def test_band_part_coverage():
    from ml_switcheroo_compiler.backends.numpy.eager.math_extras import _band_part

    res = _band_part(np.ones((3, 3)), 1, 1)
    assert res is not None
    assert res.shape == (3, 3)
