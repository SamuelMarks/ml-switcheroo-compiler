"""Test associative bessel."""

import numpy as np
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def test_new_ops_part1():
    """Test first part."""
    config.eager_mode = True

    # control flow
    AssociativeScan = numpy_eager_registry.get("AssociativeScan")
    if AssociativeScan:
        out = AssociativeScan(np, np.array([1, 2]))
        np.testing.assert_array_equal(out, np.array(0))

    # vision color
    AugMix = numpy_eager_registry.get("AugMix")
    if AugMix:
        out = AugMix(np, np.ones((1, 3, 3, 3)), factor=1.0)
        np.testing.assert_array_equal(out, np.ones((1, 3, 3, 3)))

    AutoContrast = numpy_eager_registry.get("AutoContrast")
    if AutoContrast:
        out = AutoContrast(np, np.ones((1, 3, 3, 3)))
        assert out.shape == (1, 3, 3, 3)

    # distributed
    AxisIndex = numpy_eager_registry.get("AxisIndex")
    if AxisIndex:
        out = AxisIndex(np, axis_name="batch")
        assert out == 0

    # random
    Ball = numpy_eager_registry.get("Ball")
    if Ball:
        out = Ball(np, key=0, d=2)
        assert out is not None

    Beta = numpy_eager_registry.get("Beta")
    if Beta:
        out = Beta(np, None, np.array([0.5]), np.array([0.5]))
        assert len(out) == 1


def test_new_ops_part2():
    """Test second part."""
    config.eager_mode = True

    # linalg
    BandPart = numpy_eager_registry.get("BandPart")
    if BandPart:
        t = np.ones((3, 3))
        out = BandPart(np, t, num_lower=1, num_upper=1)
        np.testing.assert_array_equal(out, t)

    BandedTriangularSolve = numpy_eager_registry.get("BandedTriangularSolve")
    if BandedTriangularSolve:
        out = BandedTriangularSolve(np, np.ones((1, 1)), np.ones((1, 1)))

    # math extras
    Bartlett = numpy_eager_registry.get("Bartlett")
    if Bartlett:
        out = Bartlett(np, 3)
        assert len(out) == 3

    # math special
    special_ops = [
        "BesselI0",
        "BesselI0e",
        "BesselI1",
        "BesselI1e",
        "BesselJ0",
        "BesselJ1",
        "BesselK0",
        "BesselK0e",
        "BesselK1",
        "BesselK1e",
        "BesselY0",
        "BesselY1",
    ]

    for op in special_ops:
        fn = numpy_eager_registry.get(op)
        if fn:
            out = fn(np, np.array([1.0, 2.0]))
            assert len(out) == 2

    BesselJn = numpy_eager_registry.get("BesselJn")
    if BesselJn:
        out = BesselJn(np, np.array([2.0]), np.array([1.0]))
        assert len(out) == 1

    Betainc = numpy_eager_registry.get("Betainc")
    if Betainc:
        out = Betainc(np, np.array([0.5]), np.array([1.0]), np.array([1.0]))
        assert len(out) == 1
