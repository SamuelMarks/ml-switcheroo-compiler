"""Test alpha dropout and angle."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def test_alpha_dropout_numpy_eager() -> object:
    """Function docstring."""
    AlphaDropout = numpy_eager_registry.get("AlphaDropout")
    t = np.array([1.0, 2.0, 3.0, 4.0])

    # Not training (should return input)
    out1 = AlphaDropout(np, t, rate=0.5)
    np.testing.assert_allclose(out1, t)

    # Training but rate=0
    out0 = AlphaDropout(np, t, rate=0.0, training=True)
    np.testing.assert_allclose(out0, t)

    # Training, rate=0.5
    out2 = AlphaDropout(np, t, rate=0.5, training=True, seed=42)
    assert not np.allclose(out2, t)

    # Training, custom noise shape
    out3 = AlphaDropout(np, t, rate=0.5, training=True, seed=42, noise_shape=(4,))
    assert not np.allclose(out3, t)

    # Activity Regularization dummy coverage
    ActivityRegularization = numpy_eager_registry.get("ActivityRegularization")
    out_act = ActivityRegularization(np, t)
    np.testing.assert_allclose(out_act, t)


def test_angle_numpy_eager() -> object:
    """Function docstring."""
    Angle = numpy_eager_registry.get("Angle")
    t = np.array([1.0 + 1.0j, 1.0 - 1.0j])
    out = Angle(np, t)
    np.testing.assert_allclose(out, np.array([np.pi / 4, -np.pi / 4]))
