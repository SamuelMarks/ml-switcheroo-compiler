"""Test module."""

from ml_switcheroo_compiler.backends.numpy.types import array, asarray, item, zeros


def test_numpy_types():
    assert zeros(None, (1,)).shape == (1,)

    assert array(None, [1]).tolist() == [1]

    assert asarray(None, [1]).tolist() == [1]

    assert item(None, [42]) == 42.0
