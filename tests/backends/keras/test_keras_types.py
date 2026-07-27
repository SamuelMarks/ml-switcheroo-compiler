"""Test module."""

from unittest.mock import patch

from ml_switcheroo_compiler.backends.keras.types import array, asarray, item, zeros


def test_keras_types():
    with patch("ml_switcheroo_compiler.backends.keras.types.kops"):
        with patch("ml_switcheroo_compiler.backends.keras.types.generic_zeros", return_value="zeros"):
            assert zeros(None, (2,)) == "zeros"
        with patch("ml_switcheroo_compiler.backends.keras.types.generic_array", return_value="array"):
            assert array(None, [1]) == "array"
        with patch("ml_switcheroo_compiler.backends.keras.types.generic_asarray", return_value="asarray"):
            assert asarray(None, [1]) == "asarray"
        with patch("ml_switcheroo_compiler.backends.keras.types.generic_item", return_value=42.0):
            assert item(None, [1]) == 42.0
