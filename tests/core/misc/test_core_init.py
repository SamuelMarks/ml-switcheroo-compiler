"""Test module."""

from ml_switcheroo_compiler.core import backend, get_uid, image_data_format


def test_core_init():
    assert image_data_format() == "channels_last"
    assert backend() == "numpy"
    uid1 = get_uid("test")
    uid2 = get_uid("test")
    assert uid2 == uid1 + 1
