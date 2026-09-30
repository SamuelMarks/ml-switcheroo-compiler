"""Test core package initialization."""

from __future__ import annotations

from ml_switcheroo_compiler.core import backend, get_uid, image_data_format


def test_core_init() -> None:
    """Test core package initialization functions.

    Returns:
        None.
    """
    assert image_data_format() == "channels_last"
    assert backend() == "numpy"
    uid1 = get_uid("test")
    uid2 = get_uid("test")
    assert uid2 == uid1 + 1
    uid_def1 = get_uid()
    uid_def2 = get_uid()
    assert uid_def2 == uid_def1 + 1
