"""Tests for test_pickle."""

from __future__ import annotations

# ruff: noqa: E501
import tempfile
import typing

from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat

if typing.TYPE_CHECKING:
    pass


def test_pickle_load_save() -> None:
    """Test function test_pickle_load_save."""
    fmt = PickleWeightFormat()
    with tempfile.NamedTemporaryFile() as f:
        fmt.save({"a": 1}, f.name)
        assert fmt.load(f.name) == {"a": 1}
