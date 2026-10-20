"""Tests for test_pickle."""

from __future__ import annotations

# ruff: noqa: E501
import tempfile
import typing

from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_pickle_load_save() -> None:
    """Test function test_pickle_load_save."""
    fmt = PickleWeightFormat()
    with tempfile.NamedTemporaryFile() as f:
        fmt.save({"a": 1}, f.name)
        assert typing.cast("EagerTensorProtocol", fmt.load(f.name)) == typing.cast("EagerTensorProtocol", {"a": 1})
