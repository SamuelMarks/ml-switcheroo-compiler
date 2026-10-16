"""Tests for test_pickle."""

from __future__ import annotations

# ruff: noqa: E501
import tempfile
from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_pickle_load_save() -> None:
    """Test function test_pickle_load_save."""
    fmt = PickleWeightFormat()
    with tempfile.NamedTemporaryFile() as f:
        fmt.save({"a": 1}, f.name)
        assert cast("EagerTensorProtocol", fmt.load(f.name)) == cast("EagerTensorProtocol", {"a": 1})
