"""Tests for test_pickle."""

from __future__ import annotations

# ruff: noqa: E501
import tempfile
import typing

from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_pickle_load_save() -> None:
    """Test function test_pickle_load_save."""
    fmt = PickleWeightFormat()
    with tempfile.NamedTemporaryFile() as f:
        fmt.save({"a": 1}, f.name)
        assert as_tensor(fmt.load(f.name)) == as_tensor({"a": 1})
