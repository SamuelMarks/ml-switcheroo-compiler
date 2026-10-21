"""Tests for test_pickle_format."""

from __future__ import annotations

import os
import tempfile
import typing

from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_pickle_format() -> None:
    """Test function test_pickle_format."""
    fmt = PickleWeightFormat()
    data = {"a": 1, "b": "test"}
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights.pkl")
        as_callable(fmt.save)(data, filepath)
        loaded = fmt.load(filepath)
        assert loaded == data
