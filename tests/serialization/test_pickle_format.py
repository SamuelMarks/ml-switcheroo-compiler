"""Tests for test_pickle_format."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_pickle_format() -> None:
    """Test function test_pickle_format."""
    fmt = PickleWeightFormat()
    data = {"a": 1, "b": "test"}
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights.pkl")
        cast("Callable[..., EagerTensorProtocol]", fmt.save)(data, filepath)
        loaded = fmt.load(filepath)
        assert loaded == data
