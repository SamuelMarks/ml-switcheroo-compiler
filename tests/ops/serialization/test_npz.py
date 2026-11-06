"""Tests for test_npz."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_npz_load(mocker: EagerTensorProtocol) -> None:
    """Test function test_npz_load.

    Args:
        mocker: Test parameter `mocker`.
    """
    fmt = NpzWeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    ensure_mutable(mock_backend.load_npz).return_value = ensure_eager_tensor({"a": 1})
    assert ensure_eager_tensor(fmt.load("test.npz")) == ensure_eager_tensor({"a": 1})
    ensure_mutable(mock_backend.load_npz).side_effect = ensure_eager_tensor(NotImplementedError)
    mocker.patch("ml_switcheroo_compiler.serialization.formats.npz.parse_npz", return_value={"b": 2})

    with pytest.warns(UserWarning, match="Backend load_npz failed"):
        assert ensure_eager_tensor(fmt.load("test.npz")) == ensure_eager_tensor({"b": 2})

    del mock_backend.load_npz
    assert ensure_eager_tensor(fmt.load("test.npz")) == ensure_eager_tensor({"b": 2})
