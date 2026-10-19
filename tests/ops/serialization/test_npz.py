"""Tests for test_npz."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_npz_load(mocker: EagerTensorProtocol) -> None:
    """Test function test_npz_load.

    Args:
        mocker: Test parameter `mocker`.
    """
    fmt = NpzWeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    setattr(mock_backend.load_npz, "return_value", {"a": 1})
    assert cast("EagerTensorProtocol", fmt.load("test.npz")) == cast("EagerTensorProtocol", {"a": 1})
    setattr(mock_backend.load_npz, "side_effect", NotImplementedError)
    mocker.patch("ml_switcheroo_compiler.serialization.formats.npz.parse_npz", return_value={"b": 2})

    with pytest.warns(UserWarning, match="Backend load_npz failed"):
        assert cast("EagerTensorProtocol", fmt.load("test.npz")) == cast("EagerTensorProtocol", {"b": 2})

    del mock_backend.load_npz
    assert cast("EagerTensorProtocol", fmt.load("test.npz")) == cast("EagerTensorProtocol", {"b": 2})
