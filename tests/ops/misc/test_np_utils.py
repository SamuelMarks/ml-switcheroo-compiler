"""Tests for test_np_utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.utils.np_utils import normalize, to_categorical

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_np_utils(mocker: EagerTensorProtocol) -> None:
    """Test function test_np_utils.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.ops.max", return_value=1)
    mocker.patch("ml_switcheroo_compiler.ops.cast", return_value="cast")
    mocker.patch("ml_switcheroo_compiler.ops.arange", return_value="arange")
    mocker.patch("ml_switcheroo_compiler.ops.expand_dims", return_value="expand")
    mocker.patch("ml_switcheroo_compiler.ops.equal", return_value="equal")
    assert to_categorical("x") == "cast"
    assert to_categorical("x", num_classes=5) == "cast"
    mocker.patch("ml_switcheroo_compiler.ops.sum", return_value="sum")
    mocker.patch("ml_switcheroo_compiler.ops.square", return_value="square")
    mocker.patch("ml_switcheroo_compiler.ops.sqrt", return_value="sqrt")
    mocker.patch("ml_switcheroo_compiler.ops.maximum", return_value="max")
    mocker.patch("ml_switcheroo_compiler.ops.divide", return_value="div")
    assert normalize("x") == "div"


def test_to_categorical_item(mocker: EagerTensorProtocol) -> None:
    """Test function test_to_categorical_item.

    Args:
        mocker: Test parameter `mocker`.
    """

    class M:
        """Test class M."""

        def item(self) -> EagerValue:
            """Test function item."""
            return 2

    cast("EagerCallableProtocol", mocker.patch)("ml_switcheroo_compiler.ops.max", return_value=cast("EagerCallableProtocol", M)())
    mocker.patch("ml_switcheroo_compiler.ops.cast", return_value="cast")
    mocker.patch("ml_switcheroo_compiler.ops.arange")
    mocker.patch("ml_switcheroo_compiler.ops.expand_dims")
    mocker.patch("ml_switcheroo_compiler.ops.equal")
    assert to_categorical("x") == "cast"
