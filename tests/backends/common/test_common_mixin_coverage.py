"""Tests for test_common_mixin_coverage."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

from ml_switcheroo_compiler.backends.common.mixins.common import CommonASTVisitor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_common_ast_visitor_mixin() -> None:
    """Verify CommonASTVisitor delegation and fallback prefix."""
    visitor_self = CommonASTVisitor()
    assert typing.cast("EagerTensorProtocol", visitor_self.generator) is typing.cast("EagerTensorProtocol", visitor_self)
    assert visitor_self.get_fallback_prefix() == ""

    mock_gen = MagicMock()
    visitor_gen = CommonASTVisitor(generator=mock_gen)
    assert visitor_gen.generator is mock_gen
