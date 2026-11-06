"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.common.mixins.common import CommonASTVisitor
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_common_ast_visitor() -> None:
    """Test function test_common_ast_visitor."""
    vis = CommonASTVisitor()
    assert ensure_eager_tensor(vis.generator) is ensure_eager_tensor(vis)
    assert vis.get_fallback_prefix() == ""

    class DummyGenerator:
        """Test class DummyGenerator."""

        pass

    gen = DummyGenerator()
    vis2 = ensure_callable(CommonASTVisitor)(generator=gen)
    assert id(vis2.generator) == id(gen)
