"""Test module."""

from ml_switcheroo_compiler.backends.common.mixins.common import CommonASTVisitor


def test_common_ast_visitor():
    vis = CommonASTVisitor()
    assert vis.generator is vis
    assert vis._get_backend_prefix() == ""

    class DummyGenerator:
        pass

    gen = DummyGenerator()
    vis2 = CommonASTVisitor(generator=gen)
    assert vis2.generator is gen
