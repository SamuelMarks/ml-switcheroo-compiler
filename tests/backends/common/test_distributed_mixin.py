"""Test module."""

from ml_switcheroo_compiler.backends.common.mixins.distributed import DistributedASTVisitor


class DummyGenerator:
    def _get_backend_prefix(self):
        return "bk"


class DummyVisitor(DistributedASTVisitor):
    def __init__(self):
        self._generator = DummyGenerator()


class DummyNode:
    pass


def test_distributed_mixin():
    vis = DummyVisitor()
    node = DummyNode()

    assert vis.visit_AllGather(node, ["a"], axis=1) == "bk_all_gather(a, axis=1)"
    assert vis.visit_AllGather(node, ["a"]) == "bk_all_gather(a, axis=0)"

    assert vis.visit_AllReduce(node, ["a"], op="mean") == "bk_all_reduce(a, op='mean')"
    assert vis.visit_AllReduce(node, ["a"]) == "bk_all_reduce(a, op='sum')"

    assert vis.visit_AllToAll(node, ["a"], split_axis=1, concat_axis=2, axis_name="p") == "bk_all_to_all(a, split_axis=1, concat_axis=2, axis_name='p')"
    assert vis.visit_AllToAll(node, ["a"]) == "bk_all_to_all(a, split_axis=0, concat_axis=0, axis_name='')"
