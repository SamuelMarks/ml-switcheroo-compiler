from ml_switcheroo_compiler.transforms.passes.spmd import (
    _create_all_gather_node,
    _create_reduce_scatter_node,
    _is_boundary_transition,
)


class MockSharding:
    def __init__(self, mesh_mapping):
        self.mesh_mapping = mesh_mapping


def test_is_boundary_transition():
    s1 = MockSharding([None, "x"])
    s2 = MockSharding([None, None])
    inp_sharded, node_sharded = _is_boundary_transition(s1, s2)
    assert inp_sharded is True
    assert node_sharded is False


def test_create_all_gather_node():
    s = MockSharding([None, None])
    node = _create_all_gather_node("test_inp", s)
    assert node.id == "test_inp_all_gather"
    assert node.op_type == "all_gather"
    assert node.inputs == ["test_inp"]
    assert node.sharding is s


def test_create_reduce_scatter_node():
    s = MockSharding([None, "x"])
    node = _create_reduce_scatter_node("test_inp", s)
    assert node.id == "test_inp_reduce_scatter"
    assert node.op_type == "reduce_scatter"
    assert node.inputs == ["test_inp"]
    assert node.sharding is s
