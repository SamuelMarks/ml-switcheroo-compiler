from types import SimpleNamespace

from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import _JVP_REGISTRY
from ml_switcheroo_compiler.transforms.autodiff_rules.reduction_rules import (
    accumulate_n_jvp,
    accumulate_n_vjp,
    add_n_jvp,
    add_n_vjp,
    all_gather_jvp,
    all_gather_vjp,
    allreduce_jvp,
    allreduce_vjp,
    average_jvp,
    average_vjp,
    broadcast_jvp,
    broadcast_vjp,
    cumulative_logsumexp_jvp,
    cumulative_logsumexp_vjp,
    logsumexp_jvp,
    logsumexp_vjp,
    max_jvp,
    max_vjp,
    mean_jvp,
    mean_vjp,
    min_jvp,
    min_vjp,
    reduce_euclidean_norm_jvp,
    reduce_euclidean_norm_vjp,
    reduce_scatter_jvp,
    reduce_scatter_vjp,
    shard_tensor_jvp,
    shard_tensor_vjp,
    sum_jvp,
    sum_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY


class MockGraph:
    def __init__(self, node_shapes):
        self.nodes = {k: SimpleNamespace(shape_metadata=v) for k, v in node_shapes.items()}
        self.nodes.update({"mock_op_id": SimpleNamespace(shape_metadata=(2, 2))})
        self._next_id = 0

    def add_node(self, node):
        node.id = f"mock_{self._next_id}"
        self._next_id += 1
        self.nodes[node.id] = node
        return node.id


def test_autodiff_reduction_rules_coverage():
    graph = MockGraph({"x": (2, 2), "y": (2, 2)})
    node = SimpleNamespace(id="mock_op_id", op_type="Sum", inputs=["x"], shape_metadata=(2,), attributes={"axis": 0})

    # For branch coverage: node without shape metadata
    node_no_shape = SimpleNamespace(id="mock_op_id_noshape", op_type="Sum", inputs=["x"], shape_metadata=None, attributes={"axis": 0})
    mean_vjp(graph, node_no_shape, "cotan")
    mean_jvp(graph, node_no_shape, "tan")

    # ... existing test logic

    # Sum
    assert sum_vjp(graph, node, "cotan")
    assert sum_jvp(graph, node, "tan")

    # Mean
    assert mean_vjp(graph, node, "cotan")
    assert mean_jvp(graph, node, "tan")

    # Max
    assert max_vjp(graph, node, "cotan")
    assert max_jvp(graph, node, "tan")

    # Min
    assert min_vjp(graph, node, "cotan")
    assert min_jvp(graph, node, "tan")

    # AddN
    node_multi = SimpleNamespace(id="mock_op_id2", op_type="AddN", inputs=["x", "y"], shape_metadata=(2, 2), attributes={})
    assert add_n_vjp(graph, node_multi, "cotan") == ("cotan", "cotan")
    assert add_n_jvp(graph, node_multi, ["tan1", "tan2"])

    # AccumulateN
    assert accumulate_n_vjp(graph, node_multi, "cotan") == ("cotan", "cotan")
    assert accumulate_n_jvp(graph, node_multi, ["tan1", "tan2"])

    # CumulativeLogsumexp
    assert cumulative_logsumexp_vjp(graph, node, "cotan")
    assert cumulative_logsumexp_jvp(graph, node, "tan")

    # ReduceEuclideanNorm
    assert reduce_euclidean_norm_vjp(graph, node, "cotan")
    assert reduce_euclidean_norm_jvp(graph, node, "tan")

    # Logsumexp
    assert logsumexp_vjp(graph, node, "cotan")
    assert logsumexp_jvp(graph, node, "tan")

    # Average
    assert average_vjp(graph, node, "cotan")
    assert average_jvp(graph, node, "tan")
    average_vjp(graph, node_no_shape, "cotan")

    # AllReduce
    node_ar = SimpleNamespace(id="ar", op_type="AllReduce", inputs=["x"], shape_metadata=(2, 2), attributes={})
    assert allreduce_vjp(graph, node_ar, "cotan")
    assert allreduce_jvp(graph, node_ar, "tan")

    # ReduceScatter
    assert reduce_scatter_vjp(graph, node_ar, "cotan")
    assert reduce_scatter_jvp(graph, node_ar, "tan")

    # AllGather
    assert all_gather_vjp(graph, node_ar, "cotan")
    assert all_gather_jvp(graph, node_ar, "tan")

    # ShardTensor
    assert shard_tensor_vjp(graph, node_ar, "cotan")
    assert shard_tensor_jvp(graph, node_ar, "tan")

    # Broadcast
    assert broadcast_vjp(graph, node_ar, "cotan")
    assert broadcast_jvp(graph, node_ar, "tan")
    assert broadcast_jvp(graph, node_ar, None) is None

    # Test all the zero VJPs / JVPs in the list
    for op in ["Prod", "Cumprod", "Nancumprod", "Nanprod", "Cummax", "Cummin", "Nanmax", "Nanmean", "Nanmedian", "Nanmin", "Nansum", "Nanpercentile", "Nanquantile", "Nanstd", "Nanvar", "Median", "Variance", "Std", "Percentile", "Quantile", "Descriptive", "Bincount"]:
        vjp_fn = _VJP_REGISTRY[op]
        jvp_fn = _JVP_REGISTRY[op]
        assert vjp_fn(graph, node, "cotan")
        assert jvp_fn(graph, node, "tan")
