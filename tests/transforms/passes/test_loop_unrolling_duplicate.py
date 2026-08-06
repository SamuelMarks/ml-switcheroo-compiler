from unittest import mock

from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
from ml_switcheroo_compiler.transforms.passes.loop_unrolling import loop_unrolling_pass


def test_unroll_duplicate_nodes():
    graph = IRGraph()
    n = LogicalNode("n1", "Add")
    graph.nodes = {"n1": n}

    with mock.patch("ml_switcheroo_compiler.transforms.passes.loop_unrolling.DAGTopologicalSorter.sort") as mock_sort:
        mock_sort.return_value = [n, n]
        loop_unrolling_pass(graph)
