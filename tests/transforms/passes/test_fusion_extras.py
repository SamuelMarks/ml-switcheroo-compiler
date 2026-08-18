def test_fusion_branch_177():
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.transforms.passes.operator_fusion import PatternMatchingEngine

    graph = LogicalGraph()
    graph.nodes["node1"] = LogicalNode(id="node1", op_type="A", inputs=[])
    graph.nodes["node2"] = LogicalNode(id="node2", op_type="B", inputs=["node1"])

    class MockEngine(PatternMatchingEngine):
        def _try_match_rules(self, graph, node_id, new_nodes, id_map):
            if node_id == "node1":
                # Pretend we matched something that also replaced node2
                new_nodes["node2"] = LogicalNode(id="node2", op_type="C", inputs=[])
                return True
            return False

    MockEngine(rules=[]).apply_passes(graph)
