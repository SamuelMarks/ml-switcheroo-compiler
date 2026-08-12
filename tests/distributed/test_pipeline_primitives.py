from unittest.mock import MagicMock, patch

import ml_switcheroo_compiler.ops.distributed_ops as dist_ops
from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
from ml_switcheroo_compiler.ops.distributed_ops import Recv, Send


def test_send_recv_ops() -> None:
    """Test send and recv ops shape inference and logic."""
    t = MagicMock()
    t.shape = (2, 3)

    send_op = Send()
    assert send_op.infer_shape(t) == ()

    recv_op = Recv()
    assert recv_op.infer_shape(shape=(5, 5)) == (5, 5)


def test_pipeline_strategy_insert() -> None:
    """Test PipelineParallelismStrategy inserts Send and Recv."""
    graph = IRGraph()
    n1 = LogicalNode(id="n1", op_type="Input")
    n1.shape_metadata = (10, 10)
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1", "n1"])
    n3 = LogicalNode(id="n3", op_type="Exp", inputs=["n2"])

    graph.nodes = {"n1": n1, "n2": n2, "n3": n3}

    strategy = PipelineParallelismStrategy(num_microbatches=2)
    stages = [["n1"], ["n2", "n3"]]

    strategy.insert_send_recv(graph, stages)

    op_types = [n.op_type for n in graph.nodes.values()]
    assert "Send" in op_types
    assert "Recv" in op_types

    recv_id = None
    for nid, n in graph.nodes.items():
        if n.op_type == "Send":
            assert n.inputs[0] == "n1"
            if "dst_rank" in n.attributes:
                assert n.attributes["dst_rank"] == 1
        if n.op_type == "Recv":
            recv_id = nid
            if "src_rank" in n.attributes:
                assert n.attributes["src_rank"] == 0
            if "shape" in n.attributes:
                assert n.attributes["shape"] == (10, 10)

    assert graph.nodes["n2"].inputs[0] == recv_id
    assert graph.nodes["n2"].inputs[1] == recv_id


@patch("ml_switcheroo_compiler.tracing.builder.TracingNodeBuilder.emit_tracing_node")
def test_send_recv_functions(mock_emit) -> None:
    """Test send and recv factory functions trace logic."""
    dist_ops.send("t_in", dst_rank=1, tag=42)
    mock_emit.assert_called_with("Send", "t_in", dst_rank=1, tag=42)

    dist_ops.recv(shape=(10, 10), dtype="float32", src_rank=0, tag=42)
    mock_emit.assert_called_with("Recv", shape=(10, 10), dtype="float32", src_rank=0, tag=42)


def test_pipeline_microbatch_loop():
    """Test pipeline generate microbatch loop slicing and concating."""
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    n1 = LogicalNode(id="n1", op_type="Input")
    n1.shape_metadata = (10, 10)
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1", "n1"])
    graph.nodes = {"n1": n1, "n2": n2}
    graph.outputs = ["n2"]

    strategy = PipelineParallelismStrategy(num_microbatches=4)
    strategy.generate_microbatch_loop(graph)

    assert "n2_concat" in graph.nodes
    assert "microbatch_loop" in graph.nodes
    assert "n1_slice" in graph.nodes["microbatch_loop"].attributes["body"].nodes
    assert graph.outputs == ["n2_concat"]
