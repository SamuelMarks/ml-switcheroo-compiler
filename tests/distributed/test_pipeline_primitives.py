"""Tests for pipeline parallelism primitives."""

from ml_switcheroo_compiler.backends.common.mixins.distributed import DistributedASTVisitor
from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class NumpyGenerator:
    def __init__(self):
        self.code = []


def test_distributed_ast_visitor():
    gen = NumpyGenerator()
    visitor = DistributedASTVisitor(generator=gen)

    send_node = IRNode(id="send", op_type="Send", inputs=["in1"], attributes={"target_stage": 1})
    code = visitor.visit_Send(send_node, ["x"])
    assert code == "_numpy_send(x, target=1)"
    assert "Send tensor to pipeline stage" in gen.code[-1]

    recv_node = IRNode(id="recv", op_type="Recv", inputs=[], attributes={"source_stage": 0})
    recv_node.shape_metadata = (10, 10)
    recv_node.dtype = "float32"
    code = visitor.visit_Recv(recv_node, [])
    assert code == "_numpy_recv(source=0, shape=(10, 10), dtype='float32')"
    assert "Recv tensor from pipeline stage" in gen.code[-1]

    gen.__class__.__name__ = "JaxGenerator"
    assert visitor.visit_Send(send_node, ["x"]) == "jax.lax.send(x, dst=1)"
    assert visitor.visit_Recv(recv_node, []) == "jax.lax.recv(src=0)"

    gen.__class__.__name__ = "MlxGenerator"
    assert visitor.visit_Send(send_node, ["x"]) == "mlx.core.distributed.send(x, dst=1)"
    assert visitor.visit_Recv(recv_node, []) == "mlx.core.distributed.recv(src=0)"

    gen.__class__.__name__ = "KerasGenerator"
    assert visitor.visit_Send(send_node, ["x"]) == "keras.distribution.send(x, target=1)"
    assert visitor.visit_Recv(recv_node, []) == "keras.distribution.recv(source=0)"


def test_pipeline_strategy_topology_loading():
    strategy = PipelineParallelismStrategy(topology_name="default")
    assert strategy.num_microbatches == 4
    assert strategy.devices_per_stage == 1
    assert strategy.strategy == "chunk"
    assert strategy.protocol == "p2p_queue"

    strategy = PipelineParallelismStrategy(topology_name="gpipe")
    assert strategy.num_microbatches == 8
    assert strategy.protocol == "rpc"


def test_pipeline_insert_send_recv():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
    g.nodes = {"n1": n1, "n2": n2}

    strategy = PipelineParallelismStrategy()
    stages = [["n1"], ["n2"]]
    strategy.insert_send_recv(g, stages)

    assert "n1_send_0_to_1" in g.nodes
    assert "n1_recv_0_to_1" in g.nodes
    assert g.nodes["n1_send_0_to_1"].inputs == ["n1"]
    assert g.nodes["n1_recv_0_to_1"].inputs == []
    assert g.nodes["n2"].inputs == ["n1_recv_0_to_1"]
