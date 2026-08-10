import numpy as np

from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_pipeline_runtime_execution():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (10,)
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["in1", "in1"])
    n3 = LogicalNode(id="n3", op_type="Exp", inputs=["n2"])

    graph.nodes = {"in1": n1, "n2": n2, "n3": n3}
    graph.outputs = ["n3"]

    strategy = PipelineParallelismStrategy(num_microbatches=1)

    inputs = {"in1": np.ones((10,), dtype=np.float32)}

    # Split into 2 stages, so in1 and n2 in stage 0, and n3 in stage 1
    outputs = strategy.execute_pipeline(graph, inputs, num_stages=2)

    assert "n3" in outputs
    np.testing.assert_allclose(outputs["n3"], np.exp(np.ones(10) * 2))
