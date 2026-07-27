"""Quantization passes."""

from ml_switcheroo_compiler.core.dataset import Dataset
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import LogicalGraph, clone_logical_node

# ruff: noqa: D107


class QuantizationConfig:
    """Quantization specification configuration."""

    def __init__(self, target_dtype: DType, per_channel: bool = False, symmetric: bool = True) -> None:
        self.target_dtype = target_dtype
        self.per_channel = per_channel
        self.symmetric = symmetric


class PTQPass:
    """Post-Training Quantization pass."""

    def __init__(self, config: QuantizationConfig, representative_dataset: Dataset) -> None:
        self.config = config
        self.dataset = representative_dataset

    def __call__(self, graph: LogicalGraph) -> LogicalGraph:
        """Run the PTQ pass on the graph.

        Annotates applicable linear algebra operations for lowering to quantized backends.
        """
        optimized = False
        new_nodes = {}
        for node_id, node in graph.nodes.items():
            if node.op_type in ["Dot", "ConvGeneralDilated"]:
                # Mark node for quantization
                new_attrs = dict(node.attributes)
                new_attrs["ptq_target_dtype"] = self.config.target_dtype.name
                new_attrs["ptq_per_channel"] = self.config.per_channel
                new_attrs["ptq_symmetric"] = self.config.symmetric
                new_node = clone_logical_node(node, attributes=new_attrs)
                new_nodes[node_id] = new_node
                optimized = True
            else:
                new_nodes[node_id] = node

        if optimized:
            graph.nodes.clear()
            graph.nodes.update(new_nodes)

        return graph
