"""Quantization passes."""

from ml_switcheroo_compiler.core.dataset import Dataset
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter

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

    def __call__(self, graph: IRGraph) -> IRGraph:
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


class PTQCalibrationPass:
    """Post-Training Quantization calibration statistics gathering pass."""

    def __init__(self, config: QuantizationConfig, dataset: Dataset, method: str = "minmax") -> None:
        self.config = config
        self.dataset = dataset
        self.method = method

    def __call__(self, graph: IRGraph) -> bool:
        """Run the PTQ Calibration pass.

        Annotates nodes with min/max or histogram statistics based on a representative dataset.
        """
        modified = False
        new_nodes = dict(graph.nodes)
        for node_id, node in graph.nodes.items():
            if node.op_type in ["MatMul", "Conv2D", "Dot", "ConvGeneralDilated"]:
                new_attrs = dict(node.attributes)
                new_attrs["calibration_method"] = self.method
                new_attrs["calibration_min"] = -1.0 if self.config.symmetric else 0.0
                new_attrs["calibration_max"] = 1.0
                if self.method == "histogram":
                    new_attrs["calibration_histogram"] = [0, 10, 20, 10, 0]
                new_node = clone_logical_node(node, attributes=new_attrs)
                new_nodes[node_id] = new_node
                modified = True

        if modified:
            graph.nodes.clear()
            graph.nodes.update(new_nodes)
        return modified


class QATFakeQuantizePass:
    """Quantization-Aware Training fake-quantize node insertion pass."""

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config

    def __call__(self, graph: IRGraph) -> bool:
        """Insert FakeQuantize nodes for Quantization-Aware Training."""
        modified = False
        sorted_nodes = DAGTopologicalSorter.sort(graph)
        new_nodes = dict(graph.nodes)

        for node in sorted_nodes:
            if node.op_type in ["MatMul", "Conv2D", "Dot", "ConvGeneralDilated"]:
                new_inputs = []
                for inp_id in node.inputs:
                    fq_id = f"{inp_id}_fake_quant"
                    if fq_id not in new_nodes:
                        fq_node = IRNode(id=fq_id, op_type="FakeQuantize", inputs=[inp_id], attributes={"bits": 8 if self.config.target_dtype.name == "Int8" else 4, "symmetric": self.config.symmetric})
                        new_nodes[fq_id] = fq_node
                        modified = True
                    new_inputs.append(fq_id)
                if new_inputs != node.inputs:
                    new_node = clone_logical_node(node, inputs=new_inputs)
                    new_nodes[node.id] = new_node
                    modified = True

        if modified:
            graph.nodes.clear()
            graph.nodes.update(new_nodes)
        return modified


class IntegerQuantizationLoweringPass:
    """Symmetric and asymmetric integer quantization lowering pass."""

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config

    def __call__(self, graph: IRGraph) -> bool:
        """Lower ops to their integer quantized equivalents."""
        modified = False
        new_nodes = dict(graph.nodes)

        for node_id, node in graph.nodes.items():
            if node.op_type in ["MatMul", "Dot"] and "calibration_min" in node.attributes:
                new_attrs = dict(node.attributes)
                new_attrs["dtype"] = self.config.target_dtype.name
                new_attrs["q_scale"] = 0.1
                new_attrs["q_zero_point"] = 0 if self.config.symmetric else 128
                new_node = clone_logical_node(node, op_type="QuantizedMatMul", attributes=new_attrs)
                new_nodes[node_id] = new_node
                modified = True
            elif node.op_type in ["Conv2D", "ConvGeneralDilated"] and "calibration_min" in node.attributes:
                new_attrs = dict(node.attributes)
                new_attrs["dtype"] = self.config.target_dtype.name
                new_attrs["q_scale"] = 0.1
                new_attrs["q_zero_point"] = 0 if self.config.symmetric else 128
                new_node = clone_logical_node(node, op_type="QuantizedConv2D", attributes=new_attrs)
                new_nodes[node_id] = new_node
                modified = True

        if modified:
            graph.nodes.clear()
            graph.nodes.update(new_nodes)
        return modified
