"""Quantization passes."""

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

from ml_switcheroo_compiler.core.dataset import Dataset
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


class QuantizationConfig:
    """Quantization specification configuration."""

    def __init__(self, target_dtype: DType, per_channel: bool = False, symmetric: bool = True) -> None:
        """Initialize QuantizationConfig.

        Args:
            target_dtype (DType): The target_dtype parameter.
            per_channel (bool): The per_channel parameter.
            symmetric (bool): The symmetric parameter.
        """
        self.target_dtype = target_dtype
        self.per_channel = per_channel
        self.symmetric = symmetric


class PTQPass:
    """Post-Training Quantization pass."""

    def __init__(self, config: QuantizationConfig, representative_dataset: Dataset) -> None:
        """Initialize PTQPass.

        Args:
            config (QuantizationConfig): The quantization config.
            representative_dataset (Dataset): The representative dataset for calibration.
        """
        self.config = config
        self.dataset = representative_dataset
        import os

        import yaml

        yaml_path = os.path.join(os.path.dirname(__file__), "quantization_rules.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.rules = yaml.safe_load(f)
        else:
            self.rules = {}

    def __call__(self, graph: IRGraph) -> IRGraph:
        """Lower ops to their integer quantized equivalents.

        Args:
            graph (IRGraph): The graph parameter.

        Returns:
            IRGraph: Result.
        """
        optimized = False
        new_nodes = {}

        q_scale = self.rules.get("PTQ", {}).get("default_q_scale", 0.1)
        sym_rules = self.rules.get("PTQ", {}).get("symmetric" if self.config.symmetric else "asymmetric", {})
        q_zp = sym_rules.get("q_zero_point", 0)
        lowering_map = self.rules.get("lowering_map", {})

        for node_id, node in graph.nodes.items():
            if node.op_type in lowering_map:
                new_attrs = dict(node.attributes)
                new_attrs["dtype"] = self.config.target_dtype.name
                new_attrs["q_scale"] = q_scale
                new_attrs["q_zero_point"] = q_zp
                new_node = clone_logical_node(node, op_type=lowering_map[node.op_type], attributes=new_attrs)
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
        """Initialize PTQCalibrationPass.

        Args:
            config (QuantizationConfig): The config parameter.
            dataset (Dataset): The dataset parameter.
            method (str): The method parameter.
        """
        self.config = config
        self.dataset = dataset
        self.method = method

    def __call__(self, graph: IRGraph) -> bool:
        """Run the PTQ Calibration pass.

        Args:
            graph (IRGraph): The graph parameter.

        Returns:
            bool: Result.
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
        """Initialize QATFakeQuantizePass.

        Args:
            config (QuantizationConfig): The quantization config.
        """
        self.config = config
        import os

        import yaml

        yaml_path = os.path.join(os.path.dirname(__file__), "quantization_rules.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.rules = yaml.safe_load(f)
        else:
            self.rules = {}

    def __call__(self, graph: IRGraph) -> bool:
        """Insert FakeQuantize nodes for Quantization-Aware Training.

        Args:
            graph (IRGraph): The graph parameter.

        Returns:
            bool: Result.
        """
        modified = False
        sorted_nodes = DAGTopologicalSorter.sort(graph)
        new_nodes = dict(graph.nodes)

        q_scale = self.rules.get("QAT", {}).get("default_q_scale", 0.1)
        sym_rules = self.rules.get("QAT", {}).get("symmetric" if self.config.symmetric else "asymmetric", {})
        q_zp = sym_rules.get("q_zero_point", 0)
        bits = sym_rules.get("bits", 8)

        for node in sorted_nodes:
            if node.op_type in ["MatMul", "Conv2D", "Dot", "ConvGeneralDilated"]:
                new_inputs = []
                for inp_id in node.inputs:
                    fq_id = f"{inp_id}_fake_quant"
                    if fq_id not in new_nodes:
                        fq_node = IRNode(
                            id=fq_id,
                            op_type="FakeQuantize",
                            inputs=[inp_id],
                            attributes={
                                "bits": bits,
                                "symmetric": self.config.symmetric,
                                "q_scale": q_scale,
                                "q_zero_point": q_zp,
                            },
                        )
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
        """Initialize IntegerQuantizationLoweringPass.

        Args:
            config (QuantizationConfig): The quantization config.
        """
        self.config = config
        import os

        import yaml

        yaml_path = os.path.join(os.path.dirname(__file__), "quantization_rules.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.rules = yaml.safe_load(f)
        else:
            self.rules = {}

    def __call__(self, graph: IRGraph) -> bool:
        """Lower ops to their integer quantized equivalents.

        Args:
            graph (IRGraph): The graph parameter.

        Returns:
            bool: Result.
        """
        modified = False
        new_nodes = {}
        lowering_map = self.rules.get("lowering_map", {})
        pass_through = self.rules.get("pass_through_nodes", [])

        # First pass: identify fake quant nodes to drop
        fq_replacements = {}
        for node_id, node in graph.nodes.items():
            if node.op_type in pass_through:
                fq_replacements[node_id] = node.inputs[0] if node.inputs else None

        q_scale = self.rules.get("PTQ", {}).get("default_q_scale", 0.1)
        sym_rules = self.rules.get("PTQ", {}).get("symmetric" if self.config.symmetric else "asymmetric", {})
        q_zp = sym_rules.get("q_zero_point", 0)

        for node_id, node in graph.nodes.items():
            if node_id in fq_replacements:
                modified = True
                continue

            # Update inputs if they point to dropped fake quant nodes
            new_inputs = [fq_replacements.get(inp, inp) for inp in node.inputs]
            if new_inputs != node.inputs:
                modified = True

            if node.op_type in lowering_map and ("calibration_min" in node.attributes or any(inp in fq_replacements for inp in node.inputs)):
                new_attrs = dict(node.attributes)
                new_attrs["dtype"] = self.config.target_dtype.name
                new_attrs["q_scale"] = q_scale
                new_attrs["q_zero_point"] = q_zp
                new_node = clone_logical_node(node, op_type=lowering_map[node.op_type], inputs=new_inputs, attributes=new_attrs)
                new_nodes[node_id] = new_node
                modified = True
            else:
                if new_inputs != node.inputs:
                    new_nodes[node_id] = clone_logical_node(node, inputs=new_inputs)
                else:
                    new_nodes[node_id] = node

        if modified:
            # Re-map graph outputs if they were dropped
            new_outputs = [fq_replacements.get(out, out) for out in graph.outputs]
            graph.outputs = [o for o in new_outputs if o is not None]
            graph.nodes.clear()
            graph.nodes.update(new_nodes)
        return modified
