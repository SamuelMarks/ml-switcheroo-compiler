# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""ONNX Target Emission and Real Binary Protobuf Serialization."""

import uuid
from typing import Any, Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


class ONNXCodeGenerator(BaseGenerator):
    """ONNX Code Generator for emitting ONNX graph representations and binary protobuf files.

    Attributes:
        graph (IRGraph): The IR graph to process.
        var_map (dict[str, str]): Mapping of IR node IDs to generated variable names.
    """

    def __init__(self, graph: IRGraph, delegates: Optional[list[Any]] = None) -> None:
        """Initialize ONNXCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list, optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}

    def generic_visit(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Process a node and return its generated ONNX variable name.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        if node is None:
            return "onnx_op"

        nid = getattr(node, "id", str(uuid.uuid4()))
        self.var_map[nid] = nid
        return nid

    # ruff: noqa: PLR0911, PLR0912
    def _get_proto_type(self, dt: str, TensorProto: Any) -> int:
        """Map data type string to ONNX TensorProto primitive integer code.

        Args:
            dt (str): The data type.
            TensorProto (object): ONNX TensorProto namespace.

        Returns:
            int: ONNX TensorProto type integer code.
        """
        dt = str(dt).lower()
        # Access attributes dynamically on the TensorProto object
        if dt == "float64":
            return TensorProto.DOUBLE
        elif dt == "float32":
            return TensorProto.FLOAT
        elif dt == "float16":
            return TensorProto.FLOAT16
        elif dt == "bfloat16":
            return TensorProto.BFLOAT16
        elif dt == "int64":
            return TensorProto.INT64
        elif dt == "int32":
            return TensorProto.INT32
        elif dt == "int16":
            return TensorProto.INT16
        elif dt == "int8":
            return TensorProto.INT8
        elif dt == "uint64":
            return TensorProto.UINT64
        elif dt == "uint32":
            return TensorProto.UINT32
        elif dt == "uint16":
            return TensorProto.UINT16
        elif dt == "uint8":
            return TensorProto.UINT8
        elif dt == "bool":
            return TensorProto.BOOL
        return TensorProto.FLOAT

    def _generate_text_fallback(self) -> str:
        """Generate a text-proto fallback string representation in case ONNX is not available.

        Returns:
            str: Serialized ONNX text representation.
        """
        lines = ["ir_version: 7", 'producer_name: "ml-switcheroo-compiler"', "graph {"]

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Input":
                nid = getattr(node, "id", "")
                shape = "x".join(str(s) for s in (getattr(node, "shape_metadata", ()) or ()))
                dtype = getattr(node, "dtype", "float32")
                lines.append(f'  input: "{nid}" [shape: {shape}, dtype: {dtype}]')

        for node in self.sorted_nodes:
            op_type = getattr(node, "op_type", "")
            if op_type != "Input":
                nid = getattr(node, "id", "")
                inps = ", ".join(f'"{i}"' for i in getattr(node, "inputs", []))
                lines.append(f'  "{nid}" = {op_type}({inps})')

        for out_id in getattr(self.graph, "outputs", []) or []:
            lines.append(f'  output: "{out_id}"')

        lines.append("}")
        return "\n".join(lines)

    def _get_node_and_name(self, item: Any, is_output: bool) -> tuple[Optional[Any], str]:
        """Retrieve a node object and its ID name.

        Args:
            item (object): The IR node or its ID string.
            is_output (bool): True if looking up an output node by ID.

        Returns:
            tuple[Optional[Any], str]: A tuple containing the node and its name string.
        """
        if is_output:
            out_id = item
            node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            return node, out_id
        return item, getattr(item, "id", "")

    def _build_single_value_info(self, item: Any, dynamic_axes: Optional[dict[str, dict[int, str]]], TensorProto: Any, is_output: bool) -> Any:
        """Construct an ONNX TensorValueInfoProto for a single node.

        Args:
            item (object): The IR node or output ID.
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.
            TensorProto (object): The ONNX TensorProto namespace object.
            is_output (bool): True if building for an output.

        Returns: Any: An ONNX ValueInfoProto object.
        """
        from onnx import helper

        node, name = self._get_node_and_name(item, is_output)
        shape = getattr(node, "shape_metadata", ()) or () if node else ()
        dt = getattr(node, "dtype", "float32") if node else "float32"
        proto_type = self._get_proto_type(dt, TensorProto)
        shape_list = list(shape)

        if dynamic_axes and name in dynamic_axes:
            for axis_idx, axis_name in dynamic_axes[name].items():
                shape_list[axis_idx] = axis_name
        return helper.make_tensor_value_info(name, proto_type, shape_list)

    def _build_onnx_value_infos(self, nodes_or_ids: list, dynamic_axes: Optional[dict[str, dict[int, str]]], TensorProto: Any, is_output: bool = False) -> list:
        """Construct a list of ONNX TensorValueInfoProtos.

        Args:
            nodes_or_ids (list): List of IR nodes or output IDs.
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.
            TensorProto (object): The ONNX TensorProto namespace object.
            is_output (bool): True if building for outputs.

        Returns:
            list: A list of ONNX ValueInfoProto objects.
        """
        return [self._build_single_value_info(item, dynamic_axes, TensorProto, is_output) for item in nodes_or_ids]

    def _build_onnx_nodes(self, TensorProto: Any) -> list:
        """Construct all intermediate ONNX NodeProtos for the graph.

        Args:
            TensorProto (object): The ONNX TensorProto namespace object.

        Returns:
            list: A list of ONNX NodeProto objects.
        """
        import math

        from onnx import helper

        onnx_nodes = []
        op_map = {
            "Add": "Add",
            "Subtract": "Sub",
            "Multiply": "Mul",
            "TrueDivide": "Div",
            "Div": "Div",
            "Exp": "Exp",
            "Log": "Log",
            "Negative": "Neg",
            "Neg": "Neg",
            "MatMul": "MatMul",
            "Conv2D": "Conv",
            "MaxPool": "MaxPool",
            "AvgPool": "AveragePool",
            "MaxPool2D": "MaxPool",
            "AvgPool2D": "AveragePool",
            "BatchNorm": "BatchNormalization",
            "Reshape": "Reshape",
            "Transpose": "Transpose",
            "Squeeze": "Squeeze",
            "Unsqueeze": "Unsqueeze",
            "Concat": "Concat",
            "Slice": "Slice",
            "Gather": "Gather",
            "ScatterND": "ScatterND",
            "ReduceSum": "ReduceSum",
            "ReduceMean": "ReduceMean",
            "ReduceMax": "ReduceMax",
            "ReduceMin": "ReduceMin",
            "Relu": "Relu",
            "Sigmoid": "Sigmoid",
            "Tanh": "Tanh",
            "Softmax": "Softmax",
            "Abs": "Abs",
            "Sqrt": "Sqrt",
            "Round": "Round",
            "Floor": "Floor",
            "Ceil": "Ceil",
            "Sin": "Sin",
            "Cos": "Cos",
            "Tan": "Tan",
            "Where": "Where",
            "Equal": "Equal",
            "Greater": "Greater",
            "Less": "Less",
            "Cast": "Cast",
            "Pad": "Pad",
            "If": "If",
            "Loop": "Loop",
            "WhileLoop": "Loop",
        }

        for node in self.sorted_nodes:
            op_type = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid = getattr(node, "id", "")
            inputs = getattr(node, "inputs", [])

            if op_type == "Constant":
                val = node.attributes.get("value", 0.0)
                dt = getattr(node, "dtype", "float32")
                shape = getattr(node, "shape_metadata", ()) or ()
                proto_type = self._get_proto_type(dt, TensorProto)
                num_elements = math.prod(shape) if shape else 1
                tensor_proto = helper.make_tensor(
                    name=nid,
                    data_type=proto_type,
                    dims=list(shape),
                    vals=[val] * num_elements,
                )
                onnx_nodes.append(helper.make_node("Constant", inputs=[], outputs=[nid], name=nid, value=tensor_proto))
            else:
                onnx_op = op_map.get(op_type, op_type)
                onnx_nodes.append(helper.make_node(onnx_op, inputs=inputs, outputs=[nid], name=nid))
        return onnx_nodes

    def _build_onnx_graph(self, dynamic_axes: Optional[dict[str, dict[int, str]]]) -> Any:
        """Construct the full ONNX GraphProto.

        Args:
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.

        Returns: Any: The ONNX GraphProto object.
        """
        from onnx import TensorProto, helper

        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        onnx_inputs = self._build_onnx_value_infos(input_nodes, dynamic_axes, TensorProto, is_output=False)

        output_ids = getattr(self.graph, "outputs", []) or []
        onnx_outputs = self._build_onnx_value_infos(output_ids, dynamic_axes, TensorProto, is_output=True)

        onnx_nodes = self._build_onnx_nodes(TensorProto)
        return helper.make_graph(onnx_nodes, "ml_switcheroo_graph", onnx_inputs, onnx_outputs)

    def generate(self, dynamic_axes: Optional[dict[str, dict[int, str]]] = None) -> str:
        """Generate a readable string/text-proto representation of the ONNX Graph.

        Args:
        dynamic_axes (object): The dynamic_axes parameter.

        Returns:
        str: Result.
        """
        try:
            from onnx import helper

            graph_def = self._build_onnx_graph(dynamic_axes)
            # Use to_text instead of printable_graph if available to avoid deprecation warning
            try:
                from onnx import printer

                res = printer.to_text(graph_def)
                if not isinstance(res, str):
                    return "PrintableGraph"
                return res
            except ImportError:
                return str(helper.printable_graph(graph_def))
        except ImportError:
            return self._generate_text_fallback()

    def export_onnx(self, file_path: str, dynamic_axes: Optional[dict[str, dict[int, str]]] = None) -> None:
        """Export the IR Graph as a real, compliant binary .onnx file to disk.

        Args:
            file_path (str): The file_path parameter.
            dynamic_axes (Optional): The dynamic_axes parameter.
        """
        from onnx import helper

        graph_def = self._build_onnx_graph(dynamic_axes)
        model_def = helper.make_model(graph_def, producer_name="ml-switcheroo-compiler")

        with open(file_path, "wb") as f:
            f.write(model_def.SerializeToString())
