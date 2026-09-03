# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""ONNX Target Emission and Real Binary Protobuf Serialization."""

import os
import uuid
from typing import Optional, TypeVar, Union

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

T = TypeVar("T")
TensorProtoType = TypeVar("TensorProtoType")
ValueInfoProtoType = TypeVar("ValueInfoProtoType")
NodeProtoType = TypeVar("NodeProtoType")
GraphProtoType = TypeVar("GraphProtoType")


class ONNXCodeGenerator(BaseGenerator):
    """ONNX Code Generator for emitting ONNX graph representations and binary protobuf files.

    Attributes:
        graph (IRGraph): The IR graph to process.
        var_map (dict[str, str]): Mapping of IR node IDs to generated variable names.
    """

    def __init__(self, graph: IRGraph, delegates: Optional[T] = None) -> None:
        """Initialize ONNXCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Optional[T], optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}

        yaml_path: str = os.path.join(os.path.dirname(__file__), "onnx_schema.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = {}

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: str) -> str:
        """Process a node and return its generated ONNX variable name.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (str): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        if node is None:
            return "onnx_op"

        nid: str = getattr(node, "id", str(uuid.uuid4()))
        self.var_map[nid] = nid
        return nid

    # ruff: noqa: PLR0911, PLR0912
    def _get_proto_type(self, dt: str, TensorProto: type[TensorProtoType]) -> int:
        """Map data type string to ONNX TensorProto primitive integer code.

        Args:
            dt (str): The dtype string.
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace.

        Returns:
            int: The protobuf enum type.
        """
        dt_str: str = str(dt).lower()
        dt_map: dict[str, int] = self.schema.get("types", {})
        return dt_map.get(dt_str, 1)

    def _generate_text_fallback(self) -> str:
        """Generate a text-proto fallback string representation in case ONNX is not available.

        Returns:
            str: Serialized ONNX text representation.
        """
        lines: list[str] = ["ir_version: 7", 'producer_name: "ml-switcheroo-compiler"', "graph {"]

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Input":
                nid: str = getattr(node, "id", "")
                shape: str = "x".join(str(s) for s in (getattr(node, "shape_metadata", ()) or ()))
                dtype: str = getattr(node, "dtype", "float32")
                lines.append(f'  input: "{nid}" [shape: {shape}, dtype: {dtype}]')

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type != "Input":
                nid: str = getattr(node, "id", "")
                inps: str = ", ".join(f'"{i}"' for i in getattr(node, "inputs", []))
                lines.append(f'  "{nid}" = {op_type}({inps})')

        for out_id in getattr(self.graph, "outputs", []) or []:
            lines.append(f'  output: "{out_id}"')

        lines.append("}")
        return "\n".join(lines)

    def _get_node_and_name(self, item: Union[IRNode, str], is_output: bool) -> tuple[Optional[IRNode], str]:
        """Retrieve a node object and its ID name.

        Args:
            item (Union[IRNode, str]): The IR node or its ID string.
            is_output (bool): True if looking up an output node by ID.

        Returns:
            tuple[Optional[IRNode], str]: A tuple containing the node and its name string.
        """
        if is_output:
            out_id: str = str(item)
            node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            return node, out_id
        return item, getattr(item, "id", "")

    def _build_single_value_info(self, item: Union[IRNode, str], dynamic_axes: Optional[dict[str, dict[int, str]]], TensorProto: type[TensorProtoType], is_output: bool) -> ValueInfoProtoType:
        """Construct an ONNX TensorValueInfoProto for a single node.

        Args:
            item (Union[IRNode, str]): The IR node or output ID.
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.
            is_output (bool): True if building for an output.

        Returns:
            ValueInfoProtoType: An ONNX ValueInfoProto object.
        """
        from onnx import helper

        node, name = self._get_node_and_name(item, is_output)
        shape: tuple[int, ...] = getattr(node, "shape_metadata", ()) or () if node else ()
        dt: str = getattr(node, "dtype", "float32") if node else "float32"
        proto_type: int = self._get_proto_type(dt, TensorProto)
        shape_list: list[int | str] = list(shape)

        if dynamic_axes and name in dynamic_axes:
            for axis_idx, axis_name in dynamic_axes[name].items():
                shape_list[axis_idx] = axis_name
        return helper.make_tensor_value_info(name, proto_type, shape_list)

    def _build_onnx_value_infos(self, nodes_or_ids: list[Union[IRNode, str]], dynamic_axes: Optional[dict[str, dict[int, str]]], TensorProto: type[TensorProtoType], is_output: bool = False) -> list[ValueInfoProtoType]:
        """Construct a list of ONNX TensorValueInfoProtos.

        Args:
            nodes_or_ids (list[Union[IRNode, str]]): List of IR nodes or output IDs.
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.
            is_output (bool): True if building for outputs.

        Returns:
            list[ValueInfoProtoType]: A list of ONNX ValueInfoProto objects.
        """
        return [self._build_single_value_info(item, dynamic_axes, TensorProto, is_output) for item in nodes_or_ids]

    def _build_onnx_nodes(self, TensorProto: type[TensorProtoType]) -> list[NodeProtoType]:
        """Construct all intermediate ONNX NodeProtos for the graph.

        Args:
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.

        Returns:
            list[NodeProtoType]: A list of ONNX NodeProto objects.
        """
        import math

        from onnx import helper

        onnx_nodes: list[NodeProtoType] = []

        # We now query ops definitions for edge_onnx mappings
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        def get_onnx_op_name(op_type: str) -> str:
            """get_onnx_op_name function.

            Args:
                op_type (str): The op_type parameter.

            Returns:
                str: Result.
            """
            op_def = OPS_REGISTRY.get(op_type, {})
            variants = op_def.get("variants", {})
            if "edge_onnx" in variants:
                gen: str = variants["edge_onnx"].get("generator")
                if gen:
                    return gen
            if op_type not in self.schema.get("operations", {}):
                from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

                raise BackendNotSupportedError(f"Operation '{op_type}' not supported in ONNX schema.")
            return str(op_type)

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid: str = getattr(node, "id", "")
            inputs: list[str] = getattr(node, "inputs", [])

            if op_type == "Constant":
                val: float = node.attributes.get("value", 0.0)
                dt: str = getattr(node, "dtype", "float32")
                shape: tuple[int, ...] = getattr(node, "shape_metadata", ()) or ()
                proto_type: int = self._get_proto_type(dt, TensorProto)
                num_elements: int = math.prod(shape) if shape else 1
                tensor_proto = helper.make_tensor(
                    name=nid,
                    data_type=proto_type,
                    dims=list(shape),
                    vals=[val] * num_elements,
                )
                onnx_nodes.append(helper.make_node("Constant", inputs=[], outputs=[nid], name=nid, value=tensor_proto))
            else:
                onnx_op: str = get_onnx_op_name(op_type)
                kwargs: dict[str, str | int | float | list[int] | GraphProtoType] = {}
                if op_type == "If":
                    if "then_branch" in node.attributes:
                        subgen = ONNXCodeGenerator(node.attributes["then_branch"])
                        kwargs["then_branch"] = subgen._build_onnx_graph(None)
                        # Fix graph name for ONNX validation
                        kwargs["then_branch"].name = f"{nid}_then"
                    if "else_branch" in node.attributes:
                        subgen = ONNXCodeGenerator(node.attributes["else_branch"])
                        kwargs["else_branch"] = subgen._build_onnx_graph(None)
                        kwargs["else_branch"].name = f"{nid}_else"
                elif op_type in ("Loop", "WhileLoop"):
                    if "body" in node.attributes:
                        subgen = ONNXCodeGenerator(node.attributes["body"])
                        kwargs["body"] = subgen._build_onnx_graph(None)
                        kwargs["body"].name = f"{nid}_body"

                onnx_nodes.append(helper.make_node(onnx_op, inputs=inputs, outputs=[nid], name=nid, **kwargs))
        return onnx_nodes

    def _build_onnx_graph(self, dynamic_axes: Optional[dict[str, dict[int, str]]] = None) -> GraphProtoType:
        """Construct the full ONNX GraphProto.

        Args:
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.

        Returns:
            GraphProtoType: The ONNX GraphProto object.
        """
        from onnx import TensorProto, helper

        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        onnx_inputs = self._build_onnx_value_infos(input_nodes, dynamic_axes, TensorProto, is_output=False)

        output_ids: list[str] = getattr(self.graph, "outputs", []) or []
        onnx_outputs = self._build_onnx_value_infos(output_ids, dynamic_axes, TensorProto, is_output=True)

        onnx_nodes = self._build_onnx_nodes(TensorProto)
        return helper.make_graph(onnx_nodes, "ml_switcheroo_graph", onnx_inputs, onnx_outputs)

    def generate(self, dynamic_axes: Optional[dict[str, dict[int, str]]] = None) -> str:
        """Generate a readable string/text-proto representation of the ONNX Graph.

        Args:
            dynamic_axes (Optional[dict[str, dict[int, str]]]): The dynamic_axes parameter.

        Returns:
            str: Result.
        """
        try:
            from onnx import helper

            graph_def = self._build_onnx_graph(dynamic_axes)
            # Use to_text instead of printable_graph if available to avoid deprecation warning
            try:
                from onnx import printer

                res: str = printer.to_text(graph_def)
                if not isinstance(res, str):
                    return str(res)
                return res
            except ImportError:
                return str(helper.printable_graph(graph_def))
        except ImportError:
            return self._generate_text_fallback()

    def export_onnx(self, file_path: str, dynamic_axes: Optional[dict[str, dict[int, str]]] = None) -> None:
        """Export the IR Graph as a real, compliant binary .onnx file to disk.

        Args:
            file_path (str): The file_path parameter.
            dynamic_axes (Optional[dict[str, dict[int, str]]]): The dynamic_axes parameter.
        """
        from onnx import checker, helper

        graph_def = self._build_onnx_graph(dynamic_axes)
        model_def = helper.make_model(graph_def, producer_name="ml-switcheroo-compiler")

        # Validate using official checker
        checker.check_model(model_def)

        with open(file_path, "wb") as f:
            f.write(model_def.SerializeToString())
