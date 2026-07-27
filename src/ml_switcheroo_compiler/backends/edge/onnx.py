# ruff: noqa: E501
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

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
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

    def _get_proto_type(self, dt: str, TensorProto: object) -> int:
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
        elif dt == "int32":
            return TensorProto.INT32
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

    def generate(self) -> str:
        """Generate a readable string/text-proto representation of the ONNX Graph.

        Returns:
            str: Serialized ONNX printable text representation.
        """
        try:
            import numpy as np
            from onnx import TensorProto, helper, numpy_helper

            input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
            onnx_inputs = []
            for node in input_nodes:
                name = getattr(node, "id", "")
                shape = getattr(node, "shape_metadata", ()) or ()
                dt = getattr(node, "dtype", "float32")
                proto_type = self._get_proto_type(dt, TensorProto)
                onnx_inputs.append(helper.make_tensor_value_info(name, proto_type, list(shape)))

            output_ids = getattr(self.graph, "outputs", []) or []
            onnx_outputs = []
            for out_id in output_ids:
                out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
                shape = getattr(out_node, "shape_metadata", ()) or () if out_node else ()
                dt = getattr(out_node, "dtype", "float32") if out_node else "float32"
                proto_type = self._get_proto_type(dt, TensorProto)
                onnx_outputs.append(helper.make_tensor_value_info(out_id, proto_type, list(shape)))

            onnx_nodes = []
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
                    if shape:
                        arr = np.full(shape, val, dtype=np.dtype(dt))
                    else:
                        arr = np.array(val, dtype=np.dtype(dt))
                    tensor_proto = numpy_helper.from_array(arr, name=nid)
                    onnx_nodes.append(
                        helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[nid],
                            name=nid,
                            value=tensor_proto,
                        )
                    )
                    continue

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
                }
                onnx_op = op_map.get(op_type, op_type)

                onnx_nodes.append(
                    helper.make_node(
                        onnx_op,
                        inputs=inputs,
                        outputs=[nid],
                        name=nid,
                    )
                )

            graph_def = helper.make_graph(onnx_nodes, "ml_switcheroo_graph", onnx_inputs, onnx_outputs)
            return str(helper.printable_graph(graph_def))

        except ImportError:
            return self._generate_text_fallback()

    def export_onnx(self, file_path: str) -> None:
        """Export the IR Graph as a real, compliant binary .onnx file to disk.

        Args:
            file_path (str): Path to write the .onnx binary file.

        Raises:
            ImportError: If the 'onnx' library is not installed.
        """
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        onnx_inputs = []
        for node in input_nodes:
            name = getattr(node, "id", "")
            shape = getattr(node, "shape_metadata", ()) or ()
            dt = getattr(node, "dtype", "float32")
            proto_type = self._get_proto_type(dt, TensorProto)
            onnx_inputs.append(helper.make_tensor_value_info(name, proto_type, list(shape)))

        output_ids = getattr(self.graph, "outputs", []) or []
        onnx_outputs = []
        for out_id in output_ids:
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            shape = getattr(out_node, "shape_metadata", ()) or () if out_node else ()
            dt = getattr(out_node, "dtype", "float32") if out_node else "float32"
            proto_type = self._get_proto_type(dt, TensorProto)
            onnx_outputs.append(helper.make_tensor_value_info(out_id, proto_type, list(shape)))

        onnx_nodes = []
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
                if shape:
                    arr = np.full(shape, val, dtype=np.dtype(dt))
                else:
                    arr = np.array(val, dtype=np.dtype(dt))
                tensor_proto = numpy_helper.from_array(arr, name=nid)
                onnx_nodes.append(
                    helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[nid],
                        name=nid,
                        value=tensor_proto,
                    )
                )
                continue

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
            }
            onnx_op = op_map.get(op_type, op_type)

            onnx_nodes.append(
                helper.make_node(
                    onnx_op,
                    inputs=inputs,
                    outputs=[nid],
                    name=nid,
                )
            )

        graph_def = helper.make_graph(onnx_nodes, "ml_switcheroo_graph", onnx_inputs, onnx_outputs)
        model_def = helper.make_model(graph_def, producer_name="ml-switcheroo-compiler")

        with open(file_path, "wb") as f:
            f.write(model_def.SerializeToString())
