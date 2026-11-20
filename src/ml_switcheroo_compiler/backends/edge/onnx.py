"""ONNX Target Emission and Pure-Python Binary Protobuf Serialization."""

from __future__ import annotations

import os
import uuid

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._onnx_graph_builder import _OnnxGraphBuilderMixin
from ml_switcheroo_compiler.backends.edge._onnx_proto import (
    _serialize_sequence_attribute,
    serialize_attribute,
    serialize_dimension,
    serialize_graph,
    serialize_model,
    serialize_node,
    serialize_operator_set_id,
    serialize_tensor_proto,
    serialize_tensor_shape,
    serialize_type_proto,
    serialize_value_info,
)
from ml_switcheroo_compiler.backends.edge._onnx_serializer import _OnnxSerializerMixin
from ml_switcheroo_compiler.backends.edge._onnx_wire import (
    GraphProtoType,
    NodeProtoType,
    T,
    TensorProtoType,
    ValueInfoProtoType,
    _node_attrs_dict,
    _node_inputs_list,
    _node_shape_tuple,
    _parse_opset_import_version,
    _skip_protobuf_wire_field,
    decode_varint,
    encode_float,
    encode_int,
    encode_length_delimited,
    encode_string,
    encode_tag,
    encode_varint,
    validate_onnx_model_bytes,
    validate_topological_sort,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.registry import register_backend

__all__ = [
    "GraphProtoType",
    "NodeProtoType",
    "ONNXCodeGenerator",
    "T",
    "TensorProtoType",
    "ValueInfoProtoType",
    "_OnnxGraphBuilderMixin",
    "_OnnxSerializerMixin",
    "_node_attrs_dict",
    "_node_inputs_list",
    "_node_shape_tuple",
    "_parse_opset_import_version",
    "_serialize_sequence_attribute",
    "_skip_protobuf_wire_field",
    "decode_varint",
    "encode_float",
    "encode_int",
    "encode_length_delimited",
    "encode_string",
    "encode_tag",
    "encode_varint",
    "serialize_attribute",
    "serialize_dimension",
    "serialize_graph",
    "serialize_model",
    "serialize_node",
    "serialize_operator_set_id",
    "serialize_tensor_proto",
    "serialize_tensor_shape",
    "serialize_type_proto",
    "serialize_value_info",
    "validate_onnx_model_bytes",
    "validate_topological_sort",
]


@register_backend("edge")
@register_backend("edge_onnx")
class ONNXCodeGenerator(_OnnxSerializerMixin, BaseGenerator):
    """ONNX Code Generator for emitting ONNX graph representations and binary protobuf files.

    Attributes:
        graph (IRGraph): The IR graph to process.
        var_map (dict[str, str]): Mapping of IR node IDs to generated variable names.
    """

    def generic_visit(self, node: IRNode | None, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Process a node and return its generated ONNX variable name.

        Args:
            node (IRNode | None): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        if node is None:
            return "onnx_op"

        nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", str(uuid.uuid4())))
        self.var_map[nid] = nid
        return nid

    def generate(self, dynamic_axes: dict[str, dict[int, str]] | None = None) -> str:
        """Generate a readable string/text-proto representation of the ONNX Graph.

        Args:
            dynamic_axes (dict[str, dict[int, str]] | None): Optional dynamic axis mapping configuration.

        Returns:
            str: Text-proto or printable representation of the ONNX computation graph.
        """
        try:
            from onnx import helper

            graph_def = self._build_onnx_graph(dynamic_axes)
            try:
                from onnx import printer

                res = printer.to_text(graph_def)
                return str(res)
            except (ImportError, AttributeError):
                return str(helper.printable_graph(graph_def))
        except (ImportError, AttributeError):
            return self._generate_text_fallback()

    def export_onnx(
        self,
        file_path: str,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 18,
    ) -> None:
        """Export the IR Graph as a real, compliant binary .onnx file to disk.

        Args:
            file_path (str): Destination path for the binary `.onnx` file.
            dynamic_axes (dict[str, dict[int, str]] | None): Optional dynamic axis mapping configuration.
            opset_version (int): Target ONNX opset version (14 to 20).
        """
        binary_data: bytes = self.serialize_model_to_bytes(dynamic_axes, opset_version=opset_version)

        try:
            from onnx import checker, load_model_from_string

            model_def = load_model_from_string(binary_data)
            checker.check_model(model_def)
        except (ImportError, AttributeError):
            validate_onnx_model_bytes(binary_data)

        with open(file_path, "wb") as f:
            f.write(binary_data)

    def verify(self, file_path: str | None = None) -> bool:
        """Verify the serialized ONNX model binary structure.

        Args:
            file_path (str | None): Optional path to saved .onnx file; if None, in-memory model is verified.

        Returns:
            bool: True if model is verified valid.
        """
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data: bytes = f.read()
        else:
            data = self.serialize_model_to_bytes()
        meta: dict[str, int | str] = validate_onnx_model_bytes(data)
        return bool(int(meta.get("opset_version", 0)) >= 14 or int(meta.get("ir_version", 0)) > 0)

    @staticmethod
    def _prepare_runner_input_feed(
        graph: IRGraph,
        w_args: tuple[BackendArg, ...],
        w_kwargs: dict[str, BackendArg],
    ) -> dict[str, BackendArg]:
        """Prepare an input feed dictionary mapping graph input IDs to raw tensor values.

        Args:
            graph (IRGraph): Target computation graph.
            w_args (tuple[BackendArg, ...]): Positional input arguments.
            w_kwargs (dict[str, BackendArg]): Keyword input arguments.

        Returns:
            dict[str, BackendArg]: Mapped input feed dictionary.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        input_nodes = [
            n for n in graph.nodes.values() if coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) == "Input"
        ]
        input_feed: dict[str, BackendArg] = {}
        for i, inp_node in enumerate(input_nodes):
            if i < len(w_args):
                arg_val = w_args[i]
                input_feed[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
        for k, v in w_kwargs.items():
            input_feed[k] = v.data if isinstance(v, Tensor) else v
        return input_feed

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Construct a serialized onnx.ModelProto artifact and initialize an execution session.

        Args:
            graph (IRGraph): The target computation graph to compile.
            **kwargs (BackendArg): Optional compiler arguments ('opset_version', 'dynamic_axes').

        Returns:
            BackendArg: Runnable execution session callable executing the ONNX model.
        """
        opset_version = int(coerce_int(kwargs.get("opset_version", 18)))
        dynamic_axes = kwargs.get("dynamic_axes")
        dyn_map = dynamic_axes if isinstance(dynamic_axes, dict) else None

        model_bytes = self.serialize_model_to_bytes(dynamic_axes=dyn_map, opset_version=opset_version)

        try:
            import onnx.reference

            session = onnx.reference.ReferenceEvaluator(model_bytes)

            def onnx_reference_runner(*w_args: BackendArg, **w_kwargs: BackendArg) -> BackendArg:
                """Execute ONNX model via ONNX ReferenceEvaluator.

                Args:
                    *w_args (BackendArg): Input tensor values.
                    **w_kwargs (BackendArg): Keyword inputs.

                Returns:
                    BackendArg: Computed outputs.
                """
                from ml_switcheroo_compiler.backends.eager_registry import require_callable

                return require_callable(session.run)(None, self._prepare_runner_input_feed(graph, w_args, w_kwargs))

            return onnx_reference_runner
        except (ImportError, ModuleNotFoundError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
            from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

            def fallback_runner(*w_args: BackendArg, **w_kwargs: BackendArg) -> BackendArg:
                """Execute graph via interpreter evaluator fallback.

                Args:
                    *w_args (BackendArg): Input tensors.
                    **w_kwargs (BackendArg): Keyword arguments.

                Returns:
                    BackendArg: Evaluated output tensors.
                """
                evaluated = evaluate_graph(graph, inputs=self._prepare_runner_input_feed(graph, w_args, w_kwargs))
                outputs = adapt_backend_module(graph).get_symbol("outputs", None)
                if isinstance(outputs, (list, tuple)) and outputs:
                    return (
                        evaluated.get(str(outputs[0]))
                        if len(outputs) == 1
                        else tuple(evaluated.get(str(out_id)) for out_id in outputs)
                    )
                return evaluated

            return fallback_runner
