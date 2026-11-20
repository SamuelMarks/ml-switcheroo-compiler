"""Pure-Python binary ONNX ModelProto and GraphProto serialization mixin."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._onnx_graph_builder import _OnnxGraphBuilderMixin
from ml_switcheroo_compiler.backends.edge._onnx_proto import (
    serialize_attribute,
    serialize_graph,
    serialize_model,
    serialize_node,
    serialize_tensor_proto,
    serialize_value_info,
)
from ml_switcheroo_compiler.backends.edge._onnx_wire import (
    _node_attrs_dict,
    _node_inputs_list,
    _node_shape_tuple,
    encode_int,
    encode_length_delimited,
    encode_string,
    validate_topological_sort,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _OnnxSerializerMixin(_OnnxGraphBuilderMixin):
    """Mixin implementing pure-Python binary protobuf serialization for ONNX graphs and models."""

    @staticmethod
    def _apply_dynamic_axes_to_shape(
        name: str,
        shape: list[int | str | None],
        dynamic_axes: dict[str, dict[int, str]] | None,
    ) -> list[int | str | None]:
        """Apply dynamic axis symbolic names to a tensor shape list.

        Args:
            name (str): Tensor identifier name.
            shape (list[int | str | None]): Mutable list of shape dimensions.
            dynamic_axes (dict[str, dict[int, str]] | None): Dynamic axes mapping.

        Returns:
            list[int | str | None]: Updated shape list with symbolic names applied.
        """
        if dynamic_axes and name in dynamic_axes:
            for axis_idx, axis_name in dynamic_axes[name].items():
                if axis_idx < len(shape):
                    shape[axis_idx] = axis_name
        return shape

    def _serialize_io_value_infos(
        self,
        dynamic_axes: dict[str, dict[int, str]] | None,
    ) -> tuple[list[bytes], list[bytes]]:
        """Serialize graph input and output ValueInfoProto byte lists.

        Args:
            dynamic_axes (dict[str, dict[int, str]] | None): Dynamic axis configuration.

        Returns:
            tuple[list[bytes], list[bytes]]: Pair of (inputs_bytes, outputs_bytes).
        """
        inputs_bytes: list[bytes] = []
        for n in self.sorted_nodes:
            if coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) == "Input":
                name: str = coerce_str(adapt_backend_module(n).get_symbol("id", ""))
                dt: str = coerce_str(adapt_backend_module(n).get_symbol("dtype", "float32"))
                shape: list[int | str | None] = list(_node_shape_tuple(n))
                self._apply_dynamic_axes_to_shape(name, shape, dynamic_axes)
                inputs_bytes.append(serialize_value_info(name, self._get_proto_type(dt), shape))

        outputs_bytes: list[bytes] = []
        raw_outs = adapt_backend_module(self.graph).get_symbol("outputs", []) or []
        for out_id in raw_outs if isinstance(raw_outs, (list, tuple)) else []:
            out_name = str(out_id)
            out_node = next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == out_name), None
            )
            dt = coerce_str(adapt_backend_module(out_node).get_symbol("dtype", "float32")) if out_node else "float32"
            shape = list(_node_shape_tuple(out_node)) if out_node else []
            self._apply_dynamic_axes_to_shape(out_name, shape, dynamic_axes)
            outputs_bytes.append(serialize_value_info(out_name, self._get_proto_type(dt), shape))

        return inputs_bytes, outputs_bytes

    def _serialize_single_ir_node(
        self,
        node: IRNode,
        sub_serializer: Callable[[IRGraph, str], bytes],
    ) -> bytes:
        """Serialize a single non-Input IRNode into NodeProto bytes.

        Args:
            node (IRNode): The IR node to serialize.
            sub_serializer (Callable[[IRGraph, str], bytes]): Callback for nested subgraphs.

        Returns:
            bytes: Serialized NodeProto wire bytes.
        """
        op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
        nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
        inputs: list[str] = _node_inputs_list(node)
        outputs: list[str] = [nid]
        attrs_dict: dict[str, BackendArg] = _node_attrs_dict(node)
        attributes: list[bytes] = []

        if op_type == "Constant":
            c_val = attrs_dict.get("value", 0.0)
            dt = coerce_str(adapt_backend_module(node).get_symbol("dtype", "float32"))
            shape_c = list(_node_shape_tuple(node))
            p_type = self._get_proto_type(dt)
            num_elem: int = math.prod(shape_c) if shape_c else 1
            raw_data: bytes = struct.pack(f"<{num_elem}f", *([float(coerce_float(c_val))] * num_elem))
            t_proto: bytes = serialize_tensor_proto(nid, p_type, shape_c, raw_data)
            attr_c: bytes = encode_string(1, "value") + encode_length_delimited(5, t_proto) + encode_int(20, 4)
            return serialize_node("Constant", [], outputs, nid, [attr_c])

        if op_type in ("If", "Cond"):
            for branch_key in ("then_branch", "else_branch"):
                branch_val = attrs_dict.get(branch_key)
                if branch_val is not None:
                    attributes.append(serialize_attribute(branch_key, branch_val, sub_serializer))
            return serialize_node("If", inputs, outputs, nid, attributes)

        if op_type in ("Loop", "WhileLoop"):
            body_b = attrs_dict.get("body")
            if body_b is not None:
                attributes.append(serialize_attribute("body", body_b, sub_serializer))
            return serialize_node("Loop", inputs, outputs, nid, attributes)

        ignored_keys = {"dtype", "shape", "is_state", "is_state_update", "param_name", "variable_name", "name"}
        for k, v in attrs_dict.items():
            if k not in ignored_keys:
                attributes.append(serialize_attribute(k, v, sub_serializer))
        return serialize_node(op_type, inputs, outputs, nid, attributes)

    def serialize_model_to_bytes(
        self,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 18,
    ) -> bytes:
        """Serialize the IR Graph into strictly compliant pure-Python ONNX binary protobuf bytes.

        Args:
            dynamic_axes (dict[str, dict[int, str]] | None): Dynamic axis configuration.
            opset_version (int): Target ONNX opset version (14 to 20).

        Returns:
            bytes: Valid ONNX binary protobuf byte sequence.

        Raises:
            ValueError: If opset_version is unsupported or topological sort fails.
        """
        import ml_switcheroo_compiler.backends.edge.onnx as _onnx_mod

        if not (14 <= opset_version <= 20):
            raise ValueError(f"Unsupported ONNX opset_version {opset_version}: must be between 14 and 20.")

        raw_inps = adapt_backend_module(self.graph).get_symbol("inputs", []) or []
        graph_inputs = [str(x) for x in raw_inps] if isinstance(raw_inps, (list, tuple)) else []
        validate_topological_sort(self.sorted_nodes, graph_inputs)
        inputs_bytes, outputs_bytes = self._serialize_io_value_infos(dynamic_axes)

        def sub_serializer(subgraph: IRGraph, sub_name: str) -> bytes:
            """Serialize nested subgraph to ONNX GraphProto bytes.

            Args:
                subgraph (IRGraph): The sub-graph to serialize.
                sub_name (str): Identifier name for the sub-graph.

            Returns:
                bytes: Serialized GraphProto.
            """
            subgen = _onnx_mod.ONNXCodeGenerator(subgraph)
            return subgen._serialize_graph_internal(sub_name, None)

        nodes_bytes: list[bytes] = [
            self._serialize_single_ir_node(node, sub_serializer)
            for node in self.sorted_nodes
            if coerce_str(adapt_backend_module(node).get_symbol("op_type", "")) != "Input"
        ]

        graph_payload: bytes = serialize_graph("ml_switcheroo_graph", nodes_bytes, inputs_bytes, outputs_bytes)
        return serialize_model(
            graph_payload, producer_name="ml-switcheroo-compiler", opset_version=opset_version, ir_version=8
        )

    def _serialize_graph_internal(self, graph_name: str, dynamic_axes: dict[str, dict[int, str]] | None) -> bytes:
        """Serialize an internal subgraph into ONNX GraphProto wire bytes.

        Args:
            graph_name (str): Name identifier for the subgraph.
            dynamic_axes (dict[str, dict[int, str]] | None): Optional dynamic axis mapping.

        Returns:
            bytes: Serialized GraphProto wire bytes.
        """
        inputs_bytes: list[bytes] = []
        for n in self.sorted_nodes:
            if coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) == "Input":
                name: str = coerce_str(adapt_backend_module(n).get_symbol("id", ""))
                dt: str = coerce_str(adapt_backend_module(n).get_symbol("dtype", "float32"))
                shape: list[int | str | None] = list(_node_shape_tuple(n))
                p_type: int = self._get_proto_type(dt)
                inputs_bytes.append(serialize_value_info(name, p_type, shape))

        outputs_bytes: list[bytes] = []
        raw_outs = adapt_backend_module(self.graph).get_symbol("outputs", []) or []
        for out_id in raw_outs if isinstance(raw_outs, (list, tuple)) else []:
            out_node = next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == str(out_id)), None
            )
            dt = coerce_str(adapt_backend_module(out_node).get_symbol("dtype", "float32")) if out_node else "float32"
            shape = list(_node_shape_tuple(out_node)) if out_node else []
            p_type = self._get_proto_type(dt)
            outputs_bytes.append(serialize_value_info(str(out_id), p_type, shape))

        nodes_bytes: list[bytes] = []
        for node in self.sorted_nodes:
            op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
            if op_type == "Input":
                continue
            nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
            inputs: list[str] = _node_inputs_list(node)
            outputs: list[str] = [nid]
            attributes: list[bytes] = []
            for k, v in _node_attrs_dict(node).items():
                if k in ("dtype", "shape", "is_state", "is_state_update", "param_name", "variable_name", "name"):
                    continue
                attributes.append(serialize_attribute(k, v))
            nodes_bytes.append(serialize_node(op_type, inputs, outputs, nid, attributes))

        return serialize_graph(graph_name, nodes_bytes, inputs_bytes, outputs_bytes)
