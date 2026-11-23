"""ONNX GraphProto builder mixin using the official `onnx.helper` API and text fallback."""

from __future__ import annotations

import math
import os
import typing

import yaml
from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._onnx_wire import (
    GraphProtoType,
    TensorProtoType,
    _node_inputs_list,
    _node_shape_tuple,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs
from ml_switcheroo_compiler.ir.core import ensure_graph


def _is_graph_proto(val: BackendArg) -> typing.TypeGuard[GraphProtoType]:
    """Check whether `val` exposes an ONNX `GraphProto` `.name` attribute.

    Args:
        val (BackendArg): Candidate graph protobuf or mock object.

    Returns:
        typing.TypeGuard[GraphProtoType]: `True` if `val` is a `GraphProto`.
    """
    found: bool = isinstance(val, GraphProtoType)
    return found or "name" in dir(val) or is_mock_with_attrs(val, "name")


class _OnnxGraphBuilderMixin(BaseGenerator):
    """Mixin providing ONNX schema loading, text-proto fallback, and `onnx.helper` graph construction."""

    def __init__(self, graph: IRGraph, delegates: list[BaseGenerator] | BackendArg | None = None) -> None:
        """Initialize the ONNX generator state and load `onnx_schema.yaml`.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list[BaseGenerator] | BackendArg | None): Visitor delegates.
        """
        import ml_switcheroo_compiler.backends.edge.onnx as _onnx_mod

        super().__init__(graph, list(delegates) if isinstance(delegates, list) else None)
        self.var_map: dict[str, str] = {}

        yaml_path: str = os.path.join(os.path.dirname(_onnx_mod.__file__), "onnx_schema.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = {}

    def _get_proto_type(self, dt: str, TensorProto: type[TensorProtoType] | None = None) -> int:
        """Map data type string to ONNX TensorProto primitive integer code.

        Args:
            dt (str): The dtype string.
            TensorProto (type[TensorProtoType] | None): Optional ONNX TensorProto namespace.

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
        lines: list[str] = ["ir_version: 7", 'producer_name: "ml-switcheroo-compiler"', "graph ml_switcheroo_graph {"]

        for node in self.sorted_nodes:
            if coerce_str(adapt_backend_module(node).get_symbol("op_type", "")) == "Input":
                nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
                shape: str = "x".join(str(s) for s in _node_shape_tuple(node))
                dtype: str = coerce_str(adapt_backend_module(node).get_symbol("dtype", "float32"))
                lines.append(f'  input: "{nid}" [shape: {shape}, dtype: {dtype}]')

        for node in self.sorted_nodes:
            op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
            if op_type != "Input":
                nid = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
                inps: str = ", ".join(f'"{i}"' for i in _node_inputs_list(node))
                lines.append(f'  "{nid}" = {op_type}({inps})')

        raw_outs = adapt_backend_module(self.graph).get_symbol("outputs", []) or []
        for out_id in raw_outs if isinstance(raw_outs, (list, tuple)) else []:
            lines.append(f'  output: "{out_id}"')

        lines.append("}")
        return "\n".join(lines)

    def _get_node_and_name(self, item: IRNode | str, is_output: bool) -> tuple[IRNode | None, str]:
        """Retrieve a node object and its ID name.

        Args:
            item (IRNode | str): The IR node or its ID string.
            is_output (bool): True if looking up an output node by ID.

        Returns:
            tuple[IRNode | None, str]: A tuple containing the node and its name string.
        """
        if is_output:
            out_id: str = str(item)
            node = next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == out_id), None
            )
            return node, out_id
        if isinstance(item, IRNode):
            return item, str(adapt_backend_module(item).get_symbol("id", ""))
        node = next((n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == str(item)), None)
        return node, str(item)

    def _build_single_value_info(
        self,
        item: IRNode | str,
        dynamic_axes: dict[str, dict[int, str]] | None,
        TensorProto: type[TensorProtoType],
        is_output: bool,
    ) -> BackendArg:
        """Construct an ONNX TensorValueInfoProto for a single node.

        Args:
            item (IRNode | str): The IR node or output ID.
            dynamic_axes (dict[str, dict[int, str]] | None): Dynamic axis mapping configuration.
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.
            is_output (bool): True if building for an output.

        Returns:
            BackendArg: An ONNX ValueInfoProto object.
        """
        from onnx import helper

        node, name = self._get_node_and_name(item, is_output)
        shape: tuple[int, ...] = _node_shape_tuple(node) if node else ()
        dt: str = coerce_str(adapt_backend_module(node).get_symbol("dtype", "float32")) if node else "float32"
        proto_type: int = self._get_proto_type(dt, TensorProto)
        shape_list: list[int | str] = list(shape)

        if dynamic_axes and name in dynamic_axes:
            for axis_idx, axis_name in dynamic_axes[name].items():
                shape_list[axis_idx] = axis_name
        from ml_switcheroo_compiler.backends.eager_registry import require_callable

        return require_callable(helper.make_tensor_value_info)(name, proto_type, shape_list)

    def _build_onnx_value_infos(
        self,
        nodes_or_ids: list[IRNode | str],
        dynamic_axes: dict[str, dict[int, str]] | None,
        TensorProto: type[TensorProtoType],
        is_output: bool = False,
    ) -> list[BackendArg]:
        """Construct a list of ONNX TensorValueInfoProtos.

        Args:
            nodes_or_ids (list[IRNode | str]): List of IR nodes or output IDs.
            dynamic_axes (dict[str, dict[int, str]] | None): Dynamic axis mapping configuration.
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.
            is_output (bool): True if building for outputs.

        Returns:
            list[BackendArg]: A list of ONNX ValueInfoProto objects.
        """
        return [self._build_single_value_info(item, dynamic_axes, TensorProto, is_output) for item in nodes_or_ids]

    def _resolve_onnx_op_name(self, op_type: str) -> str:
        """Map an IR operation type to its target ONNX opcode.

        Args:
            op_type (str): Source IR operation type name.

        Returns:
            str: Target ONNX opcode name.

        Raises:
            BackendNotSupportedError: If the operation is not supported in the ONNX schema.
        """
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        op_def = OPS_REGISTRY.get(op_type, {})
        variants = op_def.get("variants", {})
        if "edge_onnx" in variants:
            gen: str | None = variants["edge_onnx"].get("generator")
            if gen:
                return gen
        if op_type not in self.schema.get("operations", {}):
            from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

            raise BackendNotSupportedError(f"Operation '{op_type}' not supported in ONNX schema.")
        return str(op_type)

    def _build_subgraph_node_kwargs(self, node: IRNode, nid: str, op_type: str) -> dict[str, GraphProtoType]:
        """Build subgraph attribute keyword arguments for control-flow ONNX nodes.

        Args:
            node (IRNode): The control-flow IR node.
            nid (str): Node identifier string.
            op_type (str): Operation type ("If", "Loop", or "WhileLoop").

        Returns:
            dict[str, GraphProtoType]: Mapping of branch attribute names to ONNX GraphProtos.
        """
        import ml_switcheroo_compiler.backends.edge.onnx as _onnx_mod

        kwargs: dict[str, GraphProtoType] = {}
        if op_type == "If":
            for branch_key, suffix in (("then_branch", "then"), ("else_branch", "else")):
                if branch_key in node.attributes:
                    subgen = _onnx_mod.ONNXCodeGenerator(ensure_graph(node.attributes[branch_key]))
                    g_proto = subgen._build_onnx_graph(None)
                    g_proto.name = f"{nid}_{suffix}"
                    kwargs[branch_key] = g_proto
        elif op_type in ("Loop", "WhileLoop") and "body" in node.attributes:
            subgen = _onnx_mod.ONNXCodeGenerator(ensure_graph(node.attributes["body"]))
            g_proto = subgen._build_onnx_graph(None)
            g_proto.name = f"{nid}_body"
            kwargs["body"] = g_proto
        return kwargs

    def _build_onnx_nodes(self, TensorProto: type[TensorProtoType]) -> list[BackendArg]:
        """Construct all intermediate ONNX NodeProtos for the graph.

        Args:
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.

        Returns:
            list[BackendArg]: A list of ONNX NodeProto objects.
        """
        from onnx import helper

        onnx_nodes: list[BackendArg] = []

        for node in self.sorted_nodes:
            op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
            if op_type == "Input":
                continue

            nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
            inputs: list[str] = _node_inputs_list(node)

            if op_type == "Constant":
                val = float(coerce_float(node.attributes.get("value", 0.0)))
                dt: str = coerce_str(adapt_backend_module(node).get_symbol("dtype", "float32"))
                shape: tuple[int, ...] = _node_shape_tuple(node)
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
                onnx_op: str = self._resolve_onnx_op_name(op_type)
                kwargs = self._build_subgraph_node_kwargs(node, nid, op_type)
                onnx_nodes.append(helper.make_node(onnx_op, inputs=inputs, outputs=[nid], name=nid, **kwargs))
        return onnx_nodes

    def _build_onnx_graph(self, dynamic_axes: dict[str, dict[int, str]] | None = None) -> GraphProtoType:
        """Construct the full ONNX GraphProto.

        Args:
            dynamic_axes (dict[str, dict[int, str]] | None): Dynamic axis mapping configuration.

        Returns:
            GraphProtoType: The ONNX GraphProto object.
        """
        from onnx import TensorProto, helper

        input_nodes = [
            n for n in self.sorted_nodes if coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) == "Input"
        ]
        build_fn = adapt_backend_module(self).require_op("_build_onnx_value_infos")
        onnx_inputs = build_fn(input_nodes, dynamic_axes, TensorProto, is_output=False)

        raw_outs = adapt_backend_module(self.graph).get_symbol("outputs", []) or []
        output_ids: list[str] = [str(x) for x in raw_outs] if isinstance(raw_outs, (list, tuple)) else []
        onnx_outputs = build_fn(output_ids, dynamic_axes, TensorProto, is_output=True)

        onnx_nodes = self._build_onnx_nodes(TensorProto)
        graph_proto: BackendArg = helper.make_graph(onnx_nodes, "ml_switcheroo_graph", onnx_inputs, onnx_outputs)
        assert _is_graph_proto(graph_proto)
        return graph_proto
