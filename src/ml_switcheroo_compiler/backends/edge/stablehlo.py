# ruff: noqa: E501
"""StableHLO edge code generator."""

import uuid
from typing import Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


class StableHLOCodeGenerator(BaseGenerator):
    """StableHLO Code Generator for emitting MLIR text format from IR Graph."""

    def __init__(self, graph: IRGraph, delegates: Optional[list] = None) -> None:
        """Initialize StableHLOCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list, optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}

    def _map_type(self, shape: tuple[int, ...], dtype: str) -> str:
        """Map shape and dtype to StableHLO tensor type string.

        Args:
            shape (tuple[int, ...]): The shape of the tensor.
            dtype (str): The data type.

        Returns:
            str: StableHLO tensor type representation.
        """
        dt = {
            "float32": "f32",
            "float64": "f64",
            "int32": "i32",
            "bool": "i1",
        }.get(str(dtype).lower(), "f32")
        if not shape:
            return f"tensor<{dt}>"
        shape_str = "x".join(str(s) for s in shape)
        return f"tensor<{shape_str}x{dt}>"

    def _get_node_type(self, node: object) -> str:
        """Extract the type mapping for a given node.

        Args:
            node (object): The IR node.

        Returns:
            str: StableHLO tensor type string.
        """
        meta_shape = getattr(node, "shape_metadata", ()) or ()
        meta_dtype = getattr(node, "attributes", {}).get("dtype", getattr(node, "dtype", "float32"))
        return self._map_type(meta_shape, meta_dtype)

    def _resolve_input_types(self, node: object, out_type: str) -> list[str]:
        """Resolve the input types for a given node.

        Args:
            node (object): The IR node.
            out_type (str): The output type to use as a fallback.

        Returns:
            list[str]: A list of input type strings.
        """
        in_types = []
        for inp in getattr(node, "inputs", []):
            in_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None)
            if in_node:
                in_types.append(self._get_node_type(in_node))
            else:
                in_types.append(out_type)
        return in_types

    def _emit_constant(self, node: object, nid: str) -> str:
        """Emit a StableHLO constant operation.

        Args:
            node (object): The IR node representing the constant.
            nid (str): The node ID.

        Returns:
            str: The generated variable name for the constant.
        """
        val = node.attributes.get("value", 0.0)
        meta_shape = getattr(node, "shape_metadata", ()) or ()
        t_type = self._get_node_type(node)
        res_var = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var
        dense_val = f"dense<{val}>" if meta_shape else str(val)
        self.add_line(f'  {res_var} = "stablehlo.constant"() {{value = {dense_val} : {t_type}}} : () -> {t_type}')
        return res_var

    def generic_visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Process a node and return its generated code name.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        op_type = getattr(node, "op_type", "")
        nid = getattr(node, "id", str(uuid.uuid4()))

        if op_type == "Input":
            arg_idx = len(self.var_map)
            arg_name = f"%arg{arg_idx}"
            self.var_map[nid] = arg_name
            return arg_name

        if op_type == "Constant":
            return self._emit_constant(node, nid)

        op_map = {
            "Add": "stablehlo.add",
            "Subtract": "stablehlo.subtract",
            "Multiply": "stablehlo.multiply",
            "TrueDivide": "stablehlo.divide",
            "Div": "stablehlo.divide",
            "Exp": "stablehlo.exponential",
            "Log": "stablehlo.log",
            "Negative": "stablehlo.negate",
            "Neg": "stablehlo.negate",
        }

        hlo_op = op_map.get(op_type, "stablehlo.custom_call")
        res_var = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var
        out_type = self._get_node_type(node)
        in_vars_mapped = [self.var_map.get(inp, inp) for inp in getattr(node, "inputs", [])]
        in_types = self._resolve_input_types(node, out_type)

        inputs_str = ", ".join(in_vars_mapped)
        types_signature = f"({', '.join(in_types)}) -> {out_type}"

        if hlo_op == "stablehlo.custom_call":
            self.add_line(f'  {res_var} = "stablehlo.custom_call"({inputs_str}) {{call_target_name = "{op_type}"}} : {types_signature}')
        else:
            self.add_line(f'  {res_var} = "{hlo_op}"({inputs_str}) : {types_signature}')

        return res_var

    def _build_func_args(self, input_nodes: list[object]) -> list[str]:
        """Build the list of function arguments for the generated MLIR module.

        Args:
            input_nodes (list[object]): List of input IR nodes.

        Returns:
            list[str]: A list of argument strings for the MLIR function.
        """
        func_args = []
        for idx, node in enumerate(input_nodes):
            t_type = self._get_node_type(node)
            arg_name = f"%arg{idx}"
            self.var_map[getattr(node, "id", "")] = arg_name
            func_args.append(f"{arg_name}: {t_type}")
        return func_args

    def _build_out_types(self, output_ids: list[str]) -> list[str]:
        """Build the list of output types for the generated MLIR module.

        Args:
            output_ids (list[str]): List of output node IDs.

        Returns:
            list[str]: A list of return type strings for the MLIR function.
        """
        out_types = []
        for out_id in output_ids:
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            if out_node:
                out_types.append(self._get_node_type(out_node))
            else:
                out_types.append("tensor<f32>")
        return out_types

    def _get_returns_str(self, out_types: list[str]) -> str:
        """Format the return types signature.

        Args:
            out_types (list[str]): List of return types.

        Returns:
            str: Formatted return types string.
        """
        if not out_types:
            return "tensor<f32>"
        if len(out_types) == 1:
            return out_types[0]
        return ", ".join(out_types)

    def _get_ret_vars_str(self, out_vars: list[str]) -> str:
        """Format the return variables string.

        Args:
            out_vars (list[str]): List of return variable names.

        Returns:
            str: Formatted return variables string.
        """
        if not out_vars:
            return ""
        if len(out_vars) == 1:
            return out_vars[0]
        return ", ".join(out_vars)

    def generate(self) -> str:
        """Generate StableHLO MLIR text representation of the IR Graph.

        Returns:
            str: Generated StableHLO MLIR text.
        """
        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        func_args = self._build_func_args(input_nodes)

        output_ids = getattr(self.graph, "outputs", []) or []
        out_types = self._build_out_types(output_ids)

        args_str = ", ".join(func_args)
        returns_str = self._get_returns_str(out_types)

        self.code = []
        self.add_line("module @jit_fun {")
        self.add_line(f"  func.func @main({args_str}) -> {returns_str} {{")

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") != "Input":
                self.generic_visit(node, [])

        out_vars = [self.var_map.get(out_id, out_id) for out_id in output_ids]
        ret_vars_str = self._get_ret_vars_str(out_vars)
        ret_types_str = f" : {returns_str}" if returns_str else ""
        self.add_line(f"    return {ret_vars_str}{ret_types_str}")

        self.add_line("  }")
        self.add_line("}")

        return "\n".join(self.code)
