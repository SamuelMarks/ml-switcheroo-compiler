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
            val = node.attributes.get("value", 0.0)
            meta_shape = getattr(node, "shape_metadata", ()) or ()
            meta_dtype = getattr(node, "attributes", {}).get("dtype", getattr(node, "dtype", "float32"))
            t_type = self._map_type(meta_shape, meta_dtype)
            res_var = f"%v_{nid.replace('-', '_')}"
            self.var_map[nid] = res_var
            dense_val = f"dense<{val}>" if meta_shape else str(val)
            self.add_line(f'  {res_var} = "stablehlo.constant"() {{value = {dense_val} : {t_type}}} : () -> {t_type}')
            return res_var

        # Map standard binary/unary math ops to StableHLO dialect equivalents
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

        meta_shape = getattr(node, "shape_metadata", ()) or ()
        meta_dtype = getattr(node, "attributes", {}).get("dtype", getattr(node, "dtype", "float32"))
        out_type = self._map_type(meta_shape, meta_dtype)

        in_vars_mapped = [self.var_map.get(inp, inp) for inp in getattr(node, "inputs", [])]
        in_types = []
        for inp in getattr(node, "inputs", []):
            # Try to resolve input shapes
            in_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None)
            if in_node:
                in_shape = getattr(in_node, "shape_metadata", ()) or ()
                in_dtype = getattr(in_node, "attributes", {}).get("dtype", getattr(in_node, "dtype", "float32"))
                in_types.append(self._map_type(in_shape, in_dtype))
            else:
                in_types.append(out_type)

        inputs_str = ", ".join(in_vars_mapped)
        types_signature = f"({', '.join(in_types)}) -> {out_type}"

        if hlo_op == "stablehlo.custom_call":
            self.add_line(f'  {res_var} = "stablehlo.custom_call"({inputs_str}) {{call_target_name = "{op_type}"}} : {types_signature}')
        else:
            self.add_line(f'  {res_var} = "{hlo_op}"({inputs_str}) : {types_signature}')

        return res_var

    def generate(self) -> str:
        """Generate StableHLO MLIR text representation of the IR Graph.

        Returns:
            str: Generated StableHLO MLIR text.
        """
        # Collect function input nodes
        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        func_args = []
        for idx, node in enumerate(input_nodes):
            meta_shape = getattr(node, "shape_metadata", ()) or ()
            meta_dtype = getattr(node, "attributes", {}).get("dtype", getattr(node, "dtype", "float32"))
            t_type = self._map_type(meta_shape, meta_dtype)
            arg_name = f"%arg{idx}"
            self.var_map[getattr(node, "id", "")] = arg_name
            func_args.append(f"{arg_name}: {t_type}")

        # Map output types
        output_ids = getattr(self.graph, "outputs", []) or []
        out_types = []
        for out_id in output_ids:
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            if out_node:
                out_shape = getattr(out_node, "shape_metadata", ()) or ()
                out_dtype = getattr(out_node, "attributes", {}).get("dtype", getattr(out_node, "dtype", "float32"))
                out_types.append(self._map_type(out_shape, out_dtype))
            else:
                out_types.append("tensor<f32>")

        args_str = ", ".join(func_args)
        returns_str = ", ".join(out_types) if len(out_types) > 1 else out_types[0] if out_types else "tensor<f32>"

        # Start module and function
        self.code = []
        self.add_line("module @jit_fun {")
        self.add_line(f"  func.func @main({args_str}) -> {returns_str} {{")

        # Emit all operations
        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") != "Input":
                self.generic_visit(node, [])

        # Emit return statement
        out_vars = [self.var_map.get(out_id, out_id) for out_id in output_ids]
        ret_vars_str = ", ".join(out_vars) if len(out_vars) > 1 else out_vars[0] if out_vars else ""
        ret_types_str = f" : {returns_str}" if returns_str else ""
        self.add_line(f"    return {ret_vars_str}{ret_types_str}")

        self.add_line("  }")
        self.add_line("}")

        return "\n".join(self.code)
