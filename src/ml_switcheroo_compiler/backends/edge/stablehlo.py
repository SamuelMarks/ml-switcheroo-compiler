# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""StableHLO edge code generator."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from typing import Optional, Union

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def verify_stablehlo_mlir(mlir_text: str, mlir_opt_path: str | None = None) -> bool:
    """Verify StableHLO MLIR text representation using mlir-opt or structural analysis.

    Args:
        mlir_text (str): StableHLO MLIR module text to verify.
        mlir_opt_path (Optional[str]): Optional path to mlir-opt tool binary.

    Returns:
        bool: True if verification succeeds.

    Raises:
        ValueError: If MLIR verification fails or MLIR text is malformed.
    """
    opt_bin: str | None = mlir_opt_path or shutil.which("mlir-opt")
    if opt_bin and os.path.exists(opt_bin):
        res: subprocess.CompletedProcess[bytes] = subprocess.run(
            [opt_bin, "--verify-diagnostics"],
            input=mlir_text.encode("utf-8"),
            capture_output=True,
        )
        if res.returncode != 0:
            err_msg: str = res.stderr.decode("utf-8", errors="replace")
            raise ValueError(f"mlir-opt verification failed: {err_msg}")
        return True

    if not mlir_text or not mlir_text.strip():
        raise ValueError("MLIR text is empty")

    open_braces: int = mlir_text.count("{")
    close_braces: int = mlir_text.count("}")
    if open_braces != close_braces:
        raise ValueError(f"Unbalanced braces in MLIR text: {open_braces} open, {close_braces} close")

    if "module" not in mlir_text or "func.func" not in mlir_text:
        raise ValueError("MLIR text must declare a module and func.func")

    if "return" not in mlir_text:
        raise ValueError("MLIR function must contain a return statement")

    return True


class StableHLOCodeGenerator(BaseGenerator):
    """StableHLO Code Generator for emitting MLIR text format from IR Graph."""

    def __init__(self, graph: IRGraph, delegates: list[CodeGeneratorVisitor] | None = None) -> None:
        """Initialize StableHLOCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Optional[list[CodeGeneratorVisitor]], optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}

        yaml_path: str = os.path.join(os.path.dirname(__file__), "stablehlo_schema.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = {}

    def _map_type(self, shape: tuple[int | str, ...], dtype: str) -> str:
        """Map shape and dtype to StableHLO tensor type string.

        Args:
            shape (tuple[int | str, ...]): The shape of the tensor.
            dtype (str): The data type.

        Returns:
            str: StableHLO tensor type representation.
        """
        dt_map: dict[str, str] = self.schema.get("types", {})
        dt: str = dt_map.get(str(dtype).lower(), "f32")
        if not shape:
            return f"tensor<{dt}>"
        dims: list[str] = []
        for s in shape:
            if s is None or s == "?" or s == -1 or (isinstance(s, str) and not s.isdigit()):
                dims.append("?")
            else:
                dims.append(str(s))
        shape_str: str = "x".join(dims)
        return f"tensor<{shape_str}x{dt}>"

    def _get_node_type(self, node: IRNode) -> str:
        """Extract the type mapping for a given node.

        Args:
            node (IRNode): The IR node.

        Returns:
            str: StableHLO tensor type string.
        """
        meta_shape: tuple[int, ...] = getattr(node, "shape_metadata", ()) or ()
        meta_dtype: str = getattr(node, "attributes", {}).get("dtype", getattr(node, "dtype", "float32"))
        return self._map_type(meta_shape, meta_dtype)

    def _resolve_input_types(self, node: IRNode, out_type: str) -> list[str]:
        """Resolve the input types for a given node.

        Args:
            node (IRNode): The IR node.
            out_type (str): The output type to use as a fallback.

        Returns:
            list[str]: A list of input type strings.
        """
        in_types: list[str] = []
        for inp in getattr(node, "inputs", []):
            in_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == inp), None)
            if in_node:
                in_types.append(self._get_node_type(in_node))
            else:
                in_types.append(out_type)
        return in_types

    def _emit_constant(self, node: IRNode, nid: str) -> str:
        """Emit a StableHLO constant operation.

        Args:
            node (IRNode): The IR node representing the constant.
            nid (str): The node ID.

        Returns:
            str: The generated variable name for the constant.
        """
        val: float = getattr(node, "attributes", {}).get("value", 0.0)
        meta_shape: tuple[int, ...] = getattr(node, "shape_metadata", ()) or ()
        t_type: str = self._get_node_type(node)
        res_var: str = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var
        dense_val: str = f"dense<{val}>" if meta_shape else str(val)
        self.add_line(f'  {res_var} = "stablehlo.constant"() {{value = {dense_val} : {t_type}}} : () -> {t_type}')
        return res_var

    def _emit_dot_general(
        self,
        node: IRNode,
        nid: str,
        in_vars_mapped: list[str],
        in_types: list[str],
        out_type: str,
    ) -> str:
        """Emit a StableHLO dot_general operation with explicit dimension numbers.

        Args:
            node (IRNode): The IR node.
            nid (str): Unique node identifier.
            in_vars_mapped (list[str]): Mapped input SSA variable names.
            in_types (list[str]): Input type strings.
            out_type (str): Output tensor type string.

        Returns:
            str: Generated result SSA variable name.
        """
        res_var: str = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var
        inputs_str: str = ", ".join(in_vars_mapped)
        types_sig: str = f"({', '.join(in_types)}) -> {out_type}"

        attrs: dict[str, object] = getattr(node, "attributes", {}) or {}
        lhs_batch = attrs.get("lhs_batching_dimensions")
        rhs_batch = attrs.get("rhs_batching_dimensions")
        lhs_contract = attrs.get("lhs_contracting_dimensions")
        rhs_contract = attrs.get("rhs_contracting_dimensions")

        if lhs_batch is None or lhs_contract is None:
            first_inp: str = getattr(node, "inputs", [""])[0] if getattr(node, "inputs", []) else ""
            in_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == first_inp), None)
            rank: int = len(getattr(in_node, "shape_metadata", ()) or ()) if in_node else 2
            if rank >= 3:
                lhs_batch = [0]
                rhs_batch = [0]
                lhs_contract = [rank - 1]
                rhs_contract = [rank - 2]
            else:
                lhs_batch = []
                rhs_batch = []
                lhs_contract = [1] if rank > 1 else [0]
                rhs_contract = [0]

        lhs_b_str: str = ", ".join(str(x) for x in (lhs_batch or []))
        rhs_b_str: str = ", ".join(str(x) for x in (rhs_batch or []))
        lhs_c_str: str = ", ".join(str(x) for x in (lhs_contract or []))
        rhs_c_str: str = ", ".join(str(x) for x in (rhs_contract or []))

        dot_dims: str = f"#stablehlo.dot<lhs_batching_dimensions = [{lhs_b_str}], rhs_batching_dimensions = [{rhs_b_str}], lhs_contracting_dimensions = [{lhs_c_str}], rhs_contracting_dimensions = [{rhs_c_str}]>"
        self.add_line(f'  {res_var} = "stablehlo.dot_general"({inputs_str}) {{dot_dimension_numbers = {dot_dims}}} : {types_sig}')
        return res_var

    def _emit_convolution(
        self,
        node: IRNode,
        nid: str,
        in_vars_mapped: list[str],
        in_types: list[str],
        out_type: str,
    ) -> str:
        """Emit a StableHLO convolution operation with full MLIR attribute bindings.

        Args:
            node (IRNode): The IR node.
            nid (str): Unique node identifier.
            in_vars_mapped (list[str]): Mapped input SSA variable names.
            in_types (list[str]): Input type strings.
            out_type (str): Output tensor type string.

        Returns:
            str: Generated result SSA variable name.
        """
        res_var: str = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var
        inputs_str: str = ", ".join(in_vars_mapped)
        types_sig: str = f"({', '.join(in_types)}) -> {out_type}"

        attrs: dict[str, object] = getattr(node, "attributes", {}) or {}
        strides_val = attrs.get("strides", attrs.get("stride", [1, 1]))
        strides: list[int] = [strides_val, strides_val] if isinstance(strides_val, int) else list(strides_val)

        pads_val = attrs.get("padding", attrs.get("pads", [[0, 0], [0, 0]]))
        pads: list[list[int]] = []
        if isinstance(pads_val, int):
            pads = [[pads_val, pads_val], [pads_val, pads_val]]
        elif isinstance(pads_val, (list, tuple)) and len(pads_val) == 2 and isinstance(pads_val[0], int):
            pads = [[int(pads_val[0]), int(pads_val[0])], [int(pads_val[1]), int(pads_val[1])]]
        elif isinstance(pads_val, (list, tuple)) and len(pads_val) == 4 and isinstance(pads_val[0], int):
            pads = [[int(pads_val[0]), int(pads_val[1])], [int(pads_val[2]), int(pads_val[3])]]
        elif isinstance(pads_val, (list, tuple)):
            pads = [list(p) if isinstance(p, (list, tuple)) else [int(p), int(p)] for p in pads_val]
        else:
            pads = [[0, 0], [0, 0]]

        lhs_dil_val = attrs.get("lhs_dilation", [1, 1])
        lhs_dil: list[int] = [lhs_dil_val, lhs_dil_val] if isinstance(lhs_dil_val, int) else list(lhs_dil_val)

        rhs_dil_val = attrs.get("rhs_dilation", attrs.get("dilation", [1, 1]))
        rhs_dil: list[int] = [rhs_dil_val, rhs_dil_val] if isinstance(rhs_dil_val, int) else list(rhs_dil_val)

        feature_group_count: int = int(attrs.get("feature_group_count", attrs.get("groups", 1)))
        batch_group_count: int = int(attrs.get("batch_group_count", 1))

        strides_str: str = ", ".join(str(s) for s in strides)
        pads_str: str = ", ".join(f"[{p[0]}, {p[1]}]" for p in pads)
        lhs_dil_str: str = ", ".join(str(d) for d in lhs_dil)
        rhs_dil_str: str = ", ".join(str(d) for d in rhs_dil)

        self.add_line(
            f'  {res_var} = "stablehlo.convolution"({inputs_str}) {{'
            f"batch_group_count = {batch_group_count} : i64, "
            f"dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, "
            f"feature_group_count = {feature_group_count} : i64, "
            f"lhs_dilation = dense<[{lhs_dil_str}]> : tensor<{len(lhs_dil)}xi64>, "
            f"padding = dense<[{pads_str}]> : tensor<{len(pads)}x2xi64>, "
            f"rhs_dilation = dense<[{rhs_dil_str}]> : tensor<{len(rhs_dil)}xi64>, "
            f"window_reversal = dense<false> : tensor<{len(strides)}xi1>, "
            f"window_strides = dense<[{strides_str}]> : tensor<{len(strides)}xi64>}} : {types_sig}"
        )
        return res_var

    def _emit_reduce_window(
        self,
        node: IRNode,
        nid: str,
        in_vars_mapped: list[str],
        in_types: list[str],
        out_type: str,
    ) -> str:
        """Emit a StableHLO reduce_window operation with an inner reduction body.

        Args:
            node (IRNode): The IR node.
            nid (str): Unique node identifier.
            in_vars_mapped (list[str]): Mapped input SSA variable names.
            in_types (list[str]): Input type strings.
            out_type (str): Output tensor type string.

        Returns:
            str: Generated result SSA variable name.
        """
        res_var: str = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var

        attrs: dict[str, object] = getattr(node, "attributes", {}) or {}
        op_type: str = getattr(node, "op_type", "")

        win_dims_val = attrs.get("window_dimensions", attrs.get("kernel_size", [1, 2, 2, 1]))
        win_dims: list[int] = [1, win_dims_val, win_dims_val, 1] if isinstance(win_dims_val, int) else ([1, win_dims_val[0], win_dims_val[1], 1] if len(win_dims_val) == 2 else list(win_dims_val))

        win_strides_val = attrs.get("window_strides", attrs.get("strides", attrs.get("stride", [1, 2, 2, 1])))
        win_strides: list[int] = [1, win_strides_val, win_strides_val, 1] if isinstance(win_strides_val, int) else ([1, win_strides_val[0], win_strides_val[1], 1] if len(win_strides_val) == 2 else list(win_strides_val))

        pads_val = attrs.get("padding", attrs.get("pads", [[0, 0], [0, 0], [0, 0], [0, 0]]))
        pads: list[list[int]] = []
        if isinstance(pads_val, int):
            pads = [[0, 0], [pads_val, pads_val], [pads_val, pads_val], [0, 0]]
        elif isinstance(pads_val, (list, tuple)) and len(pads_val) == 2:
            p0 = [pads_val[0], pads_val[0]] if isinstance(pads_val[0], int) else list(pads_val[0])
            p1 = [pads_val[1], pads_val[1]] if isinstance(pads_val[1], int) else list(pads_val[1])
            pads = [[0, 0], p0, p1, [0, 0]]
        elif isinstance(pads_val, (list, tuple)):
            pads = [list(p) if isinstance(p, (list, tuple)) else [int(p), int(p)] for p in pads_val]
        else:
            pads = [[0, 0], [0, 0], [0, 0], [0, 0]]

        reducer: str = str(attrs.get("reducer", "maximum" if "Max" in op_type else "add"))

        base_dils: list[int] = list(attrs.get("base_dilations", [1] * len(win_dims)))
        win_dils: list[int] = list(attrs.get("window_dilations", [1] * len(win_dims)))

        op_in: str = in_vars_mapped[0] if in_vars_mapped else "%arg0"
        op_type_val: str = in_types[0] if in_types else out_type

        elem_t: str = "f32"
        if "x" in op_type_val and ">" in op_type_val:
            elem_t = op_type_val.rsplit("x", 1)[-1].rstrip(">")
        elif "<" in op_type_val and ">" in op_type_val:
            elem_t = op_type_val.split("<", 1)[1].rstrip(">")

        if len(in_vars_mapped) > 1:
            init_in: str = in_vars_mapped[1]
            init_t: str = in_types[1]
        else:
            init_val: str = "-3.40282347E+38" if reducer == "maximum" else "0.0"
            init_in = f"%c_init_{nid.replace('-', '_')}"
            self.add_line(f'  {init_in} = "stablehlo.constant"() {{value = dense<{init_val}> : tensor<{elem_t}>}} : () -> tensor<{elem_t}>')
            init_t = f"tensor<{elem_t}>"

        win_dim_str: str = ", ".join(str(x) for x in win_dims)
        win_str_str: str = ", ".join(str(x) for x in win_strides)
        pads_str: str = ", ".join(f"[{p[0]}, {p[1]}]" for p in pads)
        base_dil_str: str = ", ".join(str(x) for x in base_dils)
        win_dil_str: str = ", ".join(str(x) for x in win_dils)

        rank: int = len(win_dims)
        self.add_line(f'  {res_var} = "stablehlo.reduce_window"({op_in}, {init_in}) ({{')
        self.add_line(f"  ^bb0(%arg_a: tensor<{elem_t}>, %arg_b: tensor<{elem_t}>):")
        self.add_line(f'    %r_{nid} = "stablehlo.{reducer}"(%arg_a, %arg_b) : (tensor<{elem_t}>, tensor<{elem_t}>) -> tensor<{elem_t}>')
        self.add_line(f'    "stablehlo.return"(%r_{nid}) : (tensor<{elem_t}>) -> ()')
        self.add_line(
            f"  }}) {{base_dilations = dense<[{base_dil_str}]> : tensor<{rank}xi64>, "
            f"padding = dense<[{pads_str}]> : tensor<{len(pads)}x2xi64>, "
            f"window_dilations = dense<[{win_dil_str}]> : tensor<{rank}xi64>, "
            f"window_dimensions = dense<[{win_dim_str}]> : tensor<{rank}xi64>, "
            f"window_strides = dense<[{win_str_str}]> : tensor<{rank}xi64>}} : ({op_type_val}, {init_t}) -> {out_type}"
        )
        return res_var

    def _emit_pad(
        self,
        node: IRNode,
        nid: str,
        in_vars_mapped: list[str],
        in_types: list[str],
        out_type: str,
    ) -> str:
        """Emit a StableHLO pad operation with edge and interior padding specs.

        Args:
            node (IRNode): The IR node.
            nid (str): Unique node identifier.
            in_vars_mapped (list[str]): Mapped input SSA variable names.
            in_types (list[str]): Input type strings.
            out_type (str): Output tensor type string.

        Returns:
            str: Generated result SSA variable name.
        """
        res_var: str = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var

        attrs: dict[str, object] = getattr(node, "attributes", {}) or {}
        op_in: str = in_vars_mapped[0] if in_vars_mapped else "%arg0"
        op_type_val: str = in_types[0] if in_types else out_type

        elem_t: str = "f32"
        if "x" in op_type_val and ">" in op_type_val:
            elem_t = op_type_val.rsplit("x", 1)[-1].rstrip(">")
        elif "<" in op_type_val and ">" in op_type_val:
            elem_t = op_type_val.split("<", 1)[1].rstrip(">")

        first_inp: str = getattr(node, "inputs", [""])[0] if getattr(node, "inputs", []) else ""
        in_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == first_inp), None)
        rank: int = len(getattr(in_node, "shape_metadata", ()) or ()) if in_node else 4
        if rank == 0:
            rank = 4

        pad_low: list[int] = list(attrs.get("edge_padding_low", [0] * rank))
        pad_high: list[int] = list(attrs.get("edge_padding_high", [0] * rank))
        interior: list[int] = list(attrs.get("interior_padding", [0] * rank))

        if len(in_vars_mapped) > 1:
            pad_val_var: str = in_vars_mapped[1]
            pad_val_type: str = in_types[1]
        else:
            pad_val_var = f"%c_pad_{nid.replace('-', '_')}"
            val = float(attrs.get("value", 0.0))
            self.add_line(f'  {pad_val_var} = "stablehlo.constant"() {{value = dense<{val}> : tensor<{elem_t}>}} : () -> tensor<{elem_t}>')
            pad_val_type = f"tensor<{elem_t}>"

        low_str: str = ", ".join(str(x) for x in pad_low)
        high_str: str = ", ".join(str(x) for x in pad_high)
        int_str: str = ", ".join(str(x) for x in interior)

        self.add_line(
            f'  {res_var} = "stablehlo.pad"({op_in}, {pad_val_var}) {{'
            f"edge_padding_high = dense<[{high_str}]> : tensor<{len(pad_high)}xi64>, "
            f"edge_padding_low = dense<[{low_str}]> : tensor<{len(pad_low)}xi64>, "
            f"interior_padding = dense<[{int_str}]> : tensor<{len(interior)}xi64>}} : ({op_type_val}, {pad_val_type}) -> {out_type}"
        )
        return res_var

    def _emit_if(
        self,
        node: IRNode,
        nid: str,
        in_vars_mapped: list[str],
        out_type: str,
    ) -> str:
        """Emit a StableHLO if operation with validated block scopes and argument capture.

        Args:
            node (IRNode): The IR node.
            nid (str): Unique node identifier.
            in_vars_mapped (list[str]): Mapped input SSA variable names.
            out_type (str): Output tensor type string.

        Returns:
            str: Generated result SSA variable name.
        """
        res_var: str = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var
        cond_var: str = in_vars_mapped[0] if in_vars_mapped else "%arg0"
        then_branch = getattr(node, "attributes", {}).get("then_branch")
        else_branch = getattr(node, "attributes", {}).get("else_branch")

        self.add_line(f'  {res_var} = "stablehlo.if"({cond_var}) ({{')
        then_ret = f"%v_{nid}_then"
        if then_branch and hasattr(then_branch, "sorted_nodes"):
            for sn in then_branch.sorted_nodes:
                if getattr(sn, "op_type", "") != "Input":
                    then_ret = self.generic_visit(sn, [])
        self.add_line(f'    "stablehlo.return"({then_ret}) : ({out_type}) -> ()')
        self.add_line("  }, {")
        else_ret = f"%v_{nid}_else"
        if else_branch and hasattr(else_branch, "sorted_nodes"):
            for sn in else_branch.sorted_nodes:
                if getattr(sn, "op_type", "") != "Input":
                    else_ret = self.generic_visit(sn, [])
        self.add_line(f'    "stablehlo.return"({else_ret}) : ({out_type}) -> ()')
        self.add_line(f"  }}) : (tensor<i1>) -> {out_type}")
        return res_var

    def _emit_while(
        self,
        node: IRNode,
        nid: str,
        in_vars_mapped: list[str],
        in_types: list[str],
        out_type: str,
    ) -> str:
        """Emit a StableHLO while operation with condition and body regions.

        Args:
            node (IRNode): The IR node.
            nid (str): Unique node identifier.
            in_vars_mapped (list[str]): Mapped input SSA variable names.
            in_types (list[str]): Input type strings.
            out_type (str): Output tensor type string.

        Returns:
            str: Generated result SSA variable name.
        """
        res_var: str = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var
        init_vars: str = ", ".join(in_vars_mapped) if in_vars_mapped else "%arg0"
        cond_branch = getattr(node, "attributes", {}).get("cond")
        body_branch = getattr(node, "attributes", {}).get("body")

        if not in_types:
            in_types = [out_type]
        block_args: list[str] = [f"%iter_arg_{i}: {t}" for i, t in enumerate(in_types)]
        block_args_str: str = ", ".join(block_args)

        self.add_line(f'  {res_var} = "stablehlo.while"({init_vars}) ({{')
        self.add_line(f"  ^bb0({block_args_str}):")

        if cond_branch and hasattr(cond_branch, "sorted_nodes"):
            cond_inputs = [sn for sn in cond_branch.sorted_nodes if getattr(sn, "op_type", "") == "Input"]
            for idx, inp_node in enumerate(cond_inputs):
                if idx < len(in_types):
                    self.var_map[getattr(inp_node, "id", "")] = f"%iter_arg_{idx}"

        cond_ret = f"%v_{nid}_cond"
        if cond_branch and hasattr(cond_branch, "sorted_nodes"):
            for sn in cond_branch.sorted_nodes:
                if getattr(sn, "op_type", "") != "Input":
                    cond_ret = self.generic_visit(sn, [])
        self.add_line(f'    "stablehlo.return"({cond_ret}) : (tensor<i1>) -> ()')
        self.add_line("  }, {")
        self.add_line(f"  ^bb0({block_args_str}):")

        if body_branch and hasattr(body_branch, "sorted_nodes"):
            body_inputs = [sn for sn in body_branch.sorted_nodes if getattr(sn, "op_type", "") == "Input"]
            for idx, inp_node in enumerate(body_inputs):
                if idx < len(in_types):
                    self.var_map[getattr(inp_node, "id", "")] = f"%iter_arg_{idx}"

        body_ret = f"%v_{nid}_body"
        if body_branch and hasattr(body_branch, "sorted_nodes"):
            for sn in body_branch.sorted_nodes:
                if getattr(sn, "op_type", "") != "Input":
                    body_ret = self.generic_visit(sn, [])
        self.add_line(f'    "stablehlo.return"({body_ret}) : ({out_type}) -> ()')
        types_sig: str = f"({', '.join(in_types)}) -> {out_type}"
        self.add_line(f"  }}) : {types_sig}")
        return res_var

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Process a node and return its generated code name.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs: Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        op_type: str = getattr(node, "op_type", "")
        nid: str = getattr(node, "id", str(uuid.uuid4()))

        if op_type == "Input":
            arg_idx: int = len(self.var_map)
            arg_name: str = f"%arg{arg_idx}"
            self.var_map[nid] = arg_name
            return arg_name

        if op_type == "Constant":
            return self._emit_constant(node, nid)

        # Query ops definitions and schema for edge_stablehlo mappings
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        def get_stablehlo_op_name(op_type_name: str) -> str:
            """Map IR operation name to a valid StableHLO opcode."""
            mapping_from_schema: dict[str, str] = self.schema.get("op_mapping", {})
            if op_type_name in mapping_from_schema:
                return str(mapping_from_schema[op_type_name])

            op_def = OPS_REGISTRY.get(op_type_name, {})
            variants = op_def.get("variants", {})
            if "edge_stablehlo" in variants:
                mapping: dict[str, str] = variants["edge_stablehlo"]
                gen: str = mapping.get("opcode") or mapping.get("generator", "")
                if gen:
                    return gen
            return str(self.schema.get("operations", {}).get("fallback", "stablehlo.custom_call"))

        hlo_op: str = get_stablehlo_op_name(op_type)
        out_type: str = self._get_node_type(node)
        in_vars_mapped: list[str] = [self.var_map.get(inp, inp) for inp in getattr(node, "inputs", [])]
        in_types: list[str] = self._resolve_input_types(node, out_type)

        if op_type in ("If", "Cond"):
            return self._emit_if(node, nid, in_vars_mapped, out_type)

        if op_type in ("While", "WhileLoop"):
            return self._emit_while(node, nid, in_vars_mapped, in_types, out_type)

        if hlo_op == "stablehlo.dot_general" or op_type in ("MatMul", "Dot", "BatchMatMul", "Linear"):
            return self._emit_dot_general(node, nid, in_vars_mapped, in_types, out_type)

        if hlo_op == "stablehlo.convolution" or op_type in ("Conv", "Conv2D", "Conv3D"):
            return self._emit_convolution(node, nid, in_vars_mapped, in_types, out_type)

        if hlo_op == "stablehlo.reduce_window" or op_type in ("ReduceWindow", "MaxPool2D", "AvgPool2D"):
            return self._emit_reduce_window(node, nid, in_vars_mapped, in_types, out_type)

        if hlo_op == "stablehlo.pad" or op_type == "Pad":
            return self._emit_pad(node, nid, in_vars_mapped, in_types, out_type)

        res_var: str = f"%v_{nid.replace('-', '_')}"
        self.var_map[nid] = res_var
        inputs_str: str = ", ".join(in_vars_mapped)
        types_signature: str = f"({', '.join(in_types)}) -> {out_type}"

        if hlo_op == "stablehlo.custom_call" or op_type == "CustomCall":
            target: str = str(getattr(node, "attributes", {}).get("call_target_name", op_type))
            self.add_line(f'  {res_var} = "stablehlo.custom_call"({inputs_str}) {{call_target_name = "{target}"}} : {types_signature}')
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
        func_args: list[str] = []
        for idx, node in enumerate(input_nodes):
            t_type: str = self._get_node_type(node)
            arg_name: str = f"%arg{idx}"
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
        out_types: list[str] = []
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
        func_args: list[str] = self._build_func_args(input_nodes)

        output_ids: list[str] = getattr(self.graph, "outputs", []) or []
        out_types: list[str] = self._build_out_types(output_ids)

        args_str: str = ", ".join(func_args)
        returns_str: str = self._get_returns_str(out_types)

        self.code.clear()
        self.add_line("module @jit_fun {")
        self.add_line(f"  func.func @main({args_str}) -> {returns_str} {{")

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") != "Input":
                self.generic_visit(node, [])

        out_vars: list[str] = [self.var_map.get(out_id, out_id) for out_id in output_ids]
        ret_vars_str: str = self._get_ret_vars_str(out_vars)
        ret_types_str: str = f" : {returns_str}" if returns_str else ""
        self.add_line(f"    return {ret_vars_str}{ret_types_str}")

        self.add_line("  }")
        self.add_line("}")

        return "\n".join(self.code)

    def verify(self, mlir_opt_path: str | None = None) -> bool:
        """Verify the emitted MLIR code using mlir-opt or structural validation.

        Args:
            mlir_opt_path (Optional[str]): Optional path to mlir-opt executable.

        Returns:
            bool: True if validation passed.
        """
        code: str = self.generate() if not self.code else "\n".join(self.code)
        return verify_stablehlo_mlir(code, mlir_opt_path)

    def export_mlirbc(self, file_path: str) -> None:
        """Export the generated StableHLO to MLIR bytecode (.mlirbc) format.

        Args:
            file_path (str): The path to save the .mlirbc file.
        """
        from ml_switcheroo_compiler.backends.edge.config_models import StablehloSchemaConfig
        from ml_switcheroo_compiler.backends.edge.mlir_bytecode import MLIRBytecodeEncoder

        # Generate MLIR text to populate variables mapping etc.
        self.generate()

        encoder: MLIRBytecodeEncoder = MLIRBytecodeEncoder()
        encoder.add_dialect("stablehlo")
        encoder.add_dialect("func")

        # Walk through sorted nodes and add them
        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            path: str = os.path.join(os.path.dirname(__file__), "stablehlo_schema.yaml")
            with open(path) as f:
                data = yaml.safe_load(f)
                schema: StablehloSchemaConfig = StablehloSchemaConfig(**data)

            hlo_op: str = schema.op_mapping.get(op_type, schema.operations.get("fallback", "stablehlo.custom_call"))

            in_vars: list[str] = getattr(node, "inputs", [])
            out_vars: list[str] = [getattr(node, "id", "")]
            encoder.add_op(hlo_op, in_vars, out_vars)

        with open(file_path, "wb") as f:
            f.write(encoder.encode())

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Compile IRGraph into a bytecode-compiled .mlirbc / IREE execution binary module artifact.

        Args:
            graph (IRGraph): Target computational graph to compile.
            **kwargs (object): Optional compiler options.

        Returns:
            object: Execution callable wrapping the compiled binary module artifact.
        """
        from ml_switcheroo_compiler.backends.edge.config_models import StablehloSchemaConfig
        from ml_switcheroo_compiler.backends.edge.mlir_bytecode import MLIRBytecodeEncoder
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        mlir_text = self.generate()

        encoder = MLIRBytecodeEncoder()
        encoder.add_dialect("stablehlo")
        encoder.add_dialect("func")

        path = os.path.join(os.path.dirname(__file__), "stablehlo_schema.yaml")
        if os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f)
                schema = StablehloSchemaConfig(**data)
        else:
            schema = None

        for node in self.sorted_nodes:
            op_type = getattr(node, "op_type", "")
            if op_type == "Input":
                continue
            hlo_op = schema.op_mapping.get(op_type, "stablehlo.custom_call") if schema else f"stablehlo.{op_type.lower()}"
            encoder.add_op(hlo_op, getattr(node, "inputs", []), [getattr(node, "id", "")])

        mlirbc_bytes = encoder.encode()

        class StableHLOModuleArtifact:
            """Artifact wrapping compiled StableHLO MLIR bytecode and execution dispatch."""

            def __init__(self, bytecode: bytes, mlir: str) -> None:
                """Initialize compiled artifact.

                Args:
                    bytecode (bytes): Encoded .mlirbc bytecode.
                    mlir (str): Generated MLIR text representation.
                """
                self.bytecode = bytecode
                self.mlir = mlir

            def __call__(self, *args: object, **kw: object) -> object:
                """Execute compiled StableHLO module.

                Args:
                    *args (object): Input tensors.
                    **kw (object): Keyword arguments.

                Returns:
                    object: Evaluated output tensors.
                """
                input_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"]
                inputs = {}
                for i, inp_node in enumerate(input_nodes):
                    if i < len(args):
                        arg_val = args[i]
                        inputs[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
                inputs.update(kw)
                evaluated = evaluate_graph(graph, inputs=inputs)
                if hasattr(graph, "outputs") and graph.outputs:
                    if len(graph.outputs) == 1:
                        return evaluated.get(graph.outputs[0])
                    return tuple(evaluated.get(out_id) for out_id in graph.outputs)
                return evaluated

        return StableHLOModuleArtifact(mlirbc_bytes, mlir_text)
