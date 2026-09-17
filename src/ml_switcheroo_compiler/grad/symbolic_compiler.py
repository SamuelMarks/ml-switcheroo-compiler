"""Symbolic expression compiler for declarative autodiff rules."""

from __future__ import annotations

import ast
import os
import re
import uuid
from collections.abc import Sequence
from typing import Union

import yaml
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.grad.config_models import (
    AutodiffRuleModel,
    AutodiffRulesManifestModel,
    HigherOrderAutodiffModel,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.base import emit_ir_node

AttrPrimitive = Union[str, int, float, bool, None]


class SymbolicExpressionCompiler:
    """Compiles symbolic autodiff expressions from declarative YAML manifests into IR nodes."""

    def __init__(self, manifest_path: str | None = None) -> None:
        """Initialize the symbolic compiler with an optional custom manifest path.

        Args:
            manifest_path (str | None): Optional filesystem path to autodiff rules YAML manifest.
        """
        if manifest_path is None:
            manifest_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "transforms",
                    "autodiff_rules.yaml",
                )
            )
        self.manifest_path: str = manifest_path
        self._manifest: AutodiffRulesManifestModel | None = None

    def load_manifest(self) -> AutodiffRulesManifestModel:
        """Load and cache the declarative autodiff rules manifest.

        Returns:
            AutodiffRulesManifestModel: Validated autodiff rules manifest model.

        Raises:
            FileNotFoundError: If the configured manifest file does not exist.
        """
        if self._manifest is not None:
            return self._manifest

        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Autodiff rules manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, encoding="utf-8") as f:
            raw_data: dict[str, str | int | float | bool | list | dict | None] = yaml.safe_load(f) or {}

        self._manifest = AutodiffRulesManifestModel.model_validate(raw_data)
        return self._manifest

    def split_nested_args(self, args_str: str) -> list[str]:
        """Split a comma-separated argument string while respecting nested parentheses.

        Args:
            args_str (str): The raw argument substring.

        Returns:
            list[str]: Extracted individual argument expressions.
        """
        args: list[str] = []
        depth: int = 0
        current_arg: str = ""
        for char in args_str:
            if char in ("(", "[", "{"):
                depth += 1
            elif char in (")", "]", "}"):
                depth -= 1
            elif char == "," and depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
                continue
            current_arg += char
        if current_arg:
            args.append(current_arg.strip())
        return args

    def parse_args_and_attrs(
        self,
        graph: IRGraph | LogicalGraph,
        args: list[str],
        node: LogicalNode | IRNode,
        cotangent: str | None = None,
        tangents: Sequence[str] | None = None,
    ) -> tuple[list[str], dict[str, AttrPrimitive]]:
        """Parse positional operand expressions and keyword attributes.

        Args:
            graph (IRGraph | LogicalGraph): Target graph being populated.
            args (list[str]): Raw string arguments from the expression.
            node (LogicalNode | IRNode): Primal node being differentiated.
            cotangent (str | None): Cotangent identifier for VJP.
            tangents (Sequence[str] | None): Tangent identifiers for JVP.

        Returns:
            tuple[list[str], dict[str, AttrPrimitive]]: Input node IDs and keyword attributes.
        """
        parsed_inputs: list[str] = []
        parsed_attrs: dict[str, AttrPrimitive] = {}
        for arg in args:
            if "=" in arg and not arg.startswith("("):
                k, v = arg.split("=", 1)
                try:
                    parsed_attrs[k.strip()] = ast.literal_eval(v.strip())
                except Exception:
                    parsed_attrs[k.strip()] = v.strip()
            else:
                parsed_inputs.append(self.compile_expression(graph, arg, node, cotangent, tangents))
        return parsed_inputs, parsed_attrs

    def build_call_node(
        self,
        graph: IRGraph | LogicalGraph,
        op: str,
        args: list[str],
        node: LogicalNode | IRNode,
        cotangent: str | None = None,
        tangents: Sequence[str] | None = None,
    ) -> str:
        """Construct an IR node for an invocation parsed from symbolic expressions.

        Args:
            graph (IRGraph | LogicalGraph): Target graph receiving the new node.
            op (str): Operator opcode name.
            args (list[str]): Argument string expressions.
            node (LogicalNode | IRNode): Primal node being differentiated.
            cotangent (str | None): Cotangent identifier.
            tangents (Sequence[str] | None): Tangent identifiers.

        Returns:
            str: Identifier of the created IR node.
        """
        parsed_inputs, parsed_attrs = self.parse_args_and_attrs(graph, args, node, cotangent, tangents)

        alias_map: dict[str, str] = {
            "Neg": "Negative",
            "Sub": "Subtract",
            "Mul": "Multiply",
            "Div": "TrueDivide",
            "Divide": "TrueDivide",
        }
        canonical_op: str = alias_map.get(op, op)

        if canonical_op == "Constant":
            val_f: float = float(parsed_inputs[0]) if parsed_inputs else 0.0
            node_id: str = f"cst_sym_{val_f}_{uuid.uuid4().hex[:6]}".replace(".", "_").replace("-", "neg_")
            cst_node = LogicalNode(
                id=node_id,
                op_type="Constant",
                inputs=[],
                attributes={"value": val_f},
                shape_metadata=getattr(node, "shape_metadata", ()),
            )
            graph.nodes[node_id] = cst_node
            return node_id

        if canonical_op in ("BroadcastAlign", "BroadcastReduce"):
            cot_id, tgt_id = parsed_inputs[0], parsed_inputs[1]
            shape_meta = getattr(graph.nodes.get(tgt_id), "shape_metadata", None) if hasattr(graph, "nodes") and tgt_id in graph.nodes else None
            return emit_ir_node(
                graph,
                "BroadcastReduce",
                [cot_id, tgt_id],
                shape_metadata=shape_meta,
                attributes={"target_id": tgt_id},
            )

        if canonical_op == "BroadcastLike":
            src_id, tgt_id = parsed_inputs[0], parsed_inputs[1]
            shape_meta = getattr(graph.nodes.get(tgt_id), "shape_metadata", None) if hasattr(graph, "nodes") and tgt_id in graph.nodes else None
            return emit_ir_node(
                graph,
                "BroadcastLike",
                [src_id, tgt_id],
                shape_metadata=shape_meta,
                attributes={"target_id": tgt_id},
            )

        attrs: dict[str, AttrPrimitive] = {}
        if hasattr(node, "attributes") and node.attributes and canonical_op == getattr(node, "op_type", ""):
            attrs.update(dict(node.attributes))
        attrs.update(parsed_attrs)

        shape_meta = getattr(node, "shape_metadata", None)
        return emit_ir_node(graph, canonical_op, parsed_inputs, shape_meta, attributes=attrs)

    def _resolve_symbolic_variable(
        self,
        clean_expr: str,
        node: LogicalNode | IRNode,
        cotangent: str | None = None,
        tangents: Sequence[str] | None = None,
    ) -> str | None:
        """Resolve special placeholder symbols like $cotangent, $output, $input, $tangent.

        Args:
            clean_expr (str): Stripped symbolic string token.
            node (LogicalNode | IRNode): Primal node being differentiated.
            cotangent (str | None): Cotangent identifier.
            tangents (Sequence[str] | None): Tangent identifiers.

        Returns:
            str | None: Resolved string identifier if matched, otherwise None.
        """
        if clean_expr == "$cotangent":
            return cotangent or ""
        if clean_expr == "$output":
            return getattr(node, "id", "")

        inp_match: re.Match[str] | None = re.match(r"^\$input\[(\d+)\]$", clean_expr)
        if inp_match:
            idx: int = int(inp_match.group(1))
            return node.inputs[idx] if hasattr(node, "inputs") and idx < len(node.inputs) else ""

        tan_match: re.Match[str] | None = re.match(r"^\$tangent\[(\d+)\]$", clean_expr)
        if tan_match:
            idx: int = int(tan_match.group(1))
            return tangents[idx] if tangents and idx < len(tangents) else ""

        return None

    def compile_expression(
        self,
        graph: IRGraph | LogicalGraph,
        expr: str,
        node: LogicalNode | IRNode,
        cotangent: str | None = None,
        tangents: Sequence[str] | None = None,
    ) -> str:
        """Compile a symbolic expression into unified IR nodes.

        Args:
            graph (IRGraph | LogicalGraph): Target graph receiving synthesized nodes.
            expr (str): Symbolic derivative expression (e.g. 'Multiply($cotangent, Cos($input[0]))').
            node (LogicalNode | IRNode): Primal node being differentiated.
            cotangent (str | None): Cotangent identifier for VJP.
            tangents (Sequence[str] | None): Input tangent identifiers for JVP.

        Returns:
            str: Identifier of the emitted IR node representing the expression.
        """
        clean_expr: str = expr.strip()
        try:
            float(clean_expr)
            return clean_expr
        except ValueError:
            pass

        resolved_var: str | None = self._resolve_symbolic_variable(clean_expr, node, cotangent, tangents)
        if resolved_var is not None:
            return resolved_var

        if clean_expr in ("$zero", "Zero()"):
            zero_id: str = f"zero_sym_{uuid.uuid4().hex[:6]}"
            zero_node = LogicalNode(
                id=zero_id,
                op_type="ZerosLike",
                inputs=list(node.inputs[:1]) if getattr(node, "inputs", None) else [],
                attributes={"value": 0.0},
                shape_metadata=getattr(node, "shape_metadata", ()),
            )
            graph.nodes[zero_id] = zero_node
            return zero_id

        if clean_expr.startswith("- ") or (clean_expr.startswith("-") and not clean_expr.startswith("-=")):
            inner_str: str = clean_expr[2:].strip() if clean_expr.startswith("- ") else clean_expr[1:].strip()
            inner_id: str = self.compile_expression(graph, inner_str, node, cotangent, tangents)
            return emit_ir_node(graph, "Negative", [inner_id], getattr(node, "shape_metadata", None))

        call_match: re.Match[str] | None = re.match(r"^([A-Za-z0-9_]+)\((.*)\)$", clean_expr)
        if call_match:
            op_name: str = call_match.group(1)
            raw_args: list[str] = self.split_nested_args(call_match.group(2))
            return self.build_call_node(graph, op_name, raw_args, node, cotangent, tangents)

        return clean_expr

    def compile_vjp(
        self,
        graph: IRGraph | LogicalGraph,
        opcode: str,
        node: LogicalNode | IRNode,
        cotangent: str,
    ) -> tuple[str, ...] | None:
        """Compile reverse-mode vector-jacobian products for an operation.

        Args:
            graph (IRGraph | LogicalGraph): The computation graph.
            opcode (str): Operation type identifier.
            node (LogicalNode | IRNode): Primal node being differentiated.
            cotangent (str): Cotangent identifier.

        Returns:
            tuple[str, ...] | None: Tuple of adjoint node IDs per input, or None if unmapped.
        """
        manifest: AutodiffRulesManifestModel = self.load_manifest()
        rule: AutodiffRuleModel | None = manifest.vjp_rules.get(opcode)
        if rule is None:
            return None

        vjp_spec = rule.vjp
        if isinstance(vjp_spec, list):
            adjs: list[str] = [self.compile_expression(graph, expr, node, cotangent=cotangent) for expr in vjp_spec]
            return tuple(adjs)
        if isinstance(vjp_spec, str):
            adj: str = self.compile_expression(graph, vjp_spec, node, cotangent=cotangent)
            return (adj,)
        return None

    def compile_jvp(
        self,
        graph: IRGraph | LogicalGraph,
        opcode: str,
        node: LogicalNode | IRNode,
        tangents: Sequence[str],
    ) -> str | None:
        """Compile forward-mode jacobian-vector product for an operation.

        Args:
            graph (IRGraph | LogicalGraph): The computation graph.
            opcode (str): Operation type identifier.
            node (LogicalNode | IRNode): Primal node being differentiated.
            tangents (Sequence[str]): Input tangent identifiers.

        Returns:
            str | None: Tangent node ID representing output derivative, or None if unmapped.
        """
        manifest: AutodiffRulesManifestModel = self.load_manifest()
        rule: AutodiffRuleModel | None = manifest.jvp_rules.get(opcode)
        if rule is None or not rule.jvp:
            return None

        return self.compile_expression(graph, rule.jvp, node, tangents=tangents)

    def compile_hvp(
        self,
        graph: IRGraph | LogicalGraph,
        opcode: str,
        node: LogicalNode | IRNode,
        cotangent: str,
        tangents: Sequence[str],
    ) -> str | None:
        """Compile declarative Hessian-vector product expression for an operation.

        Args:
            graph (IRGraph | LogicalGraph): The computation graph.
            opcode (str): Operation opcode identifier.
            node (LogicalNode | IRNode): Primal node being evaluated.
            cotangent (str): Upstream cotangent node ID.
            tangents (Sequence[str]): Input tangent node IDs.

        Returns:
            str | None: Resulting HVP node ID, or None if unmapped.
        """
        manifest: AutodiffRulesManifestModel = self.load_manifest()
        rule: HigherOrderAutodiffModel | None = manifest.higher_order_rules.get(opcode)
        if rule is None or not rule.hvp:
            return None
        return self.compile_expression(graph, rule.hvp, node, cotangent=cotangent, tangents=tangents)

    def get_higher_order_rule(self, opcode: str) -> HigherOrderAutodiffModel | None:
        """Retrieve the higher-order autodiff rule model for a specified opcode.

        Args:
            opcode (str): Operation opcode identifier.

        Returns:
            HigherOrderAutodiffModel | None: Loaded rule model if present, else None.
        """
        manifest: AutodiffRulesManifestModel = self.load_manifest()
        return manifest.higher_order_rules.get(opcode)

    def compile_gelu_grad(self, graph: IRGraph | LogicalGraph, cotangent: str, x: str) -> str:
        """Compile composite GELU activation gradient.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            x (str): Forward input node ID.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id="gelu_fwd", op_type="Gelu", inputs=[x])
        expr = "Mul($cotangent, Add(Mul($input[0], Mul(Constant(0.3989422804), Exp(Mul(Constant(-0.5), Pow($input[0], Constant(2.0)))))), Mul(Constant(0.5), Add(Constant(1.0), Erf(Mul($input[0], Constant(0.7071067811)))))))"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)

    def compile_silu_grad(
        self,
        graph: IRGraph | LogicalGraph,
        cotangent: str,
        x: str,
        out: str | None = None,
    ) -> str:
        """Compile composite SiLU activation gradient.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            x (str): Forward input node ID.
            out (str | None): Optional forward output node ID.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id=out or "silu_fwd", op_type="Silu", inputs=[x])
        expr = "Mul($cotangent, Add(Sigmoid($input[0]), Mul($output, Sub(Constant(1.0), Sigmoid($input[0])))))"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)

    def compile_softmax_grad(
        self,
        graph: IRGraph | LogicalGraph,
        cotangent: str,
        out: str,
        axis: int = -1,
    ) -> str:
        """Compile composite Softmax activation gradient.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            out (str): Forward output node ID.
            axis (int): Reduction axis for softmax normalization.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id=out, op_type="Softmax", inputs=[out])
        expr = f"Mul($output, Sub($cotangent, ReduceSum(Mul($cotangent, $output), axis={axis}, keepdims=True)))"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)

    def compile_log_softmax_grad(
        self,
        graph: IRGraph | LogicalGraph,
        cotangent: str,
        out: str,
        axis: int = -1,
    ) -> str:
        """Compile composite LogSoftmax activation gradient.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            out (str): Forward output node ID.
            axis (int): Reduction axis for log-softmax normalization.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id=out, op_type="LogSoftmax", inputs=[out])
        expr = f"Sub($cotangent, Mul(Exp($output), ReduceSum($cotangent, axis={axis}, keepdims=True)))"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)

    def compile_matmul_grad_a(self, graph: IRGraph | LogicalGraph, cotangent: str, b: str) -> str:
        """Compile gradient of matrix multiplication with respect to first operand A.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            b (str): Forward operand B node ID.

        Returns:
            str: Output adjoint node ID for operand A.
        """
        dummy = LogicalNode(id="matmul_fwd", op_type="MatMul", inputs=["a", b])
        expr = "MatMul($cotangent, Transpose($input[1]))"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)

    def compile_matmul_grad_b(self, graph: IRGraph | LogicalGraph, cotangent: str, a: str) -> str:
        """Compile gradient of matrix multiplication with respect to second operand B.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            a (str): Forward operand A node ID.

        Returns:
            str: Output adjoint node ID for operand B.
        """
        dummy = LogicalNode(id="matmul_fwd", op_type="MatMul", inputs=[a, "b"])
        expr = "MatMul(Transpose($input[0]), $cotangent)"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)

    def compile_batch_matmul_grad(
        self,
        graph: IRGraph | LogicalGraph,
        cotangent: str,
        a: str,
        b: str,
    ) -> tuple[str, str]:
        """Compile gradients of batch matrix multiplication with respect to both operands.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            a (str): Forward operand A node ID.
            b (str): Forward operand B node ID.

        Returns:
            tuple[str, str]: Adjoint node IDs for operands A and B.
        """
        dummy = LogicalNode(id="bmm_fwd", op_type="BatchMatMul", inputs=[a, b])
        adj_a = self.compile_expression(graph, "BatchMatMul($cotangent, Transpose($input[1]))", dummy, cotangent)
        adj_b = self.compile_expression(graph, "BatchMatMul(Transpose($input[0]), $cotangent)", dummy, cotangent)
        return adj_a, adj_b

    def compile_transpose_grad(self, graph: IRGraph | LogicalGraph, cotangent: str) -> str:
        """Compile gradient of tensor transpose.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id="trans_fwd", op_type="Transpose", inputs=[cotangent])
        return self.compile_expression(graph, "Transpose($cotangent)", dummy, cotangent=cotangent)

    def compile_reshape_grad(self, graph: IRGraph | LogicalGraph, cotangent: str) -> str:
        """Compile gradient of tensor reshape.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id="reshape_fwd", op_type="Reshape", inputs=[cotangent])
        return self.compile_expression(graph, "Reshape($cotangent)", dummy, cotangent=cotangent)

    def compile_layer_norm_grad(self, graph: IRGraph | LogicalGraph, cotangent: str, x: str, weight: str) -> str:
        """Compile gradient of LayerNorm with respect to input.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            x (str): Forward input node ID.
            weight (str): Normalization scale/weight node ID.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id="ln_fwd", op_type="LayerNorm", inputs=[x, weight])
        expr = "Mul($cotangent, Div($input[1], Sqrt(Add(Variance($input[0]), Constant(1e-5)))))"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)

    def compile_batch_norm_grad(self, graph: IRGraph | LogicalGraph, cotangent: str, x: str, scale: str) -> str:
        """Compile gradient of BatchNorm with respect to input.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            x (str): Forward input node ID.
            scale (str): Normalization scale node ID.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id="bn_fwd", op_type="BatchNorm", inputs=[x, scale])
        expr = "Mul($cotangent, Div($input[1], Sqrt(Add(Variance($input[0]), Constant(1e-5)))))"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)

    def compile_rms_norm_grad(self, graph: IRGraph | LogicalGraph, cotangent: str, x: str, weight: str) -> str:
        """Compile gradient of RMSNorm with respect to input.

        Args:
            graph (IRGraph | LogicalGraph): Target graph.
            cotangent (str): Upstream cotangent node ID.
            x (str): Forward input node ID.
            weight (str): Normalization weight node ID.

        Returns:
            str: Output adjoint node ID.
        """
        dummy = LogicalNode(id="rmsn_fwd", op_type="RMSNorm", inputs=[x, weight])
        expr = "Mul($cotangent, Div($input[1], Sqrt(Add(Mean(Pow($input[0], Constant(2.0))), Constant(1e-5)))))"
        return self.compile_expression(graph, expr, dummy, cotangent=cotangent)
