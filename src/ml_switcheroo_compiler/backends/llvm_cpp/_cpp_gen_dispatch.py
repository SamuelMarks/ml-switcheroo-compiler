"""Named operator dispatch and YAML C++ template emission for CppGenerator."""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    AttrType,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_tiled_dynamic import _CppGenTiledDynamic
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _CppGenDispatch(_CppGenTiledDynamic):
    """C++17 emission layer resolving named visitors and declarative YAML templates."""

    _visit_if_op: Callable[[IRNode, IRGraph | None], None]
    _visit_loop_op: Callable[[IRNode, IRGraph | None], None]

    def _dispatch_named_op(self, node: IRNode, graph_to_use: IRGraph | None) -> bool:
        """Dispatch known specialized C++ visitor methods by operation name.

        Args:
            node (IRNode): The IR node to visit.
            graph_to_use (IRGraph | None): The computational graph context.

        Returns:
            bool: True if the node was handled by a specialized visitor, False otherwise.
        """
        op: str = node.op_type
        op_lower: str = op.lower()
        exact_dispatch: dict[str, Callable[[IRNode, IRGraph | None], None]] = {
            "If": self._visit_if_op,
            "Cond": self._visit_if_op,
            "Loop": self._visit_loop_op,
            "WhileLoop": self._visit_loop_op,
            "Scan": self.visit_Scan,
            "Attention": self.visit_Attention,
        }
        if op in exact_dispatch:
            exact_dispatch[op](node, graph_to_use)
            return True

        if op_lower in ("globalmaxpool2d", "global_maxpool2d"):
            self.visit_MaxPool2D(node, graph_to_use, is_global=True)
            return True
        if op_lower in ("globalavgpool2d", "global_avgpool2d"):
            self.visit_AvgPool2D(node, graph_to_use, is_global=True)
            return True

        lower_dispatch: dict[str, Callable[[IRNode, IRGraph | None], None]] = {
            "conv2d": self.visit_Conv2D,
            "conv3d": self.visit_Conv3D,
            "maxpool2d": self.visit_MaxPool2D,
            "avgpool2d": self.visit_AvgPool2D,
            "groupnorm": self.visit_GroupNorm,
            "softmax": self.visit_Softmax,
            "logsoftmax": self.visit_LogSoftmax,
            "warp_shuffle_reduce": self.visit_WarpShuffleReduce,
            "shared_memory_tree_reduce": self.visit_SharedMemoryTreeReduce,
            "tiled_matmul_2d": self.visit_TiledMatMul2D,
            "conv2d_shared_halo": self.visit_Conv2DSharedHalo,
            "maxpool2d_dynamic": self.visit_MaxPool2DDynamic,
            "avgpool2d_dynamic": self.visit_AvgPool2DDynamic,
            "maxpool3d": self.visit_MaxPool3D,
            "avgpool3d": self.visit_AvgPool3D,
            "layernorm": self.visit_LayerNorm,
            "batchnorm": self.visit_BatchNorm,
            "rmsnorm": self.visit_RMSNorm,
            "slice": self.visit_Slice,
            "pad": self.visit_Pad,
            "gathernd": self.visit_GatherND,
            "gather_nd": self.visit_GatherND,
            "scatternd": self.visit_ScatterND,
            "scatter_nd": self.visit_ScatterND,
        }
        if op_lower in lower_dispatch:
            lower_dispatch[op_lower](node, graph_to_use)
            return True
        return False

    def _resolve_cpp_op_mapping(self, node: IRNode, op: str) -> dict[str, AttrType]:
        """Resolve the C++ template mapping dictionary for a generic operation.

        Args:
            node (IRNode): The IR node being compiled.
            op (str): Operation type string.

        Returns:
            dict[str, AttrType]: Resolved template mapping dictionary (empty if unmapped).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_operation
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        op_def = OPS_REGISTRY.get(op, {})
        raw_variants = op_def.get("variants")
        variants = raw_variants if isinstance(raw_variants, dict) else {}
        raw_llvm = variants.get("llvm_cpp")
        raw_mapping = raw_llvm if isinstance(raw_llvm, dict) else {}
        mapping: dict[str, AttrType] = dict(raw_mapping) if (raw_mapping and raw_mapping.get("template")) else {}

        op_key = op.lower()
        num_inputs = len(adapt_backend_module(node).get_symbol("inputs", []) or [])
        is_registered = op in OPS_REGISTRY or op_key in OPS_REGISTRY
        decl_op = get_cpp_operation(op_key, num_inputs=num_inputs, allow_synth=not self.strict and is_registered)
        if decl_op is None:
            return mapping

        if not mapping:
            mapping = {"template": decl_op.template, "scalar_expr": decl_op.scalar_expr}
        elif mapping.get("scalar_expr") == "in0_val":
            mapping["scalar_expr"] = decl_op.scalar_expr

        if adapt_backend_module(decl_op).get_symbol("init_val", None) is not None and "init_val" not in mapping:
            mapping["init_val"] = decl_op.init_val
        if (
            adapt_backend_module(decl_op).get_symbol("final_combine", None) is not None
            and "final_combine" not in mapping
        ):
            mapping["final_combine"] = decl_op.final_combine
        return mapping

    def _resolve_node_input_vars(self, node: IRNode, graph_to_use: IRGraph | None) -> tuple[str, str, str]:
        """Resolve input variable identifiers for template substitution, accounting for multi-output edges.

        Args:
            node (IRNode): The target IR node.
            graph_to_use (IRGraph | None): The computational graph containing edges.

        Returns:
            tuple[str, str, str]: Resolved variable names (in0_var, in1_var, in2_var).
        """
        in0_var = node.inputs[0] if len(node.inputs) > 0 else "dummy"
        in1_var = node.inputs[1] if len(node.inputs) > 1 else in0_var
        in2_var = node.inputs[2] if len(node.inputs) > 2 else in0_var
        if graph_to_use and adapt_backend_module(graph_to_use).has_symbol("edges"):
            for edge in graph_to_use.edges:
                tgt = adapt_backend_module(edge).get_symbol(
                    "target", adapt_backend_module(edge).get_symbol("target_node", None)
                )
                src = adapt_backend_module(edge).get_symbol(
                    "source", adapt_backend_module(edge).get_symbol("source_node", None)
                )
                if tgt == node.id and adapt_backend_module(edge).get_symbol("source_idx", 0) > 0:
                    t_idx = adapt_backend_module(edge).get_symbol("target_idx", 0)
                    if t_idx == 0:
                        in0_var = str(src)
                    elif t_idx == 1:
                        in1_var = str(src)
                    else:
                        in2_var = str(src)
        return in0_var, in1_var, in2_var

    def _emit_mapped_cpp_template(
        self,
        node: IRNode,
        graph_to_use: IRGraph | None,
        mapping: dict[str, AttrType],
    ) -> None:
        """Format and emit C++ template lines for a mapped generic operation.

        Args:
            node (IRNode): The IR node to emit.
            graph_to_use (IRGraph | None): The computational graph context.
            mapping (dict[str, AttrType]): Resolved template configuration.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template(str(mapping["template"]))
        in0_node: IRNode | None = (
            graph_to_use.nodes.get(node.inputs[0]) if (graph_to_use and len(node.inputs) > 0) else None
        )
        in0_shape: list[int] = self._get_shape(in0_node) if in0_node else [1, 1]
        out_shape: list[int] = self._get_shape(node)
        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"

        in0_var, in1_var, in2_var = self._resolve_node_input_vars(node, graph_to_use)
        expr_format_args: dict[str, AttrType] = {
            "clean_id": node.id,
            "out_shape_str": out_shape_str,
            "in0": in0_var,
            "in1": in1_var,
            "in2": in2_var,
            "rank": len(out_shape),
            "M": out_shape[0] if len(out_shape) > 0 else 1,
            "N": out_shape[1] if len(out_shape) > 1 else 1,
            "K": in0_shape[1] if len(in0_shape) > 1 else 1,
        }
        expr_format_args.update(mapping)
        if mapping.get("template") == "reduction":
            expr_format_args.setdefault("init_val", "0.0f")
            expr_format_args.setdefault("final_combine", "res")
        if "init_val" in expr_format_args:
            ival = str(expr_format_args["init_val"])
            expr_format_args["init_val"] = ival if ("INFINITY" in ival or ival.endswith("f")) else f"{ival}f"

        body: str = str(template["body"]).format(**expr_format_args)
        for line in body.split("\n"):
            if line.strip():
                self.lines.append(f"    {line}")

        if adapt_backend_module(node).has_symbol("outputs") and len(node.outputs) > 1:
            for idx, out_id in enumerate(node.outputs):
                self.lines.append(f"    NDArrayView<float> {out_id} = {node.id}; // multi-output alias {idx}")
