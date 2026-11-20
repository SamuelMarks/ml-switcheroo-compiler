"""WebGPU WGSL Target Emission with N-Dimensional Coordinate-to-Offset Translation and JS Orchestration."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_int_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._webgpu_base import (
    _as_wgsl_visit_result as _as_wgsl_visit_result,
)
from ml_switcheroo_compiler.backends.edge._webgpu_base import (
    _node_attrs_dict as _node_attrs_dict,
)
from ml_switcheroo_compiler.backends.edge._webgpu_base import (
    _node_inputs_list as _node_inputs_list,
)
from ml_switcheroo_compiler.backends.edge._webgpu_base import (
    _WebGPUGenBase as _WebGPUGenBase,
)
from ml_switcheroo_compiler.backends.edge._webgpu_conv3d_pool3d import (
    _WebGPUConv3DPool3D as _WebGPUConv3DPool3D,
)
from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLEmitter as WGSLEmitter
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("webgpu")
@register_backend("edge_wgsl")
class WebGPUCodeGenerator(_WebGPUConv3DPool3D):
    """WebGPU WGSL Code Generator for emitting compute shader module code and browser JS orchestrator."""

    def visit_AvgPool3D(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for AvgPool3D.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = coerce_int_list(kwargs.get("shape", []))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("AvgPool3D")
        in0_shape, _ = self._get_shape_and_strides(
            next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == input_vars[0]), None
            )
        )

        out_w = shape[4] if len(shape) == 5 else 1
        out_h = shape[3] if len(shape) == 5 else 1
        out_d = shape[2] if len(shape) == 5 else 1
        channels = shape[1] if len(shape) == 5 else 1
        in_w = in0_shape[4] if len(in0_shape) == 5 else 1
        in_h = in0_shape[3] if len(in0_shape) == 5 else 1
        in_d = in0_shape[2] if len(in0_shape) == 5 else 1

        raw_strides = _node_attrs_dict(node).get("strides", (1, 1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1, 1)
        stride_d = int(coerce_int(stride_val[0]))
        stride_h = int(coerce_int(stride_val[1]))
        stride_w = int(coerce_int(stride_val[2]))

        raw_win = _node_attrs_dict(node).get("window", (2, 2, 2))
        win_val = raw_win if isinstance(raw_win, (list, tuple)) else (2, 2, 2)
        window_d = int(coerce_int(win_val[0]))
        window_h = int(coerce_int(win_val[1]))
        window_w = int(coerce_int(win_val[2]))

        body = str(template["body"]).format(
            out_w=out_w,
            out_h=out_h,
            out_d=out_d,
            channels=channels,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            window_d=window_d,
            window_h=window_h,
            window_w=window_w,
            in_d=in_d,
            in_h=in_h,
            in_w=in_w,
        )

        func = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(8, 8, 4)"],
        )
        return (
            self.emitter.emit(func).split("\n"),
            f"Math.ceil({out_w} / 8)",
            f"Math.ceil({out_h} / 8)",
            f"Math.ceil({out_d} / 4)",
        )

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> dict[str, BackendArg]:
        """Compile IRGraph into ready-to-dispatch WGSL compute shader bundles and pipeline layouts.

        Args:
            graph (IRGraph): The computation graph to compile.
            **kwargs (BackendArg): Compilation options.

        Returns:
            dict[str, object]: Bundle containing shader code, pipeline layout, and status.
        """
        if graph != self.graph:
            self.graph = graph
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            self.sorted_nodes = topological_sort(graph)

        code_bundle = self.generate()
        raw_vm = adapt_backend_module(self).get_symbol("var_map", {})
        return {
            "format": "webgpu_wgsl_bundle",
            "code": code_bundle,
            "pipeline_layout": dict(raw_vm) if isinstance(raw_vm, dict) else {},
            "status": "ready_to_dispatch",
        }

    def generate_training_step(
        self,
        wrt_inputs: list[str] | None = None,
        target_output: str | None = None,
    ) -> str:
        """Compile client-side coupled forward-backward WGSL passes for in-browser training.

        Args:
            wrt_inputs (Optional[list[str]]): Input variable IDs with respect to which gradients are calculated.
            target_output (Optional[str]): Target output scalar or loss node ID to differentiate.

        Returns:
            str: Coupled forward-backward JavaScript orchestrator module source code.

        Raises:
            ValueError: If target output cannot be identified in the graph.
        """
        from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

        fwd_code: str = self.generate()

        raw_outs = adapt_backend_module(self.graph).get_symbol("outputs", []) or []
        outputs: list[str] = [str(x) for x in raw_outs] if isinstance(raw_outs, (list, tuple)) else []
        if target_output is not None:
            loss_id: str = target_output
        elif outputs:
            loss_id = outputs[0]
        else:
            raise ValueError("Target output cannot be identified from empty graph outputs.")

        raw_inps = adapt_backend_module(self.graph).get_symbol("inputs", []) or []
        wrt: list[str] = (
            wrt_inputs
            if wrt_inputs is not None
            else ([str(x) for x in raw_inps] if isinstance(raw_inps, (list, tuple)) else [])
        )
        from collections.abc import Mapping

        bwd_graph: IRGraph = graph_grad(self.graph, wrt, loss_id)
        bwd_nodes = adapt_backend_module(bwd_graph).get_symbol("nodes", {})
        bwd_nodes_dict = bwd_nodes if isinstance(bwd_nodes, Mapping) else {}
        bwd_graph.inputs = [
            nid
            for nid, n in bwd_nodes_dict.items()
            if coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) == "Input"
        ]

        bwd_generator: WebGPUCodeGenerator = WebGPUCodeGenerator(bwd_graph)
        bwd_code: str = bwd_generator.generate()

        return (
            "// Coupled forward-backward WebGPU execution module generated by ml-switcheroo-compiler\n"
            f"{fwd_code}\n"
            "// Backward pass:\n"
            f"const bwd_execute = (() => {{\n{bwd_code}\n  return execute;\n}})();\n\n"
            "async function execute_train_step(device, inputs, params, learning_rate = 0.01) {\n"
            "  const fwd_results = await execute(device, inputs);\n"
            "  const bwd_inputs = { ...inputs, ...fwd_results };\n"
            "  const grads = await bwd_execute(device, bwd_inputs);\n"
            "  for (const [pKey, pBuf] of Object.entries(params || {})) {\n"
            "    if (grads[pKey]) {\n"
            "      for (let i = 0; i < pBuf.length; i++) {\n"
            "        pBuf[i] -= learning_rate * grads[pKey][i];\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "  return { outputs: fwd_results, gradients: grads };\n"
            "}\n"
        )
