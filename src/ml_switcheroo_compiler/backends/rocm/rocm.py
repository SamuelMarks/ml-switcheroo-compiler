"""ROCm HIP backend generator."""

from __future__ import annotations

import logging
import os
from types import ModuleType

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.hardware_config_models import (
    GridDimensionConfig,
    HardwareTemplateConfig,
    compute_liveness,
    load_hardware_templates,
    resolve_hardware_launch_grid_and_args,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    ROCM_SYNTH_MATH_MAP as ROCM_SYNTH_MATH_MAP,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    KernelArgType as KernelArgType,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _calculate_node_bytes as _calculate_node_bytes,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _evaluate_rocm_fallback as _evaluate_rocm_fallback,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _extract_kernel_name as _extract_kernel_name,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _graph_nodes_values as _graph_nodes_values,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _graph_str_list as _graph_str_list,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _marshall_kernel_params as _marshall_kernel_params,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _node_inputs_list as _node_inputs_list,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _prepare_rocm_input_buffers as _prepare_rocm_input_buffers,
)
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    _synthesize_rocm_kernel as _synthesize_rocm_kernel,
)
from ml_switcheroo_compiler.backends.rocm._rocm_runner_exec import ROCmRunner as ROCmRunner
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

cupy: ModuleType | None = None


@register_backend("rocm")
class RocmCodeGenerator(BaseGenerator):
    """ROCm HIP C++ Code Generator."""

    def __init__(self, graph: IRGraph, delegates: list[CodeGeneratorVisitor] | None = None) -> None:
        """Initialize RocmCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Optional[list[CodeGeneratorVisitor]], optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        yaml_dir: str = os.path.join(os.path.dirname(__file__), "rocm_orchestration")
        yaml_path: str = os.path.join(os.path.dirname(__file__), "rocm_templates.yaml")
        self.config = load_hardware_templates(yaml_dir, yaml_path)

    def _ensure_template(
        self,
        node: EagerTensorProtocol | IRNode,
        node_idx: int,
    ) -> (
        HardwareTemplateConfig | dict[str, str | int | float | list[int] | None] | list[str] | str | int | float | None
    ):
        """Ensure an op template exists in self.config.templates, synthesizing elementwise kernels if needed.

        Args:
            node (EagerTensorProtocol | IRNode): Target IR node.
            node_idx (int): Current node index.

        Returns:
            Optional[Union[HardwareTemplateConfig, dict[str, Union[str, int, float, list[int], None]], list[str],
                str]]: Existing or synthesized template.

        Raises:
            BackendNotSupportedError: If `_ensure_template` encounters an unsupported operation or missing template.
        """
        op_type = adapt_backend_module(node).get_symbol("op_type", "")
        if op_type.lower() in ("input", "output", "fusedelementwise"):
            return None
        if "unsupported" in op_type.lower():
            raise BackendNotSupportedError(f"Operation '{op_type}' is not supported by rocm backend.")
        tpl = self.config.templates.get(op_type.lower())
        if not tpl:
            clean_op = op_type.lower().replace("-", "_").replace(".", "_")
            inps = _node_inputs_list(node)
            raw_attrs = adapt_backend_module(node).get_symbol("attributes", {})
            raw_se = raw_attrs.get("scalar_expr") if isinstance(raw_attrs, dict) else None
            scalar_expr = str(raw_se) if raw_se is not None else None
            body = _synthesize_rocm_kernel(clean_op, inps, scalar_expr)
            tpl = HardwareTemplateConfig(
                body=body,
                grid_calc=GridDimensionConfig(
                    x=f"({{num_elements}} + block_{node_idx}.x - 1) / block_{node_idx}.x", y="1", z="1"
                ),
                workgroup_size=[256, 1, 1],
            )
            self.config.templates[op_type.lower()] = tpl
        return tpl

    def _free_dead_buffers(
        self,
        node: EagerTensorProtocol | IRNode,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        hip: list[str],
    ) -> None:
        """Emit hipFree calls for dead input buffers whose lifetime has ended.

        Args:
            node (EagerTensorProtocol | IRNode): The active IRNode.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            hip (list[str]): Output ROCm HIP source lines.
        """
        for inp in _node_inputs_list(node):
            if last_consumer.get(inp) == node.id and inp not in outputs_set and inp not in inputs_set:
                buf_to_free = node_buffers.get(inp)
                if buf_to_free:
                    hip.append(f"    HIP_CHECK(hipFree({buf_to_free}));")

    def _allocate_node_outputs(
        self,
        node: EagerTensorProtocol | IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        bytes_alloc: int,
        hip: list[str],
    ) -> str:
        """Allocate device buffers for single or multiple node outputs.

        Args:
            node (EagerTensorProtocol | IRNode): Target IR node.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            bytes_alloc (int): Total allocated bytes per output buffer.
            hip (list[str]): Output ROCm HIP source lines.

        Returns:
            str: Output argument name or comma-separated names.
        """
        raw_attrs = adapt_backend_module(node).get_symbol("attributes", {})
        raw_outs = (raw_attrs.get("outputs") if isinstance(raw_attrs, dict) else None) or (
            node.outputs if adapt_backend_module(node).has_symbol("outputs") and len(node.outputs) > 1 else None
        )
        if raw_outs and isinstance(raw_outs, (list, tuple)) and len(raw_outs) > 1:
            out_names: list[str] = []
            for out_idx, out_id in enumerate(raw_outs):
                sub_out_name = f"d_out_{node_idx}_{out_idx}"
                node_buffers[str(out_id)] = sub_out_name
                out_names.append(sub_out_name)
                hip.append(f"    float* {sub_out_name};")
                hip.append(f"    HIP_CHECK(hipMalloc(&{sub_out_name}, {bytes_alloc}));")
            node_buffers[coerce_str(adapt_backend_module(node).get_symbol("id", ""))] = out_names[0]
            return ", ".join(out_names)

        out_name = f"d_out_{node_idx}"
        node_buffers[coerce_str(adapt_backend_module(node).get_symbol("id", ""))] = out_name
        hip.append(f"    float* {out_name};")
        hip.append(f"    HIP_CHECK(hipMalloc(&{out_name}, {bytes_alloc}));")
        return out_name

    def _emit_node(
        self,
        node: IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        hip: list[str],
    ) -> None:
        """Emit ROCm kernel allocation, launch, and buffer freeing for a single IRNode.

        Args:
            node (IRNode): IR node to emit.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            hip (list[str]): Output ROCm HIP source lines.
        """
        op_type = adapt_backend_module(node).get_symbol("op_type", "")
        if op_type.lower() == "fusedelementwise":
            out_name = f"d_out_{node_idx}"
            node_buffers[coerce_str(adapt_backend_module(node).get_symbol("id", ""))] = out_name
            num_elements, bytes_alloc = _calculate_node_bytes(node)
            hip.append(f"    float* {out_name};")
            hip.append(f"    HIP_CHECK(hipMalloc(&{out_name}, {bytes_alloc}));")
            hip.append(f"    dim3 block_{node_idx}(256, 1, 1);")
            (grid_x, grid_y, grid_z), launch_args = resolve_hardware_launch_grid_and_args(
                node, {}, node_buffers, out_name, num_elements, node_idx, self.graph
            )
            hip.append(f"    dim3 grid_{node_idx}({grid_x}, {grid_y}, {grid_z});")
            kernel_fn = f"fused_elementwise_kernel_{node_idx}"
            hip.append(
                f"    hipLaunchKernelGGL({kernel_fn}, grid_{node_idx}, block_{node_idx}, 0, 0, "
                f"{', '.join(launch_args)});"
            )
            hip.append("    HIP_CHECK(hipGetLastError());")
            self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, hip)
            return

        tpl = self._ensure_template(node, node_idx)
        if not tpl:
            return
        num_elements, bytes_alloc = _calculate_node_bytes(node)
        out_name = self._allocate_node_outputs(node, node_idx, node_buffers, bytes_alloc, hip)
        wg = coerce_int_list(adapt_backend_module(tpl).require_symbol("get")("workgroup_size", [1, 1, 1]) or [1, 1, 1])
        hip.append(f"    dim3 block_{node_idx}({wg[0]}, {wg[1]}, {wg[2]});")

        (grid_x, grid_y, grid_z), launch_args = resolve_hardware_launch_grid_and_args(
            node, tpl, node_buffers, out_name, num_elements, node_idx, self.graph
        )
        hip.append(f"    dim3 grid_{node_idx}({grid_x}, {grid_y}, {grid_z});")

        kernel_fn = _extract_kernel_name(
            coerce_str(adapt_backend_module(tpl).require_symbol("get")("body", "")), op_type.lower()
        )
        hip.append(
            f"    hipLaunchKernelGGL({kernel_fn}, grid_{node_idx}, block_{node_idx}, 0, 0, {', '.join(launch_args)});"
        )
        hip.append("    HIP_CHECK(hipGetLastError());")

        self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, hip)

    def generate(self) -> str:
        """Generate ROCm HIP code with dynamic buffer allocations and memory lifetime tracking.

        Returns:
            str: Generated ROCm strings.
        """
        hip: list[str] = [
            "// ROCm HIP Generated by ml-switcheroo-compiler",
            "#include <hip/hip_runtime.h>",
            "#include <iostream>",
            "#include <vector>",
            "",
            "#define HIP_CHECK(call) ",
            "    do { ",
            "        hipError_t err = call; ",
            "        if (err != hipSuccess) { ",
            (
                '            std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << '
                '":" << __LINE__ << std::endl; '
            ),
            "            exit(err); ",
            "        } ",
            "    } while (0)",
            "",
            "// Dynamic N-dimensional coordinate-to-linear-offset index calculation",
            (
                "__device__ inline int coords_to_offset_nd(int linear_idx, const int* shape, const int* "
                "strides, int ndim) {"
            ),
            "    int offset = 0;",
            "    int rem = linear_idx;",
            "    for (int d = ndim - 1; d >= 0; --d) {",
            "        int coord = rem % shape[d];",
            "        rem /= shape[d];",
            "        offset += coord * strides[d];",
            "    }",
            "    return offset;",
            "}",
            "",
        ]

        for node_idx, node in enumerate(_graph_nodes_values(self.graph)):
            op_type = adapt_backend_module(node).get_symbol("op_type", "")
            if op_type.lower() == "fusedelementwise":
                raw_attrs = adapt_backend_module(node).get_symbol("attributes", {})
                scalar_expr = str(raw_attrs.get("scalar_expr", "in0")) if isinstance(raw_attrs, dict) else "in0"
                in_list = _node_inputs_list(node)
                in_params = ", ".join([f"const float* in_{k}" for k in range(len(in_list))])
                load_lines = "\n".join([f"        float in{k} = in_{k}[idx];" for k in range(len(in_list))])
                hip.append(f"__global__ void fused_elementwise_kernel_{node_idx}({in_params}, float* out, int N) {{")
                hip.append("    int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;")
                hip.append("    if (idx < N) {")
                if load_lines:
                    hip.append(load_lines)
                hip.append(f"        out[idx] = {scalar_expr};")
                hip.append("    }")
                hip.append("}")
            elif op_type.lower() not in ("input", "output"):
                tpl = self._ensure_template(node, node_idx)
                if tpl:
                    hip.append(coerce_str(adapt_backend_module(tpl).require_symbol("get")("body", "")))

        hip.append("void evaluate_rocm(const std::vector<float*>& inputs, std::vector<float*>& outputs) {")
        hip.append("    // 1. Dynamic Memory Allocation (Host to Device)")

        graph_inps = _graph_str_list(self.graph, "inputs")
        graph_outs = _graph_str_list(self.graph, "outputs")
        node_buffers: dict[str, str] = {inp_name: f"inputs[{i}]" for i, inp_name in enumerate(graph_inps)}
        last_consumer = compute_liveness(self.graph)
        outputs_set: set[str] = set(graph_outs)
        inputs_set: set[str] = set(graph_inps)

        node_idx = 0
        for node in _graph_nodes_values(self.graph):
            if adapt_backend_module(node).get_symbol("op_type", "") == "Input":
                continue
            self._emit_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, hip)
            node_idx += 1

        hip.append("    HIP_CHECK(hipDeviceSynchronize());")
        hip.append("}")
        return "\n".join(hip)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Compile IRGraph into ahead-of-time ROCm HIP executable artifact.

        Args:
            graph (IRGraph): Target computation graph to compile.
            **kwargs (BackendArg): Compiler options such as optimization levels and target arch.

        Returns:
            BackendArg: Executable compiled artifact.
        """
        from ml_switcheroo_compiler.backends.base_generator import CompiledArtifact
        from ml_switcheroo_compiler.backends.hardware_compilers import HIPCompiler

        hip_source = self.generate()
        compiler = HIPCompiler()
        compile_opts = kwargs.get("compile_options", None)
        comp_res = compiler.compile(hip_source, options=compile_opts if isinstance(compile_opts, list) else None)

        def runner(*args: BackendArg, **runner_kwargs: BackendArg) -> BackendArg:
            """Execute compiled ROCm artifact with runtime dispatch or fallback.

            Args:
                *args (BackendArg): Input tensor or buffer arguments.
                **runner_kwargs (BackendArg): Optional execution options.

            Returns:
                BackendArg: Output tensor or tuple of output tensors.
            """
            if ROCmRunner.is_available():
                try:
                    runner_inst = ROCmRunner()
                    input_buffers = _prepare_rocm_input_buffers(graph, args)
                    out_dict = adapt_backend_module(runner_inst).require_symbol("execute_graph")(graph, input_buffers)
                    if len(graph.outputs) == 1:
                        return out_dict.get(graph.outputs[0])
                    return tuple(out_dict.get(out_name) for out_name in graph.outputs)
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

            return _evaluate_rocm_fallback(graph, args)

        return CompiledArtifact(
            callable_fn=runner,
            binary_bytes=comp_res.assembly_or_ptx.encode("utf-8") if comp_res.assembly_or_ptx else b"",
            source_code=hip_source,
            metadata={"backend": "rocm", "compiler_cli": comp_res.compiler_cli, "success": comp_res.success},
        )
