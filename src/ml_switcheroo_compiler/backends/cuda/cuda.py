"""CUDA backend generator."""

from __future__ import annotations

import logging
from types import ModuleType

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.cuda._cuda_codegen_base import _CudaCodeGenBase as _CudaCodeGenBase
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    CUDA_SYNTH_MATH_MAP as CUDA_SYNTH_MATH_MAP,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    KernelArgType as KernelArgType,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _as_eager_dict as _as_eager_dict,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _as_tpl_opt as _as_tpl_opt,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _calculate_node_bytes as _calculate_node_bytes,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _evaluate_cuda_fallback as _evaluate_cuda_fallback,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _extract_kernel_name as _extract_kernel_name,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _marshall_kernel_params as _marshall_kernel_params,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _prepare_cuda_input_buffers as _prepare_cuda_input_buffers,
)
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _synthesize_cuda_kernel as _synthesize_cuda_kernel,
)
from ml_switcheroo_compiler.backends.cuda._cuda_runner_exec import CUDARunner as CUDARunner
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int_list,
    coerce_operand_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.hardware_config_models import (
    compute_liveness,
    resolve_hardware_launch_grid_and_args,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.registry import register_backend

cupy: ModuleType | None = None


@register_backend("cuda")
class CudaCodeGenerator(_CudaCodeGenBase):
    """CUDA C++ Code Generator."""

    def _emit_node(
        self,
        node: IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        cuda: list[str],
    ) -> None:
        """Emit CUDA kernel allocation, launch, and buffer freeing for a single IRNode.

        Args:
            node (IRNode): IR node to emit.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            cuda (list[str]): Output CUDA source lines.
        """
        op_type = str(adapt_backend_module(node).get_symbol("op_type", ""))
        if op_type.lower() in ("allreduce", "ncclallreduce", "allgather", "reducescatter", "broadcast"):
            self._emit_nccl_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, cuda)
            return

        if op_type.lower() == "fusedelementwise":
            self._emit_fused_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, cuda)
            return

        if op_type.lower() in ("loop", "whileloop"):
            self._emit_loop_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, cuda)
            return

        tpl = self._ensure_template(node, node_idx)
        if not tpl:
            return
        raw_outs = _as_eager_dict(adapt_backend_module(node).get_symbol("attributes", {})).get("outputs") or (
            node.outputs if adapt_backend_module(node).has_symbol("outputs") and len(node.outputs) > 1 else None
        )
        num_elements, bytes_alloc = _calculate_node_bytes(node)

        if raw_outs and isinstance(raw_outs, (list, tuple)) and len(raw_outs) > 1:
            out_names = []
            for out_idx, out_id in enumerate(raw_outs):
                sub_out_name = f"d_out_{node_idx}_{out_idx}"
                node_buffers[str(out_id)] = sub_out_name
                out_names.append(sub_out_name)
                cuda.append(f"    float* {sub_out_name};")
                cuda.append(f"    CUDA_CHECK(cudaMalloc(&{sub_out_name}, {bytes_alloc}));")
            node_buffers[node.id] = out_names[0]
            out_name = ", ".join(out_names)
        else:
            out_name = f"d_out_{node_idx}"
            node_buffers[node.id] = out_name
            cuda.append(f"    float* {out_name};")
            cuda.append(f"    CUDA_CHECK(cudaMalloc(&{out_name}, {bytes_alloc}));")
        wg = coerce_int_list(adapt_backend_module(tpl).require_op("get")("workgroup_size", [1, 1, 1]) or [1, 1, 1])
        cuda.append(f"    dim3 block_{node_idx}({wg[0]}, {wg[1]}, {wg[2]});")

        (grid_x, grid_y, grid_z), launch_args = resolve_hardware_launch_grid_and_args(
            node, tpl, node_buffers, out_name, num_elements, node_idx, self.graph
        )
        cuda.append(f"    dim3 grid_{node_idx}({grid_x}, {grid_y}, {grid_z});")

        kernel_fn = _extract_kernel_name(
            coerce_str(adapt_backend_module(tpl).require_op("get")("body", "")), op_type.lower()
        )
        cuda.append(f"    {kernel_fn}<<<grid_{node_idx}, block_{node_idx}>>>({', '.join(launch_args)});")
        cuda.append("    CUDA_CHECK(cudaGetLastError());")

        self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, cuda)

    def generate(self) -> str:
        """Generate CUDA C++ code with dynamic buffer allocations and memory lifetime tracking.

        Returns:
            str: Generated CUDA strings.
        """
        has_collectives: bool = any(
            str(adapt_backend_module(n).get_symbol("op_type", "")).lower()
            in ("allreduce", "ncclallreduce", "allgather", "reducescatter", "broadcast")
            for n in (self.graph.nodes.values() if adapt_backend_module(self.graph).get_symbol("nodes", None) else ())
        )

        cuda: list[str] = [
            "// CUDA Generated by ml-switcheroo-compiler",
            "#include <cuda_runtime.h>",
            "#include <iostream>",
            "#include <vector>",
        ]
        if has_collectives:
            cuda.extend(
                [
                    "#include <nccl.h>",
                    "",
                    "#define NCCL_CHECK(cmd) ",
                    "    do { ",
                    "        ncclResult_t r = cmd; ",
                    "        if (r != ncclSuccess) { ",
                    (
                        '            std::cerr << "NCCL Error: " << ncclGetErrorString(r) << " at " << __FILE__ << '
                        '":" << __LINE__ << std::endl; '
                    ),
                    "            exit(r); ",
                    "        } ",
                    "    } while (0)",
                ]
            )

        cuda.extend(
            [
                "",
                "#define CUDA_CHECK(call) ",
                "    do { ",
                "        cudaError_t err = call; ",
                "        if (err != cudaSuccess) { ",
                (
                    '            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << '
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
        )

        for node_idx, node in enumerate(
            self.graph.nodes.values() if adapt_backend_module(self.graph).get_symbol("nodes", None) else ()
        ):
            op_type: str = str(adapt_backend_module(node).get_symbol("op_type", ""))
            if op_type.lower() == "fusedelementwise":
                scalar_expr = str(node.attributes.get("scalar_expr", "in0"))
                in_params = ", ".join([f"const float* in_{k}" for k in range(len(node.inputs))])
                load_lines = "\n".join([f"        float in{k} = in_{k}[idx];" for k in range(len(node.inputs))])
                cuda.append(f"__global__ void fused_elementwise_kernel_{node_idx}({in_params}, float* out, int N) {{")
                cuda.append("    int idx = blockDim.x * blockIdx.x + threadIdx.x;")
                cuda.append("    if (idx < N) {")
                if load_lines:
                    cuda.append(load_lines)
                cuda.append(f"        out[idx] = {scalar_expr};")
                cuda.append("    }")
                cuda.append("}")
            elif op_type.lower() not in ("input", "output"):
                tpl = self._ensure_template(node, node_idx)
                if tpl:
                    cuda.append(coerce_str(adapt_backend_module(tpl).require_op("get")("body", "")))

        eval_fn_sig: str = (
            (
                "void evaluate_cuda(const std::vector<float*>& inputs, std::vector<float*>& outputs, "
                "ncclComm_t comm = 0, cudaStream_t stream = 0) {"
            )
            if has_collectives
            else "void evaluate_cuda(const std::vector<float*>& inputs, std::vector<float*>& outputs) {"
        )
        cuda.append(eval_fn_sig)
        cuda.append("    // 1. Dynamic Memory Allocation and Kernel Dispatch")

        node_buffers: dict[str, str] = {
            str(inp_name): f"inputs[{i}]"
            for i, inp_name in enumerate(coerce_operand_list(adapt_backend_module(self.graph).get_symbol("inputs", [])))
        }
        last_consumer = compute_liveness(self.graph)
        outputs_set = {str(x) for x in coerce_operand_list(adapt_backend_module(self.graph).get_symbol("outputs", []))}
        inputs_set = {str(x) for x in coerce_operand_list(adapt_backend_module(self.graph).get_symbol("inputs", []))}

        node_idx = 0
        for node in self.graph.nodes.values() if adapt_backend_module(self.graph).get_symbol("nodes", None) else ():
            if adapt_backend_module(node).get_symbol("op_type", "") == "Input":
                continue
            self._emit_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, cuda)
            node_idx += 1

        cuda.append("    CUDA_CHECK(cudaDeviceSynchronize());")
        cuda.append("}")
        return "\n".join(cuda)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Compile IRGraph into ahead-of-time CUDA executable artifact.

        Args:
            graph (IRGraph): Target computation graph to compile.
            **kwargs (BackendArg): Compiler options such as optimization levels and target arch.

        Returns:
            BackendArg: Executable compiled artifact.
        """
        from ml_switcheroo_compiler.backends.base_generator import CompiledArtifact
        from ml_switcheroo_compiler.backends.hardware_compilers import CUDACompiler

        cuda_source = self.generate()
        compiler = CUDACompiler()
        compile_opts = kwargs.get("compile_options", None)
        comp_res = compiler.compile(cuda_source, options=compile_opts if isinstance(compile_opts, list) else None)

        def runner(*args: BackendArg, **runner_kwargs: BackendArg) -> BackendArg:
            """Execute compiled CUDA artifact with runtime dispatch or fallback.

            Args:
                *args (BackendArg): Input tensor or buffer arguments.
                **runner_kwargs (BackendArg): Optional execution options.

            Returns:
                BackendArg: Output tensor or tuple of output tensors.
            """
            if CUDARunner.is_available():
                try:
                    runner_inst = CUDARunner()
                    input_buffers = _prepare_cuda_input_buffers(graph, args)
                    out_dict = _as_eager_dict(
                        adapt_backend_module(runner_inst).require_op("execute_graph")(graph, input_buffers)
                    )
                    if len(graph.outputs) == 1:
                        return out_dict.get(graph.outputs[0])
                    return tuple(out_dict.get(out_name) for out_name in graph.outputs)
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

            return _evaluate_cuda_fallback(graph, args)

        return CompiledArtifact(
            callable_fn=runner,
            binary_bytes=comp_res.assembly_or_ptx.encode("utf-8") if comp_res.assembly_or_ptx else b"",
            source_code=cuda_source,
            metadata={"backend": "cuda", "compiler_cli": comp_res.compiler_cli, "success": comp_res.success},
        )
