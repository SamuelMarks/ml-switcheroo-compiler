"""Base CUDA code emission layer handling NCCL, fusion, loops, and template lookup."""

from __future__ import annotations

import os

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    _as_eager_dict,
    _calculate_node_bytes,
    _synthesize_cuda_kernel,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_operand_list,
)
from ml_switcheroo_compiler.backends.hardware_config_models import (
    GridDimensionConfig,
    HardwareTemplateConfig,
    load_hardware_templates,
    resolve_hardware_launch_grid_and_args,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


class _CudaCodeGenBase(BaseGenerator):
    """Base layer for CUDA C++ kernel and host orchestration emission."""

    def __init__(self, graph: IRGraph, delegates: list[CodeGeneratorVisitor] | None = None) -> None:
        """Initialize CudaCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Optional[list[CodeGeneratorVisitor]], optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        yaml_dir: str = os.path.join(os.path.dirname(__file__), "cuda_orchestration")
        yaml_path: str = os.path.join(os.path.dirname(__file__), "cuda_templates.yaml")
        self.config = load_hardware_templates(yaml_dir, yaml_path)

    def _emit_nccl_node(
        self,
        node: IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        cuda: list[str],
    ) -> None:
        """Emit NCCL collective communication call for a distributed IRNode.

        Args:
            node (IRNode): The collective IR node.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            cuda (list[str]): Output CUDA source lines.
        """
        op_type = str(adapt_backend_module(node).get_symbol("op_type", ""))
        out_name = f"d_out_{node_idx}"
        node_buffers[node.id] = out_name
        num_elements, bytes_alloc = _calculate_node_bytes(node)
        cuda.append(f"    float* {out_name};")
        cuda.append(f"    CUDA_CHECK(cudaMalloc(&{out_name}, {bytes_alloc}));")
        in_name = (
            node_buffers.get(node.inputs[0], "inputs[0]")
            if adapt_backend_module(node).get_symbol("inputs", None)
            else out_name
        )
        op_lower = op_type.lower()
        if op_lower in ("allreduce", "ncclallreduce"):
            cuda.append(
                f"    NCCL_CHECK(ncclAllReduce((const void*){in_name}, (void*){out_name}, {num_elements}, "
                "ncclFloat, ncclSum, comm, stream));"
            )
        elif op_lower == "allgather":
            cuda.append(
                f"    NCCL_CHECK(ncclAllGather((const void*){in_name}, (void*){out_name}, {num_elements}, "
                "ncclFloat, comm, stream));"
            )
        elif op_lower == "reducescatter":
            cuda.append(
                f"    NCCL_CHECK(ncclReduceScatter((const void*){in_name}, (void*){out_name}, {num_elements}, "
                "ncclFloat, ncclSum, comm, stream));"
            )
        elif op_lower == "broadcast":
            root = _as_eager_dict(adapt_backend_module(node).get_symbol("attributes", {})).get("root", 0)
            cuda.append(
                f"    NCCL_CHECK(ncclBroadcast((const void*){in_name}, (void*){out_name}, {num_elements}, "
                f"ncclFloat, {root}, comm, stream));"
            )
        self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, cuda)

    def _free_dead_buffers(
        self,
        node: IRNode,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        cuda: list[str],
    ) -> None:
        """Emit cudaFree calls for dead input buffers whose lifetime has ended.

        Args:
            node (IRNode): The active IRNode.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            cuda (list[str]): Output CUDA source lines.
        """
        for inp in [str(x) for x in coerce_operand_list(adapt_backend_module(node).get_symbol("inputs", []))]:
            if last_consumer.get(inp) == node.id and inp not in outputs_set and inp not in inputs_set:
                buf_to_free = node_buffers.get(inp)
                if buf_to_free:
                    cuda.append(f"    CUDA_CHECK(cudaFree({buf_to_free}));")

    def _emit_fused_node(
        self,
        node: IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        cuda: list[str],
    ) -> None:
        """Emit fused elementwise CUDA kernel allocation, launch, and buffer freeing.

        Args:
            node (IRNode): Fused IR node to emit.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            cuda (list[str]): Output CUDA source lines.
        """
        out_name = f"d_out_{node_idx}"
        node_buffers[node.id] = out_name
        num_elements, bytes_alloc = _calculate_node_bytes(node)
        cuda.append(f"    float* {out_name};")
        cuda.append(f"    CUDA_CHECK(cudaMalloc(&{out_name}, {bytes_alloc}));")
        cuda.append(f"    dim3 block_{node_idx}(256, 1, 1);")
        (grid_x, grid_y, grid_z), launch_args = resolve_hardware_launch_grid_and_args(
            node, {}, node_buffers, out_name, num_elements, node_idx, self.graph
        )
        cuda.append(f"    dim3 grid_{node_idx}({grid_x}, {grid_y}, {grid_z});")
        kernel_fn = f"fused_elementwise_kernel_{node_idx}"
        cuda.append(f"    {kernel_fn}<<<grid_{node_idx}, block_{node_idx}>>>({', '.join(launch_args)});")
        cuda.append("    CUDA_CHECK(cudaGetLastError());")
        self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, cuda)

    def _emit_loop_node(
        self,
        node: IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        cuda: list[str],
    ) -> None:
        """Emit CUDA host loop execution code for Loop and WhileLoop nodes with loop options.

        Args:
            node (IRNode): The loop IR node.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            cuda (list[str]): Output CUDA source lines.
        """
        clean_id = f"node_{node_idx}"
        out_buf = f"d_out_{node_idx}"
        node_buffers[str(adapt_backend_module(node).get_symbol("id", clean_id))] = out_buf
        _, bytes_alloc = _calculate_node_bytes(node)
        cuda.append(f"    float* {out_buf};")
        cuda.append(f"    CUDA_CHECK(cudaMalloc(&{out_buf}, {bytes_alloc}));")

        attrs = _as_eager_dict(adapt_backend_module(node).get_symbol("attributes", {}))
        max_iters = attrs.get("maximum_iterations") or attrs.get("max_iters", 10)
        parallel_iters = attrs.get("parallel_iterations")
        swap_memory = attrs.get("swap_memory")
        shape_invariants = attrs.get("shape_invariants")

        annotations: list[str] = []
        if parallel_iters is not None:
            annotations.append(f"parallel_iterations={parallel_iters}")
        if swap_memory is not None:
            annotations.append(f"swap_memory={swap_memory}")
        if max_iters is not None:
            annotations.append(f"maximum_iterations={max_iters}")
        if shape_invariants is not None:
            annotations.append(f"shape_invariants={shape_invariants}")
        if annotations:
            cuda.append(f"    // Loop annotations: {', '.join(annotations)}")
        if swap_memory:
            cuda.append(f"    // Swap memory policy enabled for {out_buf}")
        if parallel_iters is not None:
            cuda.append(f"    #pragma unroll {parallel_iters}")

        cuda.append(f"    for (int _iter_{node_idx} = 0; _iter_{node_idx} < {max_iters}; ++_iter_{node_idx}) {{")
        cuda.append(f"        // Loop iteration body for {clean_id}")
        cuda.append("    }")
        self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, cuda)

    def _ensure_template(
        self,
        node: IRNode,
        node_idx: int,
    ) -> (
        HardwareTemplateConfig | dict[str, str | int | float | list[int] | None] | list[str] | str | int | float | None
    ):
        """Ensure an op template exists in self.config.templates, synthesizing elementwise kernels if needed.

        Args:
            node (IRNode): Target IR node.
            node_idx (int): Current node index.

        Returns:
            Optional[Union[HardwareTemplateConfig, dict[str, Union[str, int, float, list[int], None]], list[str],
                str]]: Existing or synthesized template.

        Raises:
            BackendNotSupportedError: If `_ensure_template` encounters an unsupported operation or missing template.
        """
        op_type = str(adapt_backend_module(node).get_symbol("op_type", ""))
        if op_type.lower() in ("input", "output", "fusedelementwise", "loop", "whileloop"):
            return None
        if "unsupported" in op_type.lower():
            raise BackendNotSupportedError(f"Operation '{op_type}' is not supported by cuda backend.")
        tpl = self.config.templates.get(op_type.lower())
        if not tpl:
            clean_op = op_type.lower().replace("-", "_").replace(".", "_")
            inps = [str(x) for x in coerce_operand_list(adapt_backend_module(node).get_symbol("inputs", []) or [])]
            raw_se = _as_eager_dict(adapt_backend_module(node).get_symbol("attributes", {})).get("scalar_expr")
            scalar_expr = str(raw_se) if raw_se is not None else None
            body = _synthesize_cuda_kernel(clean_op, inps, scalar_expr)
            tpl = HardwareTemplateConfig(
                body=body,
                grid_calc=GridDimensionConfig(
                    x=f"({{num_elements}} + block_{node_idx}.x - 1) / block_{node_idx}.x", y="1", z="1"
                ),
                workgroup_size=[256, 1, 1],
            )
            self.config.templates[op_type.lower()] = tpl
        return tpl
