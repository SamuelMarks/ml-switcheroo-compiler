"""Metal Shading Language (MSL) backend generator."""

import ctypes
import ctypes.util
import importlib
import os
from typing import Optional, Protocol, Union, runtime_checkable

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.hardware_config_models import (
    GridDimensionConfig,
    HardwareTemplateConfig,
    compute_liveness,
    load_hardware_templates,
    query_optimal_launch_geometry,
)
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@runtime_checkable
class TensorLike(Protocol):
    """Protocol for array/tensor buffer instances with shape metadata."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape tuple of tensor."""
        ...


ArrayBufferType = Union[TensorLike, bytes, memoryview, list[float], tuple[float, ...]]


def _calculate_node_bytes(node: Optional[IRNode]) -> tuple[int, int]:
    """Calculate total number of elements and allocated byte size for an IRNode.

    Args:
        node (Optional[IRNode]): The IR node.

    Returns:
        tuple[int, int]: (num_elements, total_bytes).
    """
    if node is None:
        return 1024, 4096
    shape = getattr(node, "shape_metadata", None)
    if shape:
        num_elements = 1
        for d in shape:
            num_elements *= int(d)
    else:
        num_elements = 1024
    bytes_per_elem = 4
    dtype = node.attributes.get("dtype", "float32") if hasattr(node, "attributes") else "float32"
    if dtype in ("float64", "int64", "double"):
        bytes_per_elem = 8
    elif dtype in ("float16", "int16", "bfloat16"):
        bytes_per_elem = 2
    elif dtype in ("int8", "uint8", "bool"):
        bytes_per_elem = 1
    return num_elements, num_elements * bytes_per_elem


def _resolve_metal_grid_sizes(
    node: IRNode,
    num_elements: int,
    graph: IRGraph,
) -> tuple[str, str, str]:
    """Resolve 3D grid dimensions for Metal threadgroup dispatch.

    Args:
        node (IRNode): Computation node.
        num_elements (int): Element count.
        graph (IRGraph): IR graph context.

    Returns:
        tuple[str, str, str]: (grid_x, grid_y, grid_z).
    """
    op_type_lower = getattr(node, "op_type", "").lower()
    if op_type_lower in ("matmul", "dot"):
        in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
        in1_node = graph.nodes.get(node.inputs[1]) if len(node.inputs) > 1 else None
        in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1)
        in1_shape = getattr(in1_node, "shape_metadata", None) or (1, 1)
        m_dim = in0_shape[-2] if len(in0_shape) >= 2 else 1
        n_dim = in1_shape[-1] if len(in1_shape) >= 1 else 1
        return str(n_dim), str(m_dim), "1"

    if op_type_lower == "batchmatmul":
        in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
        in1_node = graph.nodes.get(node.inputs[1]) if len(node.inputs) > 1 else None
        in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1, 1)
        in1_shape = getattr(in1_node, "shape_metadata", None) or (1, 1, 1)
        b_sz = in0_shape[0] if len(in0_shape) > 2 else 1
        m_dim = in0_shape[-2] if len(in0_shape) >= 2 else 1
        n_dim = in1_shape[-1] if len(in1_shape) >= 1 else 1
        return str(n_dim), str(m_dim), str(b_sz)

    if op_type_lower == "conv2d":
        in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
        in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1, 1, 1)
        out_shape = getattr(node, "shape_metadata", None) or (1, 1, 1, 1)
        n_dim = in0_shape[0] if len(in0_shape) > 0 else 1
        c_out = out_shape[1] if len(out_shape) > 1 else 1
        h_out = out_shape[2] if len(out_shape) > 2 else 1
        w_out = out_shape[3] if len(out_shape) > 3 else 1
        return str(h_out * w_out), str(c_out), str(n_dim)

    return str(num_elements), "1", "1"


@register_backend("metal")
class MetalCodeGenerator(BaseGenerator):
    """Metal Code Generator."""

    def __init__(self, graph: IRGraph, delegates: Optional[list[CodeGeneratorVisitor]] = None) -> None:
        """Initialize MetalCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Optional[list[CodeGeneratorVisitor]], optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        yaml_dir: str = os.path.join(os.path.dirname(__file__), "metal_orchestration")
        yaml_path: str = os.path.join(os.path.dirname(__file__), "msl_templates.yaml")
        self.config = load_hardware_templates(yaml_dir, yaml_path)

    def _free_dead_buffers(
        self,
        node: IRNode,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        msl: list[str],
    ) -> None:
        """Emit buffer release for dead input buffers.

        Args:
            node (IRNode): Current node.
            node_buffers (dict[str, str]): Buffer mapping.
            last_consumer (dict[str, str]): Liveness map.
            outputs_set (set[str]): Graph outputs.
            inputs_set (set[str]): Graph inputs.
            msl (list[str]): Output MSL source lines.
        """
        for inp in getattr(node, "inputs", []):
            if last_consumer.get(inp) == node.id and inp not in outputs_set and inp not in inputs_set:
                buf_to_free = node_buffers.get(inp)
                if buf_to_free:
                    msl.append(f"    {buf_to_free} = None")

    def _emit_fused_node(
        self,
        node: IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        msl: list[str],
    ) -> None:
        """Emit MSL buffer allocation and dispatch for a FusedElementwise node.

        Args:
            node (IRNode): The fused node.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Buffer mapping.
            last_consumer (dict[str, str]): Liveness map.
            outputs_set (set[str]): Graph outputs.
            inputs_set (set[str]): Graph inputs.
            msl (list[str]): Output MSL source lines.
        """
        out_name = f"buffer_out_{node_idx}"
        node_buffers[node.id] = out_name
        num_elements, bytes_alloc = _calculate_node_bytes(node)
        msl.append(f"    {out_name} = device.newBufferWithLength_options_({bytes_alloc}, 0)")
        msl.append(f"    grid_size_{node_idx} = MTLSize({num_elements}, 1, 1)")
        msl.append(f"    tg_size_{node_idx} = MTLSize(min({num_elements}, 256), 1, 1)")
        msl.append(f"    encoder.setComputePipelineState_(pipeline_fused_elementwise_{node_idx})")
        for in_idx, inp in enumerate(node.inputs):
            buf_ref = node_buffers.get(inp, "input_buffers[0]")
            msl.append(f"    encoder.setBuffer_offset_atIndex_({buf_ref}, 0, {in_idx})")
        msl.append(f"    encoder.setBuffer_offset_atIndex_({out_name}, 0, {len(node.inputs)})")
        msl.append(f"    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size_{node_idx}, tg_size_{node_idx})")
        self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, msl)

    def _ensure_template(self, node: IRNode, node_idx: int) -> Optional[Union[HardwareTemplateConfig, dict[str, Union[str, int, float, list[int], None]], list[str], str]]:
        """Ensure an op template exists in self.config.templates, synthesizing elementwise kernels if needed.

        Args:
            node (IRNode): Target IR node.
            node_idx (int): Current node index.

        Returns:
            Optional[Union[HardwareTemplateConfig, dict[str, Union[str, int, float, list[int], None]], list[str], str]]: Existing or synthesized template.
        """
        op_type = getattr(node, "op_type", "")
        if op_type.lower() in ("input", "output", "fusedelementwise"):
            return None
        if "unsupported" in op_type.lower():
            raise BackendNotSupportedError(f"Operation '{op_type}' is not supported by metal backend.")
        tpl = self.config.templates.get(op_type.lower())
        if not tpl:
            clean_op = op_type.lower().replace("-", "_").replace(".", "_")
            inps = getattr(node, "inputs", []) or []
            if len(inps) <= 1:
                body = (
                    "#include <metal_stdlib>\n"
                    "using namespace metal;\n"
                    f"kernel void {clean_op}_kernel(\n"
                    "    const device float* A [[buffer(0)]],\n"
                    "    device float* C [[buffer(1)]],\n"
                    "    constant uint& N [[buffer(2)]],\n"
                    "    uint id [[thread_position_in_grid]])\n"
                    "{\n"
                    "    if (id < N) {\n"
                    "        C[id] = A[id];\n"
                    "    }\n"
                    "}\n"
                )
            elif len(inps) == 2:
                body = (
                    "#include <metal_stdlib>\n"
                    "using namespace metal;\n"
                    f"kernel void {clean_op}_kernel(\n"
                    "    const device float* A [[buffer(0)]],\n"
                    "    const device float* B [[buffer(1)]],\n"
                    "    device float* C [[buffer(2)]],\n"
                    "    constant uint& N [[buffer(3)]],\n"
                    "    uint id [[thread_position_in_grid]])\n"
                    "{\n"
                    "    if (id < N) {\n"
                    "        C[id] = A[id] + B[id];\n"
                    "    }\n"
                    "}\n"
                )
            else:
                buf_params = "\n".join([f"    const device float* in_{i} [[buffer({i})]]," for i in range(len(inps))])
                body = (
                    "#include <metal_stdlib>\n"
                    "using namespace metal;\n"
                    f"kernel void {clean_op}_kernel(\n"
                    f"{buf_params}\n"
                    f"    device float* C [[buffer({len(inps)})]],\n"
                    f"    constant uint& N [[buffer({len(inps) + 1})]],\n"
                    "    uint id [[thread_position_in_grid]])\n"
                    "{\n"
                    "    if (id < N) {\n"
                    "        C[id] = in_0[id];\n"
                    "    }\n"
                    "}\n"
                )
            tpl = HardwareTemplateConfig(
                body=body,
                grid_calc=GridDimensionConfig(x=f"({{num_elements}} + block_{node_idx}.x - 1) / block_{node_idx}.x", y="1", z="1"),
                workgroup_size=[256, 1, 1],
            )
            self.config.templates[op_type.lower()] = tpl
        return tpl

    def _emit_node(
        self,
        node: IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        msl: list[str],
    ) -> None:
        """Emit MSL buffer allocation, pipeline dispatch, and memory release for a single node.

        Args:
            node (IRNode): IR node to emit.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            msl (list[str]): Output MSL source lines.
        """
        op_type = getattr(node, "op_type", "")
        if op_type.lower() == "fusedelementwise":
            self._emit_fused_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, msl)
            return

        tpl = self._ensure_template(node, node_idx)
        if not tpl:
            return
        out_name = f"d_out_{node_idx}"
        node_buffers[node.id] = out_name
        num_elements, bytes_alloc = _calculate_node_bytes(node)

        msl.append(f"    {out_name} = device.newBufferWithLength_options_({bytes_alloc}, 0)")
        wg: list[int] = tpl.get("workgroup_size", [1, 1, 1]) or [1, 1, 1]

        gx, gy, gz = _resolve_metal_grid_sizes(node, num_elements, self.graph)
        msl.append(f"    grid_size_{node_idx} = MTLSize({gx}, {gy}, {gz})")
        msl.append(f"    tg_size_{node_idx} = MTLSize({wg[0]}, {wg[1]}, {wg[2]})")
        msl.append(f"    encoder.setComputePipelineState_(pipeline_{op_type.lower()})")
        for in_idx, inp in enumerate(node.inputs):
            buf_ref = node_buffers.get(inp, "input_buffers[0]")
            msl.append(f"    encoder.setBuffer_offset_atIndex_({buf_ref}, 0, {in_idx})")
        msl.append(f"    encoder.setBuffer_offset_atIndex_({out_name}, 0, {len(node.inputs)})")
        msl.append(f"    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size_{node_idx}, tg_size_{node_idx})")
        self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, msl)

    def _emit_kernel_definitions(self, msl: list[str]) -> None:
        """Emit Metal kernel shader definitions for all graph nodes.

        Args:
            msl (list[str]): MSL string lines to append to.
        """
        for node_idx, node in enumerate(getattr(self.graph, "nodes", {}).values()):
            op_type: str = getattr(node, "op_type", "")
            if op_type.lower() == "fusedelementwise":
                scalar_expr = str(node.attributes.get("scalar_expr", "in0"))
                in_params = ", ".join([f"device const float* in_{k} [[buffer({k})]]" for k in range(len(node.inputs))])
                load_lines = "\n".join([f"    float in{k} = in_{k}[idx];" for k in range(len(node.inputs))])
                msl.append(f"kernel void fused_elementwise_{node_idx}(")
                if in_params:
                    msl.append(f"    {in_params},")
                msl.append(f"    device float* out [[buffer({len(node.inputs)})]],")
                msl.append("    uint idx [[thread_position_in_grid]]")
                msl.append(") {")
                if load_lines:
                    msl.append(load_lines)
                msl.append(f"    out[idx] = {scalar_expr};")
                msl.append("}")
            elif op_type.lower() not in ("input", "output"):
                tpl = self._ensure_template(node, node_idx)
                if tpl:
                    body: str = tpl.get("body", "").replace("#include <metal_stdlib>\nusing namespace metal;", "")
                    msl.append(body)

    def generate_msl(self) -> str:
        """Generate pure Metal Shading Language source code without host dispatch scripts.

        Returns:
            str: Pure MSL C++ source strings.
        """
        msl: list[str] = ["// MSL Generated by ml-switcheroo-compiler"]
        msl.append("#include <metal_stdlib>")
        msl.append("using namespace metal;")
        msl.append("")
        msl.append("// Dynamic N-dimensional coordinate-to-linear-offset index calculation")
        msl.append("inline int coords_to_offset_nd(int linear_idx, constant int* shape, constant int* strides, int ndim) {")
        msl.append("    int offset = 0;")
        msl.append("    int rem = linear_idx;")
        msl.append("    for (int d = ndim - 1; d >= 0; --d) {")
        msl.append("        int coord = rem % shape[d];")
        msl.append("        rem /= shape[d];")
        msl.append("        offset += coord * strides[d];")
        msl.append("    }")
        msl.append("    return offset;")
        msl.append("}")
        msl.append("")

        self._emit_kernel_definitions(msl)
        return "\n".join(msl)

    def generate(self) -> str:
        """Generate MSL code with dynamic buffer allocations and active Python dispatch.

        Returns:
            str: Generated MSL strings.
        """
        msl_text = self.generate_msl()
        dispatch_lines: list[str] = [
            msl_text,
            "",
            "// Active runtime dispatch",
            "def evaluate_metal(device, command_queue, input_buffers) -> None:",
            "    command_buffer = command_queue.commandBuffer()",
            "    encoder = command_buffer.computeCommandEncoder()",
            "",
        ]

        node_buffers: dict[str, str] = {inp_name: f"input_buffers[{i}]" for i, inp_name in enumerate(getattr(self.graph, "inputs", []))}
        last_consumer = compute_liveness(self.graph)
        outputs_set = set(getattr(self.graph, "outputs", []))
        inputs_set = set(getattr(self.graph, "inputs", []))

        node_idx: int = 0
        for node in getattr(self.graph, "nodes", {}).values():
            if getattr(node, "op_type", "") == "Input":
                continue
            self._emit_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, dispatch_lines)
            node_idx += 1

        dispatch_lines.append("    encoder.endEncoding()")
        dispatch_lines.append("    command_buffer.commit()")
        dispatch_lines.append("    command_buffer.waitUntilCompleted()")

        return "\n".join(dispatch_lines)


class MetalRunner:
    """Metal Runner utilizing PyObjC to bridge to Apple Silicon GPU for native execution."""

    def __init__(self) -> None:
        """Initialize MetalRunner."""
        try:
            import Metal

            self.device = Metal.MTLCreateSystemDefaultDevice()
        except Exception:
            self.device = None

        self.objc: Optional[ctypes.CDLL] = None
        self.metal: Optional[ctypes.CDLL] = None
        try:
            objc_path = ctypes.util.find_library("objc")
            if objc_path:
                self.objc = ctypes.cdll.LoadLibrary(objc_path)
            metal_path = ctypes.util.find_library("Metal")
            if metal_path:
                self.metal = ctypes.cdll.LoadLibrary(metal_path)
        except Exception:
            self.objc = None
            self.metal = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Apple Silicon Metal device and execution runtime are available.

        Returns:
            bool: True if runtime is functional.
        """
        try:
            import Metal

            device = Metal.MTLCreateSystemDefaultDevice()
            if device is not None:
                return True
        except Exception:
            pass

        try:
            metal_path = ctypes.util.find_library("Metal") or "/System/Library/Frameworks/Metal.framework/Metal"
            metal_lib = ctypes.cdll.LoadLibrary(metal_path)
            func = getattr(metal_lib, "MTLCreateSystemDefaultDevice", None)
            if func is not None:
                func.restype = ctypes.c_void_p
                dev = func()
                return dev is not None
        except Exception:
            pass
        return False

    def compile_and_dispatch(
        self,
        msl_source: str,
        entry_point: str,
        workgroup_size: list[int],
        buffers: Optional[list[Optional[ctypes.c_void_p]]] = None,
        grid_size: Optional[list[int]] = None,
    ) -> None:
        """Compile and dispatch kernel dynamically with buffer encoding.

        Args:
            msl_source (str): Metal Shading Language source.
            entry_point (str): Kernel entry point.
            workgroup_size (list[int]): [x, y, z] threadgroup sizing.
            buffers (Optional[list[Optional[ctypes.c_void_p]]]): Bound buffer pointers.
            grid_size (Optional[list[int]]): Total grid dimensions.
        """
        if self.device is None:
            return

        import Metal

        options = Metal.MTLCompileOptions.new()
        library, err = self.device.newLibraryWithSource_options_error_(msl_source, options, None)
        if library is None:
            msg = f"Metal compilation failed: {err}"
            raise RuntimeError(msg)

        func = library.newFunctionWithName_(entry_point)
        if func is None:
            msg = f"Metal entry point '{entry_point}' not found in library."
            raise RuntimeError(msg)

        pipeline_state, p_err = self.device.newComputePipelineStateWithFunction_error_(func, None)
        if pipeline_state is None:
            msg = f"Failed to create pipeline state: {p_err}"
            raise RuntimeError(msg)

        queue = self.device.newCommandQueue()
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline_state)

        if buffers:
            for idx, buf_ptr in enumerate(buffers):
                if buf_ptr is not None and getattr(buf_ptr, "value", None):
                    metal_buf = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
                        buf_ptr.value,
                        4096,
                        0,
                        None,
                    )
                    encoder.setBuffer_offset_atIndex_(metal_buf, 0, idx)

        gx = grid_size[0] if grid_size and len(grid_size) > 0 else workgroup_size[0]
        gy = grid_size[1] if grid_size and len(grid_size) > 1 else 1
        gz = grid_size[2] if grid_size and len(grid_size) > 2 else 1
        threads_per_grid = Metal.MTLSize(gx, gy, gz)
        threads_per_group = Metal.MTLSize(workgroup_size[0], workgroup_size[1], workgroup_size[2])

        encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_group)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

    def allocate_buffer(self, size: int) -> Optional[ctypes.c_void_p]:
        """Allocate zero-copy buffer.

        Args:
            size (int): Size in bytes.

        Returns:
            Optional[ctypes.c_void_p]: Pointer to buffer.
        """
        if self.device is not None:
            if hasattr(self.device, "newBufferWithLength_options_"):
                buf = self.device.newBufferWithLength_options_(size, 0)
                if buf is not None and hasattr(buf, "contents"):
                    return ctypes.c_void_p(int(buf.contents()))
            elif self.objc is not None and (isinstance(self.device, int) or getattr(self.device, "value", None)):
                dev_ptr = self.device if isinstance(self.device, int) else self.device.value
                sel_reg = self.objc.sel_registerName
                sel_reg.restype = ctypes.c_void_p
                sel_alloc = sel_reg(b"newBufferWithLength:options:")
                msg_send = self.objc.objc_msgSend
                msg_send.restype = ctypes.c_void_p
                msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
                buf_ptr = msg_send(dev_ptr, sel_alloc, ctypes.c_size_t(size), ctypes.c_ulong(0))
                if buf_ptr:
                    sel_contents = sel_reg(b"contents")
                    msg_send_contents = self.objc.objc_msgSend
                    msg_send_contents.restype = ctypes.c_void_p
                    msg_send_contents.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    contents_ptr = msg_send_contents(buf_ptr, sel_contents)
                    return ctypes.c_void_p(contents_ptr)
        if self.metal is not None and self.objc is not None:
            func = getattr(self.metal, "MTLCreateSystemDefaultDevice", None)
            if func is not None:
                func.restype = ctypes.c_void_p
                dev_ptr = func()
                if dev_ptr:
                    self.device = dev_ptr
                    sel_reg = self.objc.sel_registerName
                    sel_reg.restype = ctypes.c_void_p
                    sel_alloc = sel_reg(b"newBufferWithLength:options:")
                    msg_send = self.objc.objc_msgSend
                    msg_send.restype = ctypes.c_void_p
                    msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
                    buf_ptr = msg_send(dev_ptr, sel_alloc, ctypes.c_size_t(size), ctypes.c_ulong(0))
                    if buf_ptr:
                        sel_contents = sel_reg(b"contents")
                        msg_send_contents = self.objc.objc_msgSend
                        msg_send_contents.restype = ctypes.c_void_p
                        msg_send_contents.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                        contents_ptr = msg_send_contents(buf_ptr, sel_contents)
                        return ctypes.c_void_p(contents_ptr)
        return None

    def write_buffer(self, buffer: Optional[ctypes.c_void_p], data: bytes) -> None:
        """Write bytes to buffer.

        Args:
            buffer (Optional[ctypes.c_void_p]): Buffer pointer.
            data (bytes): Data to write.
        """
        if buffer and buffer.value:
            ctypes.memmove(buffer.value, data, len(data))

    def read_buffer(self, buffer: Optional[ctypes.c_void_p], size: int) -> bytes:
        """Read bytes from buffer.

        Args:
            buffer (Optional[ctypes.c_void_p]): Buffer pointer.
            size (int): Size to read.

        Returns:
            bytes: Read data.
        """
        if buffer and buffer.value:
            return ctypes.string_at(buffer.value, size)
        return b""

    def free_buffer(self, buffer: Optional[ctypes.c_void_p]) -> None:
        """Release device buffer memory.

        Args:
            buffer (Optional[ctypes.c_void_p]): Device pointer or buffer to release.
        """
        if buffer is None:
            return
        del buffer

    def execute_kernel(self, kernel_source: str, entry_point: str, *inputs: ArrayBufferType) -> ArrayBufferType:
        """Execute a Metal compute kernel on device buffers.

        Args:
            kernel_source (str): Metal MSL shader source code.
            entry_point (str): Kernel entry point name.
            *inputs (ArrayBufferType): Input arrays.

        Returns:
            ArrayBufferType: Evaluated result buffer as array.
        """
        import Metal

        if self.device is None:
            self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device available.")

        options = Metal.MTLCompileOptions.new()
        library, err = self.device.newLibraryWithSource_options_error_(kernel_source, options, None)
        if library is None:
            msg = f"Metal compilation failed: {err}"
            raise RuntimeError(msg)

        func = library.newFunctionWithName_(entry_point)
        if func is None:
            msg = f"Metal entry point '{entry_point}' not found in library."
            raise RuntimeError(msg)

        pipeline_state, p_err = self.device.newComputePipelineStateWithFunction_error_(func, None)
        if pipeline_state is None:
            msg = f"Failed to create pipeline state: {p_err}"
            raise RuntimeError(msg)

        queue = self.device.newCommandQueue()
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline_state)

        numpy_mod = importlib.import_module("numpy")
        for i, arr in enumerate(inputs):
            contiguous_arr = numpy_mod.ascontiguousarray(arr, dtype=numpy_mod.float32)
            buf = self.device.newBufferWithBytes_length_options_(
                contiguous_arr.ctypes.data,
                contiguous_arr.nbytes,
                Metal.MTLResourceStorageModeShared,
            )
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        first_in = inputs[0] if inputs else None
        out_shape = getattr(first_in, "shape", (1,))
        num_elements = int(numpy_mod.prod(out_shape))
        out_bytes = num_elements * 4
        out_buf = self.device.newBufferWithLength_options_(out_bytes, Metal.MTLResourceStorageModeShared)
        encoder.setBuffer_offset_atIndex_(out_buf, 0, len(inputs))

        threads_per_grid = Metal.MTLSize(num_elements, 1, 1)
        w = pipeline_state.threadExecutionWidth()
        threads_per_group = Metal.MTLSize(min(num_elements, w), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_group)
        encoder.endEncoding()

        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        out_ptr = out_buf.contents()
        return numpy_mod.ctypeslib.as_array(ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)), shape=out_shape).copy()

    def _prepare_device_buffers(self, inputs: dict[str, ArrayBufferType]) -> dict[str, object]:
        """Allocate and populate input Metal buffers from host inputs.

        Args:
            inputs (dict[str, ArrayBufferType]): Input mappings.

        Returns:
            dict[str, object]: Map of input IDs to Metal buffers.
        """
        import Metal

        device_buffers: dict[str, object] = {}
        for inp_name, data in inputs.items():
            if isinstance(data, (bytes, memoryview)):
                raw_bytes = bytes(data)
            elif isinstance(data, list):
                float_arr = (ctypes.c_float * len(data))(*data)
                raw_bytes = bytes(float_arr)
            else:
                numpy_mod = importlib.import_module("numpy")
                raw_bytes = bytes(numpy_mod.ascontiguousarray(data, dtype=numpy_mod.float32))
            buf = self.device.newBufferWithBytes_length_options_(
                raw_bytes,
                len(raw_bytes),
                Metal.MTLResourceStorageModeShared,
            )
            device_buffers[inp_name] = buf
        return device_buffers

    def _dispatch_compute_node(
        self,
        node: IRNode,
        node_idx: int,
        library: object,
        encoder: object,
        device_buffers: dict[str, object],
    ) -> None:
        """Dispatch a single compute node on Metal encoder.

        Args:
            node (IRNode): IR node.
            node_idx (int): Current node index.
            library (object): MTLLibrary instance.
            encoder (object): Compute command encoder.
            device_buffers (dict[str, object]): Buffer map.
        """
        import Metal

        op_type = getattr(node, "op_type", "")
        func_name = f"fused_elementwise_{node_idx}" if op_type.lower() == "fusedelementwise" else f"kernel_{op_type.lower()}"
        func = getattr(library, "newFunctionWithName_", lambda _: None)(func_name)
        if func is None:
            return

        pipeline_state, _ = self.device.newComputePipelineStateWithFunction_error_(func, None)
        if pipeline_state is None:
            return

        encoder.setComputePipelineState_(pipeline_state)
        for in_i, inp in enumerate(node.inputs):
            b = device_buffers.get(inp)
            if b is not None:
                encoder.setBuffer_offset_atIndex_(b, 0, in_i)
        elem_count, byte_size = _calculate_node_bytes(node)
        out_buf = self.device.newBufferWithLength_options_(byte_size, Metal.MTLResourceStorageModeShared)
        encoder.setBuffer_offset_atIndex_(out_buf, 0, len(node.inputs))
        device_buffers[node.id] = out_buf

        block_cfg, grid_cfg = query_optimal_launch_geometry(
            getattr(node, "shape_metadata", None) or (elem_count,),
        )
        threads_per_grid = Metal.MTLSize(grid_cfg[0] * block_cfg[0], grid_cfg[1] * block_cfg[1], grid_cfg[2] * block_cfg[2])
        threads_per_group = Metal.MTLSize(block_cfg[0], block_cfg[1], block_cfg[2])
        encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_group)

    def execute_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, ArrayBufferType],
    ) -> dict[str, bytes]:
        """Execute multi-node IRGraph through dynamic MSL compilation and Metal pipeline dispatch.

        Args:
            graph (IRGraph): Target computation graph.
            inputs (dict[str, ArrayBufferType]): Named input host buffers.

        Returns:
            dict[str, bytes]: Dictionary of output buffers as bytes.

        Raises:
            BackendNotSupportedError: When Metal runtime or device is unavailable.
        """
        if not self.is_available() or self.device is None:
            raise BackendNotSupportedError("Apple Silicon Metal GPU runtime is not available.")

        import Metal

        msl_source: str = MetalCodeGenerator(graph).generate_msl()
        options = Metal.MTLCompileOptions.new()
        library, err = self.device.newLibraryWithSource_options_error_(msl_source, options, None)
        if library is None:
            raise RuntimeError(f"Metal compilation failed: {err}")

        device_buffers = self._prepare_device_buffers(inputs)
        queue = self.device.newCommandQueue()
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        node_idx: int = 0
        for node in getattr(graph, "nodes", {}).values():
            if getattr(node, "op_type", "") == "Input":
                continue
            self._dispatch_compute_node(node, node_idx, library, encoder, device_buffers)
            node_idx += 1

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        results: dict[str, bytes] = {}
        for out_id in getattr(graph, "outputs", []):
            buf = device_buffers.get(out_id)
            if buf is not None and hasattr(buf, "contents"):
                node = graph.nodes.get(out_id)
                _, byte_size = _calculate_node_bytes(node)
                ptr = buf.contents()
                results[out_id] = bytes(ctypes.string_at(ptr, byte_size))

        return results
