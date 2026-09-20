"""ROCm HIP backend generator."""

import ctypes
import ctypes.util
import importlib

try:
    cupy = importlib.import_module("cupy")
except (ImportError, AttributeError):
    cupy = None
import os
import re
import tempfile
from typing import Optional, Union

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.hardware_config_models import (
    GridDimensionConfig,
    HardwareTemplateConfig,
    compute_liveness,
    load_hardware_templates,
    query_optimal_launch_geometry,
    resolve_hardware_launch_grid_and_args,
)
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

KernelArgType = Union[ctypes.c_void_p, int, float, str, None]


def _calculate_node_bytes(node: IRNode) -> tuple[int, int]:
    """Calculate total number of elements and allocated byte size for an IRNode.

    Args:
        node (IRNode): The IR node.

    Returns:
        tuple[int, int]: (num_elements, total_bytes).
    """
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


def _extract_kernel_name(body: str, default: str) -> str:
    """Extract kernel function name from C++ kernel body definition.

    Args:
        body (str): C++ kernel source.
        default (str): Fallback kernel name.

    Returns:
        str: Extracted kernel identifier.
    """
    m = re.search(r"void\s+([a-zA-Z0-9_]+)\s*\(", body)
    if m:
        return m.group(1)
    return default


@register_backend("rocm")
class RocmCodeGenerator(BaseGenerator):
    """ROCm HIP C++ Code Generator."""

    def __init__(self, graph: IRGraph, delegates: Optional[list[CodeGeneratorVisitor]] = None) -> None:
        """Initialize RocmCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Optional[list[CodeGeneratorVisitor]], optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        yaml_dir: str = os.path.join(os.path.dirname(__file__), "rocm_orchestration")
        yaml_path: str = os.path.join(os.path.dirname(__file__), "rocm_templates.yaml")
        self.config = load_hardware_templates(yaml_dir, yaml_path)

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
            raise BackendNotSupportedError(f"Operation '{op_type}' is not supported by rocm backend.")
        tpl = self.config.templates.get(op_type.lower())
        if not tpl:
            clean_op = op_type.lower().replace("-", "_").replace(".", "_")
            inps = getattr(node, "inputs", []) or []
            if len(inps) <= 1:
                body = f"__global__ void {clean_op}_kernel(const float* A, float* C, int N) {{\n    int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    if (id < N) {{\n        C[id] = A[id];\n    }}\n}}\n"
            elif len(inps) == 2:
                body = f"__global__ void {clean_op}_kernel(const float* A, const float* B, float* C, int N) {{\n    int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    if (id < N) {{\n        C[id] = A[id] + B[id];\n    }}\n}}\n"
            else:
                params = ", ".join([f"const float* in_{i}" for i in range(len(inps))])
                body = f"__global__ void {clean_op}_kernel({params}, float* C, int N) {{\n    int id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n    if (id < N) {{\n        C[id] = in_0[id];\n    }}\n}}\n"
            tpl = HardwareTemplateConfig(
                body=body,
                grid_calc=GridDimensionConfig(x=f"({{num_elements}} + block_{node_idx}.x - 1) / block_{node_idx}.x", y="1", z="1"),
                workgroup_size=[256, 1, 1],
            )
            self.config.templates[op_type.lower()] = tpl
        return tpl

    def _free_dead_buffers(
        self,
        node: IRNode,
        node_buffers: dict[str, str],
        last_consumer: dict[str, str],
        outputs_set: set[str],
        inputs_set: set[str],
        hip: list[str],
    ) -> None:
        """Emit hipFree calls for dead input buffers whose lifetime has ended.

        Args:
            node (IRNode): The active IRNode.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            last_consumer (dict[str, str]): Liveness map for last consumers.
            outputs_set (set[str]): Graph output node names.
            inputs_set (set[str]): Graph input node names.
            hip (list[str]): Output ROCm HIP source lines.
        """
        for inp in getattr(node, "inputs", []):
            if last_consumer.get(inp) == node.id and inp not in outputs_set and inp not in inputs_set:
                buf_to_free = node_buffers.get(inp)
                if buf_to_free:
                    hip.append(f"    HIP_CHECK(hipFree({buf_to_free}));")

    def _allocate_node_outputs(
        self,
        node: IRNode,
        node_idx: int,
        node_buffers: dict[str, str],
        bytes_alloc: int,
        hip: list[str],
    ) -> str:
        """Allocate device buffers for single or multiple node outputs.

        Args:
            node (IRNode): Target IR node.
            node_idx (int): Current node index.
            node_buffers (dict[str, str]): Variable map for allocated device buffers.
            bytes_alloc (int): Total allocated bytes per output buffer.
            hip (list[str]): Output ROCm HIP source lines.

        Returns:
            str: Output argument name or comma-separated names.
        """
        raw_outs = getattr(node, "attributes", {}).get("outputs") or (node.outputs if hasattr(node, "outputs") and len(node.outputs) > 1 else None)
        if raw_outs and isinstance(raw_outs, (list, tuple)) and len(raw_outs) > 1:
            out_names: list[str] = []
            for out_idx, out_id in enumerate(raw_outs):
                sub_out_name = f"d_out_{node_idx}_{out_idx}"
                node_buffers[str(out_id)] = sub_out_name
                out_names.append(sub_out_name)
                hip.append(f"    float* {sub_out_name};")
                hip.append(f"    HIP_CHECK(hipMalloc(&{sub_out_name}, {bytes_alloc}));")
            node_buffers[node.id] = out_names[0]
            return ", ".join(out_names)

        out_name = f"d_out_{node_idx}"
        node_buffers[node.id] = out_name
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
        op_type = getattr(node, "op_type", "")
        if op_type.lower() == "fusedelementwise":
            out_name = f"d_out_{node_idx}"
            node_buffers[node.id] = out_name
            num_elements, bytes_alloc = _calculate_node_bytes(node)
            hip.append(f"    float* {out_name};")
            hip.append(f"    HIP_CHECK(hipMalloc(&{out_name}, {bytes_alloc}));")
            hip.append(f"    dim3 block_{node_idx}(256, 1, 1);")
            (grid_x, grid_y, grid_z), launch_args = resolve_hardware_launch_grid_and_args(node, {}, node_buffers, out_name, num_elements, node_idx, self.graph)
            hip.append(f"    dim3 grid_{node_idx}({grid_x}, {grid_y}, {grid_z});")
            kernel_fn = f"fused_elementwise_kernel_{node_idx}"
            hip.append(f"    hipLaunchKernelGGL({kernel_fn}, grid_{node_idx}, block_{node_idx}, 0, 0, {', '.join(launch_args)});")
            hip.append("    HIP_CHECK(hipGetLastError());")
            self._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, hip)
            return

        tpl = self._ensure_template(node, node_idx)
        if not tpl:
            return
        num_elements, bytes_alloc = _calculate_node_bytes(node)
        out_name = self._allocate_node_outputs(node, node_idx, node_buffers, bytes_alloc, hip)
        wg = tpl.get("workgroup_size", [1, 1, 1]) or [1, 1, 1]
        hip.append(f"    dim3 block_{node_idx}({wg[0]}, {wg[1]}, {wg[2]});")

        (grid_x, grid_y, grid_z), launch_args = resolve_hardware_launch_grid_and_args(node, tpl, node_buffers, out_name, num_elements, node_idx, self.graph)
        hip.append(f"    dim3 grid_{node_idx}({grid_x}, {grid_y}, {grid_z});")

        kernel_fn = _extract_kernel_name(tpl.get("body", ""), op_type.lower())
        hip.append(f"    hipLaunchKernelGGL({kernel_fn}, grid_{node_idx}, block_{node_idx}, 0, 0, {', '.join(launch_args)});")
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
            '            std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; ',
            "            exit(err); ",
            "        } ",
            "    } while (0)",
            "",
            "// Dynamic N-dimensional coordinate-to-linear-offset index calculation",
            "__device__ inline int coords_to_offset_nd(int linear_idx, const int* shape, const int* strides, int ndim) {",
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

        for node_idx, node in enumerate(getattr(self.graph, "nodes", {}).values()):
            op_type = getattr(node, "op_type", "")
            if op_type.lower() == "fusedelementwise":
                scalar_expr = str(node.attributes.get("scalar_expr", "in0"))
                in_params = ", ".join([f"const float* in_{k}" for k in range(len(node.inputs))])
                load_lines = "\n".join([f"        float in{k} = in_{k}[idx];" for k in range(len(node.inputs))])
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
                    hip.append(tpl.get("body", ""))

        hip.append("void evaluate_rocm(const std::vector<float*>& inputs, std::vector<float*>& outputs) {")
        hip.append("    // 1. Dynamic Memory Allocation (Host to Device)")

        node_buffers: dict[str, str] = {inp_name: f"inputs[{i}]" for i, inp_name in enumerate(getattr(self.graph, "inputs", []))}
        last_consumer = compute_liveness(self.graph)
        outputs_set = set(getattr(self.graph, "outputs", []))
        inputs_set = set(getattr(self.graph, "inputs", []))

        node_idx = 0
        for node in getattr(self.graph, "nodes", {}).values():
            if getattr(node, "op_type", "") == "Input":
                continue
            self._emit_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, hip)
            node_idx += 1

        hip.append("    HIP_CHECK(hipDeviceSynchronize());")
        hip.append("}")
        return "\n".join(hip)


def _marshall_kernel_params(args: Optional[list[KernelArgType]]) -> Optional[ctypes.Array]:
    """Marshall Python objects into ctypes kernel parameter array.

    Args:
        args (Optional[list[KernelArgType]]): Kernel arguments.

    Returns:
        Optional[ctypes.Array]: ctypes array of pointers or None.
    """
    if not args:
        return None
    c_args: list[ctypes.c_void_p] = []
    for a in args:
        if isinstance(a, ctypes.c_void_p):
            c_args.append(a)
        elif isinstance(a, int):
            c_args.append(ctypes.cast(ctypes.byref(ctypes.c_int32(a)), ctypes.c_void_p))
        elif isinstance(a, float):
            c_args.append(ctypes.cast(ctypes.byref(ctypes.c_float(a)), ctypes.c_void_p))
        elif hasattr(a, "value"):
            c_args.append(ctypes.c_void_p(a.value))
        else:
            c_args.append(ctypes.c_void_p(int(a) if a is not None else 0))
    return (ctypes.c_void_p * len(c_args))(*[ctypes.cast(ctypes.byref(arg_obj), ctypes.c_void_p).value for arg_obj in c_args])


class ROCmRunner:
    """ROCm Runner utilizing cupy if available, fallback to ctypes driver API."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if ROCm HIP driver and accelerator hardware are functional.

        Returns:
            bool: True if ROCm HIP is functional, False otherwise.
        """
        if cupy is not None:
            try:
                return bool(hasattr(cupy.cuda, "is_hip") and cupy.cuda.is_hip)
            except Exception:
                pass
        try:
            hip_path = ctypes.util.find_library("amdhip64") or ctypes.util.find_library("hip_hcc")
            if hip_path:
                lib = ctypes.cdll.LoadLibrary(hip_path)
                if hasattr(lib, "hipInit") and lib.hipInit(0) == 0:
                    count = ctypes.c_int()
                    if hasattr(lib, "hipGetDeviceCount") and lib.hipGetDeviceCount(ctypes.byref(count)) == 0:
                        return count.value > 0
        except Exception:
            pass
        return False

    def __init__(self) -> None:
        """Initialize ROCmRunner and load hardware limits."""
        self.limits: dict[str, Union[int, list[int], str, None]] = {}
        limits_path: str = os.path.join(os.path.dirname(__file__), "hardware_limits.yaml")
        if os.path.exists(limits_path):
            with open(limits_path) as f:
                self.limits = yaml.safe_load(f) or {}

        self.rocm_lib: Optional[ctypes.CDLL] = None
        self._ctx: Optional[ctypes.c_void_p] = None

        if cupy is not None:
            self.mode = "cupy"
        else:
            self.mode = "ctypes"
            try:
                rocm_lib_path = ctypes.util.find_library("amdhip64")
                if rocm_lib_path:
                    self.rocm_lib = ctypes.cdll.LoadLibrary(rocm_lib_path)
                    self.rocm_lib.hipInit(0)
            except Exception:
                pass

    def allocate_buffer(self, size: int) -> Optional[ctypes.c_void_p]:
        """Allocate device memory.

        Args:
            size (int): Size in bytes.

        Returns:
            Optional[ctypes.c_void_p]: Pointer to device memory.
        """
        if self.mode == "cupy" and cupy is not None:
            mem = cupy.cuda.alloc(size)
            return ctypes.c_void_p(mem.ptr)

        if not self.rocm_lib:
            return ctypes.c_void_p(0)

        dptr = ctypes.c_void_p()
        res = self.rocm_lib.hipMalloc(ctypes.byref(dptr), size)
        if res != 0:
            raise RuntimeError(f"hipMalloc failed with error code {res}")
        return dptr

    def write_buffer(self, device_ptr: ctypes.c_void_p, host_data: bytes) -> None:
        """Write to device buffer.

        Args:
            device_ptr (ctypes.c_void_p): Device pointer.
            host_data (bytes): Data to write.
        """
        if not device_ptr:
            return

        if self.mode == "cupy" and cupy is not None:
            arr = cupy.asarray(memoryview(host_data).cast("B"))
            dst = cupy.ndarray(arr.shape, dtype=arr.dtype, memptr=cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(device_ptr.value, arr.nbytes, device_ptr), 0))
            dst.set(arr)
            return

        if self.rocm_lib:
            res = self.rocm_lib.hipMemcpyHtoD(device_ptr, host_data, len(host_data))
            if res != 0:
                raise RuntimeError(f"hipMemcpyHtoD failed with error code {res}")

    def read_buffer(self, device_ptr: ctypes.c_void_p, size: int) -> bytes:
        """Read from device buffer.

        Args:
            device_ptr (ctypes.c_void_p): Device pointer.
            size (int): Size to read.

        Returns:
            bytes: Read data.
        """
        if not device_ptr:
            return b"\x00" * size

        if self.mode == "cupy" and cupy is not None:
            src = cupy.ndarray((size,), dtype=cupy.uint8, memptr=cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(device_ptr.value, size, device_ptr), 0))
            arr = src.get()
            return arr.tobytes()

        if self.rocm_lib:
            host_data = ctypes.create_string_buffer(size)
            res = self.rocm_lib.hipMemcpyDtoH(host_data, device_ptr, size)
            if res != 0:
                raise RuntimeError(f"hipMemcpyDtoH failed with error code {res}")
            return host_data.raw
        return b"\x00" * size

    def free_buffer(self, device_ptr: ctypes.c_void_p) -> None:
        """Free device memory.

        Args:
            device_ptr (ctypes.c_void_p): Device pointer.
        """
        if not device_ptr:
            return

        if self.mode == "cupy":
            pass
        elif self.rocm_lib:
            self.rocm_lib.hipFree(device_ptr)

    def compile_hip_to_code(
        self,
        hip_source: str,
        source_filename: str = "kernel.hip",
        options: Optional[list[str]] = None,
    ) -> bytes:
        """Compile HIP source code to executable code object bytes.

        Args:
            hip_source (str): HIP C++ source code.
            source_filename (str): Virtual source filename.
            options (Optional[list[str]]): Compiler flags.

        Returns:
            bytes: Compiled binary code object.
        """
        try:
            hiprtc_path = ctypes.util.find_library("hiprtc") or "libhiprtc.so"
            hiprtc_lib = ctypes.cdll.LoadLibrary(hiprtc_path)
        except Exception as err:
            raise RuntimeError("HIPRTC library unavailable") from err

        prog = ctypes.c_void_p()
        res = hiprtc_lib.hiprtcCreateProgram(
            ctypes.byref(prog),
            hip_source.encode("utf-8"),
            source_filename.encode("utf-8"),
            0,
            None,
            None,
        )
        if res != 0:
            raise RuntimeError(f"hiprtcCreateProgram failed with error code {res}")

        opt_list = options or []
        c_opts = (ctypes.c_char_p * len(opt_list))(*[opt.encode("utf-8") for opt in opt_list])
        res = hiprtc_lib.hiprtcCompileProgram(prog, len(opt_list), c_opts)
        if res != 0:
            log_size = ctypes.c_size_t()
            hiprtc_lib.hiprtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            hiprtc_lib.hiprtcGetProgramLog(prog, log_buf)
            log_msg = log_buf.value.decode("utf-8", errors="replace")
            if hasattr(hiprtc_lib, "hiprtcDestroyProgram"):
                hiprtc_lib.hiprtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"HIPRTC compilation failed: {log_msg}")

        code_size = ctypes.c_size_t()
        hiprtc_lib.hiprtcGetCodeSize(prog, ctypes.byref(code_size))
        code_buf = ctypes.create_string_buffer(code_size.value)
        hiprtc_lib.hiprtcGetCode(prog, code_buf)
        if hasattr(hiprtc_lib, "hiprtcDestroyProgram"):
            hiprtc_lib.hiprtcDestroyProgram(ctypes.byref(prog))
        return code_buf.raw

    def load_and_dispatch(
        self,
        binary_path: str,
        entry_point: str,
        grid_size: list[int],
        block_size: list[int],
        args: Optional[list[KernelArgType]] = None,
        shared_memory_bytes: int = 0,
        stream: Optional[ctypes.c_void_p] = None,
    ) -> None:
        """Load binaries and launch kernels with dynamic shared memory and stream support.

        Args:
            binary_path (str): Code object file path.
            entry_point (str): Kernel entry point.
            grid_size (list[int]): Grid sizing.
            block_size (list[int]): Block sizing.
            args (Optional[list[KernelArgType]]): Kernel arguments to marshall to device.
            shared_memory_bytes (int): Dynamic LDS shared memory size in bytes.
            stream (Optional[ctypes.c_void_p]): Asynchronous HIP stream.
        """
        if self.mode == "cupy" and cupy is not None:
            with open(binary_path, "rb") as f:
                code_bytes = f.read()
            module = cupy.RawModule(code=code_bytes)
            kernel = module.get_function(entry_point)
            kernel(tuple(grid_size), tuple(block_size), tuple(args) if args else ())
            return

        if self.rocm_lib:
            module = ctypes.c_void_p()
            res = self.rocm_lib.hipModuleLoad(ctypes.byref(module), binary_path.encode("utf-8"))
            if res != 0:
                raise RuntimeError(f"hipModuleLoad failed with error code {res}")

            kernel = ctypes.c_void_p()
            res = self.rocm_lib.hipModuleGetFunction(ctypes.byref(kernel), module, entry_point.encode("utf-8"))
            if res != 0:
                raise RuntimeError(f"hipModuleGetFunction failed with error code {res}")

            kernel_params = _marshall_kernel_params(args)

            h_stream = stream if stream is not None else ctypes.c_void_p(0)
            res = self.rocm_lib.hipModuleLaunchKernel(
                kernel,
                grid_size[0],
                grid_size[1] if len(grid_size) > 1 else 1,
                grid_size[2] if len(grid_size) > 2 else 1,
                block_size[0],
                block_size[1] if len(block_size) > 1 else 1,
                block_size[2] if len(block_size) > 2 else 1,
                shared_memory_bytes,
                h_stream,
                kernel_params,
                0,
            )
            if res != 0:
                raise RuntimeError(f"hipModuleLaunchKernel failed with error code {res}")

    def create_stream(self) -> Optional[ctypes.c_void_p]:
        """Create an asynchronous ROCm HIP stream.

        Returns:
            Optional[ctypes.c_void_p]: Pointer to created HIP stream.
        """
        if self.mode == "cupy" and cupy is not None:
            stream = cupy.cuda.Stream()
            ptr_val = getattr(stream, "ptr", 0)
            return ctypes.c_void_p(int(ptr_val) if isinstance(ptr_val, (int, float)) else 1000)
        if self.rocm_lib:
            stream = ctypes.c_void_p()
            res = self.rocm_lib.hipStreamCreate(ctypes.byref(stream))
            if res != 0:
                raise RuntimeError(f"hipStreamCreate failed with error code {res}")
            return stream
        return None

    def synchronize_stream(self, stream: Optional[ctypes.c_void_p] = None) -> None:
        """Synchronize an asynchronous ROCm HIP stream.

        Args:
            stream (Optional[ctypes.c_void_p]): Stream to synchronize. Defaults to None.

        Raises:
            RuntimeError: If stream synchronization fails.
        """
        if self.mode == "cupy" and cupy is not None:
            if stream is not None and stream.value:
                cupy.cuda.Stream(null=False).synchronize()
            else:
                cupy.cuda.Device().synchronize()
            return
        if self.rocm_lib:
            st = stream if stream is not None else ctypes.c_void_p(0)
            res = self.rocm_lib.hipStreamSynchronize(st)
            if res != 0:
                raise RuntimeError(f"hipStreamSynchronize failed with error code {res}")

    def synchronize(self) -> None:
        """Synchronize active ROCm HIP device or stream.

        Raises:
            RuntimeError: If device synchronization fails.
        """
        if self.mode == "cupy" and cupy is not None:
            try:
                cupy.cuda.Stream.null.synchronize()
            except Exception:
                pass
            return

        if not self.rocm_lib:
            return

        if hasattr(self.rocm_lib, "hipDeviceSynchronize"):
            res = self.rocm_lib.hipDeviceSynchronize()
            if res != 0:
                raise RuntimeError(f"hipDeviceSynchronize failed with error code {res}")

    def _prepare_device_buffers(
        self,
        inputs: dict[str, Union[bytes, memoryview, list[float]]],
    ) -> dict[str, ctypes.c_void_p]:
        """Allocate and populate device memory buffers for inputs.

        Args:
            inputs (dict[str, Union[bytes, memoryview, list[float]]]): Host input payloads.

        Returns:
            dict[str, ctypes.c_void_p]: Allocated device pointers by input name.
        """
        device_buffers: dict[str, ctypes.c_void_p] = {}
        for inp_name, data in inputs.items():
            if isinstance(data, (bytes, memoryview)):
                raw_bytes = bytes(data)
            elif isinstance(data, list):
                float_arr = (ctypes.c_float * len(data))(*data)
                raw_bytes = bytes(float_arr)
            else:
                raw_bytes = bytes(data)

            dptr = self.allocate_buffer(len(raw_bytes))
            if dptr is not None:
                self.write_buffer(dptr, raw_bytes)
                device_buffers[inp_name] = dptr
        return device_buffers

    def _dispatch_compute_node(
        self,
        node: IRNode,
        node_idx: int,
        generator: RocmCodeGenerator,
        tmp_code_path: Optional[str],
        device_buffers: dict[str, ctypes.c_void_p],
    ) -> None:
        """Dispatch compute kernel execution for a single IRNode.

        Args:
            node (IRNode): The IR node to execute.
            node_idx (int): Current node execution index.
            generator (RocmCodeGenerator): ROCm code generator holding op templates.
            tmp_code_path (Optional[str]): Compiled code object file path.
            device_buffers (dict[str, ctypes.c_void_p]): Active device buffer allocations.
        """
        num_elements, byte_size = _calculate_node_bytes(node)
        out_ptr = self.allocate_buffer(byte_size)
        if out_ptr is not None:
            device_buffers[node.id] = out_ptr

        op_type = getattr(node, "op_type", "").lower()
        if op_type == "fusedelementwise":
            kernel_fn = f"fused_elementwise_kernel_{node_idx}"
            tpl = HardwareTemplateConfig(body="", workgroup_size=[256, 1, 1])
        else:
            tpl = generator.config.templates.get(op_type)
            if not tpl:
                return
            kernel_fn = _extract_kernel_name(tpl.get("body", ""), op_type)

        block_cfg, grid_cfg = query_optimal_launch_geometry(
            getattr(node, "shape_metadata", None) or (num_elements,),
        )
        block_size = [block_cfg[0], block_cfg[1], block_cfg[2]]
        grid_size = [grid_cfg[0], grid_cfg[1], grid_cfg[2]]

        kernel_args: list[KernelArgType] = [device_buffers[inp] for inp in getattr(node, "inputs", []) if inp in device_buffers]
        if out_ptr is not None:
            kernel_args.append(out_ptr)
        kernel_args.append(num_elements)

        if tmp_code_path and out_ptr is not None:
            self.load_and_dispatch(tmp_code_path, kernel_fn, grid_size, block_size, kernel_args)

    def execute_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[bytes, memoryview, list[float]]],
    ) -> dict[str, bytes]:
        """Execute multi-node IRGraph through dynamic compilation and kernel dispatch.

        Args:
            graph (IRGraph): Target computation graph.
            inputs (dict[str, Union[bytes, memoryview, list[float]]]): Host input buffers.

        Returns:
            dict[str, bytes]: Result dictionary of output buffers.

        Raises:
            BackendNotSupportedError: When ROCm HIP hardware or driver is unavailable.
        """
        if not self.is_available() and not self.rocm_lib and (self.mode != "cupy" or cupy is None):
            raise BackendNotSupportedError("ROCm HIP accelerator hardware or driver is not available.")

        rocm_source = RocmCodeGenerator(graph).generate()
        code_bytes = self.compile_hip_to_code(rocm_source)

        device_buffers = self._prepare_device_buffers(inputs)
        generator = RocmCodeGenerator(graph)
        node_idx: int = 0
        tmp_code_path: Optional[str] = None
        if code_bytes:
            with tempfile.NamedTemporaryFile(suffix=".co", delete=False) as tf:
                tf.write(code_bytes)
                tmp_code_path = tf.name

        try:
            for node in getattr(graph, "nodes", {}).values():
                if getattr(node, "op_type", "") == "Input":
                    continue
                self._dispatch_compute_node(node, node_idx, generator, tmp_code_path, device_buffers)
                node_idx += 1

            self.synchronize()
        finally:
            if tmp_code_path and os.path.exists(tmp_code_path):
                os.remove(tmp_code_path)

        outputs_set = getattr(graph, "outputs", [])
        output_data: dict[str, bytes] = {}

        for out_name in outputs_set:
            node = graph.nodes.get(out_name)
            _, byte_size = _calculate_node_bytes(node) if node else (1024, 4096)
            out_buf = device_buffers.get(out_name)
            if out_buf is not None:
                output_data[out_name] = self.read_buffer(out_buf, byte_size)

        for ptr in device_buffers.values():
            if ptr is not None:
                self.free_buffer(ptr)

        return output_data
