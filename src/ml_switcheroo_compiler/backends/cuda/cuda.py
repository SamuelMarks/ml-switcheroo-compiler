"""CUDA backend generator."""

import ctypes
import ctypes.util
import importlib

try:
    cupy = importlib.import_module("cupy")
except (ImportError, AttributeError):
    cupy = None
import os
import re
from typing import Optional, Union

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.hardware_config_models import (
    HardwareTemplateConfig,
    calculate_hardware_launch_config,
    compute_liveness,
    load_hardware_templates,
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


@register_backend("cuda")
class CudaCodeGenerator(BaseGenerator):
    """CUDA C++ Code Generator."""

    def __init__(self, graph: IRGraph, delegates: Optional[list[CodeGeneratorVisitor]] = None) -> None:
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
        op_type = getattr(node, "op_type", "")
        out_name = f"d_out_{node_idx}"
        node_buffers[node.id] = out_name
        num_elements, bytes_alloc = _calculate_node_bytes(node)
        cuda.append(f"    float* {out_name};")
        cuda.append(f"    CUDA_CHECK(cudaMalloc(&{out_name}, {bytes_alloc}));")
        in_name = node_buffers.get(node.inputs[0], "inputs[0]") if getattr(node, "inputs", None) else out_name
        op_lower = op_type.lower()
        if op_lower in ("allreduce", "ncclallreduce"):
            cuda.append(f"    NCCL_CHECK(ncclAllReduce((const void*){in_name}, (void*){out_name}, {num_elements}, ncclFloat, ncclSum, comm, stream));")
        elif op_lower == "allgather":
            cuda.append(f"    NCCL_CHECK(ncclAllGather((const void*){in_name}, (void*){out_name}, {num_elements}, ncclFloat, comm, stream));")
        elif op_lower == "reducescatter":
            cuda.append(f"    NCCL_CHECK(ncclReduceScatter((const void*){in_name}, (void*){out_name}, {num_elements}, ncclFloat, ncclSum, comm, stream));")
        elif op_lower == "broadcast":
            root = getattr(node, "attributes", {}).get("root", 0)
            cuda.append(f"    NCCL_CHECK(ncclBroadcast((const void*){in_name}, (void*){out_name}, {num_elements}, ncclFloat, {root}, comm, stream));")
        for inp in getattr(node, "inputs", []):
            if last_consumer.get(inp) == node.id and inp not in outputs_set and inp not in inputs_set:
                buf_to_free = node_buffers.get(inp)
                if buf_to_free:
                    cuda.append(f"    CUDA_CHECK(cudaFree({buf_to_free}));")

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
        op_type = getattr(node, "op_type", "")
        if op_type.lower() in ("allreduce", "ncclallreduce", "allgather", "reducescatter", "broadcast"):
            self._emit_nccl_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, cuda)
            return

        if op_type.lower() == "fusedelementwise":
            out_name = f"d_out_{node_idx}"
            node_buffers[node.id] = out_name
            num_elements, bytes_alloc = _calculate_node_bytes(node)
            cuda.append(f"    float* {out_name};")
            cuda.append(f"    CUDA_CHECK(cudaMalloc(&{out_name}, {bytes_alloc}));")
            cuda.append(f"    dim3 block_{node_idx}(256, 1, 1);")
            (grid_x, grid_y, grid_z), launch_args = resolve_hardware_launch_grid_and_args(node, {}, node_buffers, out_name, num_elements, node_idx, self.graph)
            cuda.append(f"    dim3 grid_{node_idx}({grid_x}, {grid_y}, {grid_z});")
            kernel_fn = f"fused_elementwise_kernel_{node_idx}"
            cuda.append(f"    {kernel_fn}<<<grid_{node_idx}, block_{node_idx}>>>({', '.join(launch_args)});")
            cuda.append("    CUDA_CHECK(cudaGetLastError());")
            for inp in getattr(node, "inputs", []):
                if last_consumer.get(inp) == node.id and inp not in outputs_set and inp not in inputs_set:
                    buf_to_free = node_buffers.get(inp)
                    if buf_to_free:
                        cuda.append(f"    CUDA_CHECK(cudaFree({buf_to_free}));")
            return

        tpl = self.config.templates.get(op_type.lower())
        if not tpl:
            return
        out_name = f"d_out_{node_idx}"
        node_buffers[node.id] = out_name
        num_elements, bytes_alloc = _calculate_node_bytes(node)

        cuda.append(f"    float* {out_name};")
        cuda.append(f"    CUDA_CHECK(cudaMalloc(&{out_name}, {bytes_alloc}));")
        wg = tpl.get("workgroup_size", [1, 1, 1]) or [1, 1, 1]
        cuda.append(f"    dim3 block_{node_idx}({wg[0]}, {wg[1]}, {wg[2]});")

        (grid_x, grid_y, grid_z), launch_args = resolve_hardware_launch_grid_and_args(node, tpl, node_buffers, out_name, num_elements, node_idx, self.graph)
        cuda.append(f"    dim3 grid_{node_idx}({grid_x}, {grid_y}, {grid_z});")

        kernel_fn = _extract_kernel_name(tpl.get("body", ""), op_type.lower())
        cuda.append(f"    {kernel_fn}<<<grid_{node_idx}, block_{node_idx}>>>({', '.join(launch_args)});")
        cuda.append("    CUDA_CHECK(cudaGetLastError());")

        for inp in getattr(node, "inputs", []):
            if last_consumer.get(inp) == node.id and inp not in outputs_set and inp not in inputs_set:
                buf_to_free = node_buffers.get(inp)
                if buf_to_free:
                    cuda.append(f"    CUDA_CHECK(cudaFree({buf_to_free}));")

    def generate(self) -> str:
        """Generate CUDA C++ code with dynamic buffer allocations and memory lifetime tracking.

        Returns:
            str: Generated CUDA strings.
        """
        has_collectives: bool = any(getattr(n, "op_type", "").lower() in ("allreduce", "ncclallreduce", "allgather", "reducescatter", "broadcast") for n in getattr(self.graph, "nodes", {}).values())

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
                    '            std::cerr << "NCCL Error: " << ncclGetErrorString(r) << " at " << __FILE__ << ":" << __LINE__ << std::endl; ',
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
                '            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; ',
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
        )

        for node_idx, node in enumerate(getattr(self.graph, "nodes", {}).values()):
            op_type: str = getattr(node, "op_type", "")
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
            else:
                tpl = self.config.templates.get(op_type.lower())
                if tpl:
                    cuda.append(tpl.get("body", ""))

        eval_fn_sig: str = "void evaluate_cuda(const std::vector<float*>& inputs, std::vector<float*>& outputs, ncclComm_t comm = 0, cudaStream_t stream = 0) {" if has_collectives else "void evaluate_cuda(const std::vector<float*>& inputs, std::vector<float*>& outputs) {"
        cuda.append(eval_fn_sig)
        cuda.append("    // 1. Dynamic Memory Allocation and Kernel Dispatch")

        node_buffers: dict[str, str] = {inp_name: f"inputs[{i}]" for i, inp_name in enumerate(getattr(self.graph, "inputs", []))}
        last_consumer = compute_liveness(self.graph)
        outputs_set = set(getattr(self.graph, "outputs", []))
        inputs_set = set(getattr(self.graph, "inputs", []))

        node_idx: int = 0
        for node in getattr(self.graph, "nodes", {}).values():
            if getattr(node, "op_type", "") == "Input":
                continue
            self._emit_node(node, node_idx, node_buffers, last_consumer, outputs_set, inputs_set, cuda)
            node_idx += 1

        cuda.append("    CUDA_CHECK(cudaDeviceSynchronize());")
        cuda.append("}")
        return "\n".join(cuda)


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


class CUDARunner:
    """CUDA Runner utilizing cupy if available, fallback to ctypes driver API."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if CUDA driver and accelerator hardware are functional.

        Returns:
            bool: True if CUDA is functional, False otherwise.
        """
        if cupy is not None:
            try:
                return bool(cupy.cuda.is_available() and cupy.cuda.runtime.getDeviceCount() > 0)
            except Exception:
                pass
        try:
            cuda_lib_path = ctypes.util.find_library("cuda")
            if cuda_lib_path:
                lib = ctypes.cdll.LoadLibrary(cuda_lib_path)
                if hasattr(lib, "cuInit") and lib.cuInit(0) == 0:
                    count = ctypes.c_int()
                    if hasattr(lib, "cuDeviceGetCount") and lib.cuDeviceGetCount(ctypes.byref(count)) == 0:
                        return count.value > 0
        except Exception:
            pass
        return False

    def __init__(self) -> None:
        """Initialize CUDARunner and load hardware limits."""
        self.limits: dict[str, Union[int, list[int], str, None]] = {}
        limits_path: str = os.path.join(os.path.dirname(__file__), "hardware_limits.yaml")
        if os.path.exists(limits_path):
            with open(limits_path) as f:
                self.limits = yaml.safe_load(f) or {}

        self.cuda_lib: Optional[ctypes.CDLL] = None
        self._ctx: Optional[ctypes.c_void_p] = None

        if cupy is not None:
            self.mode = "cupy"
        else:
            self.mode = "ctypes"
            try:
                cuda_lib_path = ctypes.util.find_library("cuda")
                if cuda_lib_path:
                    self.cuda_lib = ctypes.cdll.LoadLibrary(cuda_lib_path)
                    self.cuda_lib.cuInit(0)
                    device = ctypes.c_int()
                    self.cuda_lib.cuDeviceGet(ctypes.byref(device), 0)
                    self._ctx = ctypes.c_void_p()
                    self.cuda_lib.cuCtxCreate_v2(ctypes.byref(self._ctx), 0, device)
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

        if not self.cuda_lib:
            return ctypes.c_void_p(0)

        dptr = ctypes.c_void_p()
        res = self.cuda_lib.cuMemAlloc_v2(ctypes.byref(dptr), size)
        if res != 0:
            raise RuntimeError(f"cuMemAlloc failed with error code {res}")
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

        if self.cuda_lib:
            res = self.cuda_lib.cuMemcpyHtoD_v2(device_ptr, host_data, len(host_data))
            if res != 0:
                raise RuntimeError(f"cuMemcpyHtoD failed with error code {res}")

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

        if self.cuda_lib:
            host_data = ctypes.create_string_buffer(size)
            res = self.cuda_lib.cuMemcpyDtoH_v2(host_data, device_ptr, size)
            if res != 0:
                raise RuntimeError(f"cuMemcpyDtoH failed with error code {res}")
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
        elif self.cuda_lib:
            self.cuda_lib.cuMemFree_v2(device_ptr)

    def compile_cuda_to_ptx(
        self,
        cuda_source: str,
        source_filename: str = "kernel.cu",
        options: Optional[list[str]] = None,
    ) -> str:
        """Compile CUDA source code to PTX string.

        Args:
            cuda_source (str): CUDA C++ source code.
            source_filename (str): Virtual filename.
            options (Optional[list[str]]): Compiler options.

        Returns:
            str: Compiled PTX code.
        """
        if self.mode == "cupy" and cupy is not None:
            try:
                opts = tuple(options) if options else ()
                if hasattr(cupy.cuda.nvrtc, "create_program"):
                    cupy.cuda.nvrtc.create_program(cuda_source, source_filename)
                ptx_bytes = cupy.cuda.nvrtc.get_ptx(cuda_source, options=opts)
                if isinstance(ptx_bytes, bytes):
                    return ptx_bytes.decode("utf-8")
                return str(ptx_bytes)
            except Exception:
                pass

        try:
            nvrtc_path = ctypes.util.find_library("nvrtc") or "libnvrtc.so"
            nvrtc_lib = ctypes.cdll.LoadLibrary(nvrtc_path)
        except Exception as err:
            raise RuntimeError("NVRTC library unavailable") from err

        prog = ctypes.c_void_p()
        res = nvrtc_lib.nvrtcCreateProgram(
            ctypes.byref(prog),
            cuda_source.encode("utf-8"),
            source_filename.encode("utf-8"),
            0,
            None,
            None,
        )
        if res != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed with error code {res}")

        opt_list = options or []
        c_opts = (ctypes.c_char_p * len(opt_list))(*[opt.encode("utf-8") for opt in opt_list])
        res = nvrtc_lib.nvrtcCompileProgram(prog, len(opt_list), c_opts)
        if res != 0:
            log_size = ctypes.c_size_t()
            nvrtc_lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            nvrtc_lib.nvrtcGetProgramLog(prog, log_buf)
            log_msg = log_buf.value.decode("utf-8", errors="replace")
            if hasattr(nvrtc_lib, "nvrtcDestroyProgram"):
                nvrtc_lib.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"NVRTC compilation failed: {log_msg}")

        ptx_size = ctypes.c_size_t()
        nvrtc_lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        nvrtc_lib.nvrtcGetPTX(prog, ptx_buf)
        if hasattr(nvrtc_lib, "nvrtcDestroyProgram"):
            nvrtc_lib.nvrtcDestroyProgram(ctypes.byref(prog))
        return ptx_buf.value.decode("utf-8", errors="replace")

    def compile_and_dispatch(
        self,
        ptx_source: str,
        entry_point: str,
        grid_size: list[int],
        block_size: list[int],
        args: Optional[list[KernelArgType]] = None,
    ) -> None:
        """Compile PTX modules and launch kernels.

        Args:
            ptx_source (str): PTX source code.
            entry_point (str): Kernel entry point.
            grid_size (list[int]): Grid sizing.
            block_size (list[int]): Block sizing.
            args (Optional[list[KernelArgType]]): Kernel arguments to marshall to device.
        """
        if self.mode == "cupy" and cupy is not None:
            module = cupy.RawModule(code=ptx_source)
            kernel = module.get_function(entry_point)
            kernel(tuple(grid_size), tuple(block_size), tuple(args) if args else ())
            return

        if self.cuda_lib:
            module = ctypes.c_void_p()
            ptx_bytes = ptx_source.encode("utf-8")
            res = self.cuda_lib.cuModuleLoadData(ctypes.byref(module), ptx_bytes)
            if res != 0:
                raise RuntimeError(f"cuModuleLoadData failed with error code {res}")

            kernel = ctypes.c_void_p()
            res = self.cuda_lib.cuModuleGetFunction(ctypes.byref(kernel), module, entry_point.encode("utf-8"))
            if res != 0:
                raise RuntimeError(f"cuModuleGetFunction failed with error code {res}")

            kernel_params = _marshall_kernel_params(args)

            res = self.cuda_lib.cuLaunchKernel(
                kernel,
                grid_size[0],
                grid_size[1] if len(grid_size) > 1 else 1,
                grid_size[2] if len(grid_size) > 2 else 1,
                block_size[0],
                block_size[1] if len(block_size) > 1 else 1,
                block_size[2] if len(block_size) > 2 else 1,
                0,
                0,
                kernel_params,
                0,
            )
            if res != 0:
                raise RuntimeError(f"cuLaunchKernel failed with error code {res}")

    def synchronize(self) -> None:
        """Synchronize the active CUDA stream or device barrier.

        Raises:
            RuntimeError: If device synchronization fails.
        """
        if self.mode == "cupy" and cupy is not None:
            cupy.cuda.Stream.null.synchronize()
            return

        if not self.cuda_lib:
            return

        if hasattr(self.cuda_lib, "cuCtxSynchronize"):
            res = self.cuda_lib.cuCtxSynchronize()
            if res != 0:
                raise RuntimeError(f"cuCtxSynchronize failed with error code {res}")
        elif hasattr(self.cuda_lib, "cudaDeviceSynchronize"):
            res = self.cuda_lib.cudaDeviceSynchronize()
            if res != 0:
                raise RuntimeError(f"cudaDeviceSynchronize failed with error code {res}")

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
        generator: CudaCodeGenerator,
        ptx_source: str,
        device_buffers: dict[str, ctypes.c_void_p],
    ) -> None:
        """Dispatch compute kernel execution for a single IRNode.

        Args:
            node (IRNode): The IR node to execute.
            node_idx (int): Current node execution index.
            generator (CudaCodeGenerator): CUDA code generator holding op templates.
            ptx_source (str): Compiled PTX source string.
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

        wg: list[int] = tpl.get("workgroup_size", [256, 1, 1]) or [256, 1, 1]
        block_size = [wg[0], wg[1] if len(wg) > 1 else 1, wg[2] if len(wg) > 2 else 1]
        _, grid_cfg = calculate_hardware_launch_config(
            getattr(node, "shape_metadata", None) or (num_elements,),
            max_threads_per_block=block_size[0],
        )
        grid_size = [grid_cfg[0], grid_cfg[1], grid_cfg[2]]

        kernel_args: list[KernelArgType] = [device_buffers[inp] for inp in getattr(node, "inputs", []) if inp in device_buffers]
        if out_ptr is not None:
            kernel_args.append(out_ptr)
        kernel_args.append(num_elements)

        if ptx_source and out_ptr is not None:
            self.compile_and_dispatch(ptx_source, kernel_fn, grid_size, block_size, kernel_args)

    def execute_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[bytes, memoryview, list[float]]],
    ) -> dict[str, bytes]:
        """Execute multi-node IRGraph through dynamic JIT compilation and kernel dispatch.

        Args:
            graph (IRGraph): Target computation graph.
            inputs (dict[str, Union[bytes, memoryview, list[float]]]): Host input buffers.

        Returns:
            dict[str, bytes]: Result dictionary of output buffers.

        Raises:
            BackendNotSupportedError: When CUDA hardware or driver is unavailable.
        """
        if not self.is_available() and not self.cuda_lib and (self.mode != "cupy" or cupy is None):
            raise BackendNotSupportedError("CUDA accelerator hardware or driver is not available.")

        cuda_source = CudaCodeGenerator(graph).generate()
        ptx_source = self.compile_cuda_to_ptx(cuda_source)
        device_buffers = self._prepare_device_buffers(inputs)
        generator = CudaCodeGenerator(graph)

        node_idx: int = 0
        for node in getattr(graph, "nodes", {}).values():
            if getattr(node, "op_type", "") == "Input":
                continue
            self._dispatch_compute_node(node, node_idx, generator, ptx_source, device_buffers)
            node_idx += 1

        self.synchronize()

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
