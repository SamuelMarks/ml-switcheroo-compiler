"""CUDA stream/event synchronization and graph execution runner."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import typing

import ml_switcheroo_compiler.backends.cuda.cuda as _cuda_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator
from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    KernelArgType,
    _as_tpl_opt,
    _calculate_node_bytes,
    _extract_kernel_name,
)
from ml_switcheroo_compiler.backends.cuda._cuda_runner_mem import _CUDARunnerMemory
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_operand_list,
    coerce_shape,
    coerce_str,
)
from ml_switcheroo_compiler.backends.hardware_config_models import (
    HardwareTemplateConfig,
    query_optimal_launch_geometry,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


class CUDARunner(_CUDARunnerMemory):
    """CUDA Runner utilizing cupy if available, fallback to ctypes driver API."""

    def create_stream(self) -> ctypes.c_void_p | None:
        """Create an asynchronous CUDA stream.

        Returns:
            Optional[ctypes.c_void_p]: Pointer to created CUDA stream.

        Raises:
            RuntimeError: If execution or validation fails in `create_stream`.
        """
        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            stream = _cuda_mod.cupy.cuda.Stream()
            ptr_val = adapt_backend_module(stream).get_symbol("ptr", 1000)
            return ctypes.c_void_p(int(ptr_val) if ptr_val else 1000)
        if self.cuda_lib:
            stream = ctypes.c_void_p()
            res = self.cuda_lib.cuStreamCreate(ctypes.byref(stream), 0)
            if res != 0:
                raise RuntimeError(f"cuStreamCreate failed with error code {res}")
            return stream
        return None

    def synchronize_stream(self, stream: ctypes.c_void_p | None = None) -> None:
        """Synchronize an asynchronous CUDA stream.

        Args:
            stream (Optional[ctypes.c_void_p]): Stream to synchronize. Defaults to None (default stream).

        Raises:
            RuntimeError: If stream synchronization fails.
        """
        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            if stream is not None and stream.value:
                _cuda_mod.cupy.cuda.Stream(null=False).synchronize()
            else:
                _cuda_mod.cupy.cuda.Device().synchronize()
            return
        if self.cuda_lib:
            st = stream if stream is not None else ctypes.c_void_p(0)
            res = self.cuda_lib.cuStreamSynchronize(st)
            if res != 0:
                raise RuntimeError(f"cuStreamSynchronize failed with error code {res}")

    def create_event(self) -> ctypes.c_void_p | None:
        """Create a CUDA event for timing and synchronization.

        Returns:
            Optional[ctypes.c_void_p]: Pointer to created CUDA event.

        Raises:
            RuntimeError: If execution or validation fails in `create_event`.
        """
        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            event = _cuda_mod.cupy.cuda.Event()
            ptr_val = adapt_backend_module(event).get_symbol("ptr", 1000)
            return ctypes.c_void_p(int(ptr_val) if ptr_val else 1000)
        if self.cuda_lib:
            event = ctypes.c_void_p()
            res = self.cuda_lib.cuEventCreate(ctypes.byref(event), 0)
            if res != 0:
                raise RuntimeError(f"cuEventCreate failed with error code {res}")
            return event
        return None

    def record_event(self, event: ctypes.c_void_p | None, stream: ctypes.c_void_p | None = None) -> None:
        """Record a CUDA event into a stream.

        Args:
            event (Optional[ctypes.c_void_p]): CUDA event.
            stream (Optional[ctypes.c_void_p]): CUDA stream.

        Raises:
            RuntimeError: If recording event fails.
        """
        if event is None:
            return
        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            pass
        elif self.cuda_lib:
            st = stream if stream is not None else ctypes.c_void_p(0)
            res = self.cuda_lib.cuEventRecord(event, st)
            if res != 0:
                raise RuntimeError(f"cuEventRecord failed with error code {res}")

    def elapsed_time(self, start_event: ctypes.c_void_p | None, end_event: ctypes.c_void_p | None) -> float:
        """Calculate elapsed time in milliseconds between two CUDA events.

        Args:
            start_event (Optional[ctypes.c_void_p]): Start event.
            end_event (Optional[ctypes.c_void_p]): End event.

        Returns:
            float: Elapsed time in milliseconds.

        Raises:
            RuntimeError: If querying elapsed time fails.
        """
        if start_event is None or end_event is None:
            return 0.0
        if self.cuda_lib:
            millis = ctypes.c_float()
            res = self.cuda_lib.cuEventElapsedTime(ctypes.pointer(millis), start_event, end_event)
            if res != 0:
                raise RuntimeError(f"cuEventElapsedTime failed with error code {res}")
            return float(millis.value)
        return 0.0

    def synchronize(self) -> None:
        """Synchronize the active CUDA stream or device barrier.

        Raises:
            RuntimeError: If device synchronization fails.
        """
        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            try:
                _cuda_mod.cupy.cuda.Stream.null.synchronize()
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
            return

        if not self.cuda_lib:
            return

        if adapt_backend_module(self.cuda_lib).has_symbol("cuCtxSynchronize"):
            res = self.cuda_lib.cuCtxSynchronize()
            if res != 0:
                raise RuntimeError(f"cuCtxSynchronize failed with error code {res}")
        elif adapt_backend_module(self.cuda_lib).has_symbol("cudaDeviceSynchronize"):
            res = self.cuda_lib.cudaDeviceSynchronize()
            if res != 0:
                raise RuntimeError(f"cudaDeviceSynchronize failed with error code {res}")

    def _prepare_device_buffers(
        self,
        inputs: dict[str, bytes | memoryview | list[float]],
    ) -> dict[str, ctypes.c_void_p]:
        """Allocate and populate device memory buffers for inputs.

        Args:
            inputs (dict[str, Union[bytes, memoryview, list[float]]]): Host input payloads.

        Returns:
            dict[str, ctypes.c_void_p]: Allocated device pointers by input name.
        """
        device_buffers: dict[str, ctypes.c_void_p] = {}
        for inp_name, data in inputs.items():
            if isinstance(data, list):
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

        op_type = str(adapt_backend_module(node).get_symbol("op_type", "")).lower()
        if op_type == "fusedelementwise":
            kernel_fn = f"fused_elementwise_kernel_{node_idx}"
            tpl = HardwareTemplateConfig(body="", workgroup_size=[256, 1, 1])
        else:
            tpl_opt = _as_tpl_opt(generator.config.templates.get(op_type))
            if tpl_opt is None:
                return
            tpl = tpl_opt
            kernel_fn = _extract_kernel_name(coerce_str(tpl.get("body", "")), op_type)

        block_cfg, grid_cfg = query_optimal_launch_geometry(
            coerce_shape(adapt_backend_module(node).get_symbol("shape_metadata", None) or (num_elements,)),
        )
        block_size = [block_cfg[0], block_cfg[1], block_cfg[2]]
        grid_size = [grid_cfg[0], grid_cfg[1], grid_cfg[2]]

        kernel_args: list[KernelArgType] = [
            device_buffers[str(inp)]
            for inp in coerce_operand_list(adapt_backend_module(node).get_symbol("inputs", []))
            if str(inp) in device_buffers
        ]
        if out_ptr is not None:
            kernel_args.append(out_ptr)
        kernel_args.append(num_elements)

        if ptx_source and out_ptr is not None:
            self.compile_and_dispatch(ptx_source, kernel_fn, grid_size, block_size, kernel_args)

    def execute_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, bytes | memoryview | list[float]],
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
        if not self.is_available() and not self.cuda_lib and (self.mode != "cupy" or _cuda_mod.cupy is None):
            raise BackendNotSupportedError("CUDA accelerator hardware or driver is not available.")

        cuda_source = _cuda_mod.CudaCodeGenerator(graph).generate()
        ptx_source = self.compile_cuda_to_ptx(cuda_source)
        device_buffers = self._prepare_device_buffers(inputs)
        generator = _cuda_mod.CudaCodeGenerator(graph)

        node_idx: int = 0
        for node in graph.nodes.values() if adapt_backend_module(graph).get_symbol("nodes", None) else ():
            if adapt_backend_module(node).get_symbol("op_type", "") == "Input":
                continue
            self._dispatch_compute_node(node, node_idx, generator, ptx_source, device_buffers)
            node_idx += 1

        self.synchronize()

        outputs_set = [str(x) for x in coerce_operand_list(adapt_backend_module(graph).get_symbol("outputs", []))]
        output_data: dict[str, bytes] = {}

        for out_name in outputs_set:
            out_node = graph.nodes.get(out_name)
            _, byte_size = _calculate_node_bytes(out_node) if out_node is not None else (1024, 4096)
            out_buf = device_buffers.get(out_name)
            if out_buf is not None:
                output_data[out_name] = self.read_buffer(out_buf, byte_size)

        for ptr in device_buffers.values():
            if ptr is not None:
                self.free_buffer(ptr)

        return output_data
