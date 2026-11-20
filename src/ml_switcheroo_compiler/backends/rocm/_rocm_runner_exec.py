"""ROCm HIP stream synchronization and graph execution runner."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import tempfile
import typing

import ml_switcheroo_compiler.backends.rocm.rocm as _rocm_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator
from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    coerce_str,
)
from ml_switcheroo_compiler.backends.hardware_config_models import (
    query_optimal_launch_geometry,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    KernelArgType,
    _calculate_node_bytes,
    _extract_kernel_name,
    _graph_nodes_values,
    _graph_str_list,
    _node_inputs_list,
)
from ml_switcheroo_compiler.backends.rocm._rocm_runner_mem import _ROCmRunnerMemory
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


class ROCmRunner(_ROCmRunnerMemory):
    """ROCm Runner utilizing cupy if available, fallback to ctypes driver API."""

    def create_stream(self) -> ctypes.c_void_p | None:
        """Create an asynchronous ROCm HIP stream.

        Returns:
            Optional[ctypes.c_void_p]: Pointer to created HIP stream.

        Raises:
            RuntimeError: If execution or validation fails in `create_stream`.
        """
        if self.mode == "cupy" and _rocm_mod.cupy is not None:
            stream = _rocm_mod.cupy.cuda.Stream()
            ptr_val = adapt_backend_module(stream).get_symbol("ptr", 1000)
            return ctypes.c_void_p(int(ptr_val) if ptr_val else 1000)
        if self.rocm_lib:
            stream = ctypes.c_void_p()
            res = self.rocm_lib.hipStreamCreate(ctypes.byref(stream))
            if res != 0:
                raise RuntimeError(f"hipStreamCreate failed with error code {res}")
            return stream
        return None

    def synchronize_stream(self, stream: ctypes.c_void_p | None = None) -> None:
        """Synchronize an asynchronous ROCm HIP stream.

        Args:
            stream (Optional[ctypes.c_void_p]): Stream to synchronize. Defaults to None.

        Raises:
            RuntimeError: If stream synchronization fails.
        """
        if self.mode == "cupy" and _rocm_mod.cupy is not None:
            if stream is not None and stream.value:
                _rocm_mod.cupy.cuda.Stream(null=False).synchronize()
            else:
                _rocm_mod.cupy.cuda.Device().synchronize()
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
        if self.mode == "cupy" and _rocm_mod.cupy is not None:
            try:
                _rocm_mod.cupy.cuda.Stream.null.synchronize()
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
            return

        if not self.rocm_lib:
            return

        if adapt_backend_module(self.rocm_lib).has_symbol("hipDeviceSynchronize"):
            res = self.rocm_lib.hipDeviceSynchronize()
            if res != 0:
                raise RuntimeError(f"hipDeviceSynchronize failed with error code {res}")

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
        node: EagerTensorProtocol | IRNode,
        node_idx: int,
        generator: RocmCodeGenerator,
        tmp_code_path: str | None,
        device_buffers: dict[str, ctypes.c_void_p],
    ) -> None:
        """Dispatch compute kernel execution for a single IRNode.

        Args:
            node (EagerTensorProtocol | IRNode): The IR node to execute.
            node_idx (int): Current node execution index.
            generator (RocmCodeGenerator): ROCm code generator holding op templates.
            tmp_code_path (Optional[str]): Compiled code object file path.
            device_buffers (dict[str, ctypes.c_void_p]): Active device buffer allocations.
        """
        num_elements, byte_size = _calculate_node_bytes(node)
        out_ptr = self.allocate_buffer(byte_size)
        if out_ptr is not None:
            device_buffers[coerce_str(adapt_backend_module(node).get_symbol("id", ""))] = out_ptr

        op_type = adapt_backend_module(node).get_symbol("op_type", "").lower()
        if op_type == "fusedelementwise":
            kernel_fn = f"fused_elementwise_kernel_{node_idx}"
        else:
            tpl_opt = generator.config.templates.get(op_type)
            if tpl_opt is None:
                return
            kernel_fn = _extract_kernel_name(
                coerce_str(adapt_backend_module(tpl_opt).require_symbol("get")("body", "")), op_type
            )

        raw_sh = adapt_backend_module(node).get_symbol("shape_metadata", None)
        shape_arg: tuple[int | str, ...] = (
            tuple(int(x) if isinstance(x, (int, float)) else str(x) for x in raw_sh)
            if isinstance(raw_sh, (list, tuple)) and raw_sh
            else (num_elements,)
        )
        block_cfg, grid_cfg = query_optimal_launch_geometry(shape_arg)
        block_size = [block_cfg[0], block_cfg[1], block_cfg[2]]
        grid_size = [grid_cfg[0], grid_cfg[1], grid_cfg[2]]

        kernel_args: list[KernelArgType] = [
            device_buffers[inp] for inp in _node_inputs_list(node) if inp in device_buffers
        ]
        if out_ptr is not None:
            kernel_args.append(out_ptr)
        kernel_args.append(num_elements)

        if tmp_code_path and out_ptr is not None:
            self.load_and_dispatch(tmp_code_path, kernel_fn, grid_size, block_size, kernel_args)

    def execute_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, bytes | memoryview | list[float]],
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
        if not self.is_available() and not self.rocm_lib and (self.mode != "cupy" or _rocm_mod.cupy is None):
            raise BackendNotSupportedError("ROCm HIP accelerator hardware or driver is not available.")

        rocm_source = _rocm_mod.RocmCodeGenerator(graph).generate()
        code_bytes = self.compile_hip_to_code(rocm_source)

        device_buffers = self._prepare_device_buffers(inputs)
        generator = _rocm_mod.RocmCodeGenerator(graph)
        node_idx: int = 0
        tmp_code_path: str | None = None
        if code_bytes:
            with tempfile.NamedTemporaryFile(suffix=".co", delete=False) as tf:
                tf.write(code_bytes)
                tmp_code_path = tf.name

        try:
            for node in _graph_nodes_values(graph):
                if adapt_backend_module(node).get_symbol("op_type", "") == "Input":
                    continue
                self._dispatch_compute_node(node, node_idx, generator, tmp_code_path, device_buffers)
                node_idx += 1

            self.synchronize()
        finally:
            if tmp_code_path and os.path.exists(tmp_code_path):
                os.remove(tmp_code_path)

        outputs_list = _graph_str_list(graph, "outputs")
        output_data: dict[str, bytes] = {}

        for out_name in outputs_list:
            out_node = graph.nodes.get(out_name)
            _, byte_size = _calculate_node_bytes(out_node) if out_node else (1024, 4096)
            out_buf = device_buffers.get(out_name)
            if out_buf is not None:
                output_data[out_name] = self.read_buffer(out_buf, byte_size)

        for ptr in device_buffers.values():
            if ptr is not None:
                self.free_buffer(ptr)

        return output_data
