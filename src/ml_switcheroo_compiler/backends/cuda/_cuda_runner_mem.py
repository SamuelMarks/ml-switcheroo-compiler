"""CUDA device memory management and NVRTC/PTX compilation layer."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os

import yaml

import ml_switcheroo_compiler.backends.cuda.cuda as _cuda_mod
from ml_switcheroo_compiler.backends.cuda._cuda_helpers import (
    KernelArgType,
    _marshall_kernel_params,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_bytes,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _CUDARunnerMemory:
    """Device buffer allocation, host-device transfer, and PTX compilation support."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if CUDA driver and accelerator hardware are functional.

        Returns:
            bool: True if CUDA is functional, False otherwise.
        """
        if _cuda_mod.cupy is not None:
            try:
                return bool(_cuda_mod.cupy.cuda.is_available() and _cuda_mod.cupy.cuda.runtime.getDeviceCount() > 0)
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        try:
            cuda_lib_path = ctypes.util.find_library("cuda")
            if cuda_lib_path:
                lib = ctypes.cdll.LoadLibrary(cuda_lib_path)
                if adapt_backend_module(lib).has_symbol("cuInit") and lib.cuInit(0) == 0:
                    count = ctypes.c_int()
                    if (
                        adapt_backend_module(lib).has_symbol("cuDeviceGetCount")
                        and lib.cuDeviceGetCount(ctypes.byref(count)) == 0
                    ):
                        return count.value > 0
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return False

    def __init__(self) -> None:
        """Initialize CUDARunner and load hardware limits."""
        self.limits: dict[str, int | list[int] | str | None] = {}
        limits_path: str = os.path.join(os.path.dirname(__file__), "hardware_limits.yaml")
        if os.path.exists(limits_path):
            with open(limits_path) as f:
                self.limits = yaml.safe_load(f) or {}

        self.cuda_lib: ctypes.CDLL | None = None
        self._ctx: ctypes.c_void_p | None = None

        if _cuda_mod.cupy is not None:
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
            except (OSError, AttributeError, TypeError, ValueError, RuntimeError, ctypes.ArgumentError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    def allocate_buffer(self, size: int) -> ctypes.c_void_p | None:
        """Allocate device memory.

        Args:
            size (int): Size in bytes.

        Returns:
            Optional[ctypes.c_void_p]: Pointer to device memory.

        Raises:
            RuntimeError: If execution or validation fails in `allocate_buffer`.
        """
        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            mem = _cuda_mod.cupy.cuda.alloc(size)
            return ctypes.c_void_p(mem.ptr)

        if not self.cuda_lib:
            return ctypes.c_void_p(0)

        dptr = ctypes.c_void_p()
        res = self.cuda_lib.cuMemAlloc_v2(ctypes.byref(dptr), size)
        if res != 0:
            raise RuntimeError(f"cuMemAlloc failed with error code {res}")
        return dptr

    def write_buffer(self, device_ptr: ctypes.c_void_p | None, host_data: bytes) -> None:
        """Write to device buffer.

        Args:
            device_ptr (ctypes.c_void_p | None): Device pointer.
            host_data (bytes): Data to write.

        Raises:
            RuntimeError: If execution or validation fails in `write_buffer`.
        """
        if device_ptr is None or not (
            adapt_backend_module(device_ptr).get_symbol("value", None)
            if adapt_backend_module(device_ptr).has_symbol("value")
            else device_ptr
        ):
            return

        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            arr = _cuda_mod.cupy.asarray(bytearray(host_data), dtype="uint8")
            dst = _cuda_mod.cupy.ndarray(
                arr.shape,
                dtype=arr.dtype,
                memptr=_cuda_mod.cupy.cuda.MemoryPointer(
                    _cuda_mod.cupy.cuda.UnownedMemory(device_ptr.value, arr.nbytes, device_ptr), 0
                ),
            )
            dst.set(arr)
            return

        if self.cuda_lib:
            res = self.cuda_lib.cuMemcpyHtoD_v2(device_ptr, host_data, len(host_data))
            if res != 0:
                raise RuntimeError(f"cuMemcpyHtoD failed with error code {res}")

    def read_buffer(self, device_ptr: ctypes.c_void_p | None, size: int) -> bytes:
        """Read from device buffer.

        Args:
            device_ptr (ctypes.c_void_p | None): Device pointer.
            size (int): Size to read.

        Returns:
            bytes: Read data.

        Raises:
            RuntimeError: If execution or validation fails in `read_buffer`.
        """
        if device_ptr is None or not (
            adapt_backend_module(device_ptr).get_symbol("value", None)
            if adapt_backend_module(device_ptr).has_symbol("value")
            else device_ptr
        ):
            return b"\x00" * size

        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            src = _cuda_mod.cupy.ndarray(
                (size,),
                dtype=_cuda_mod.cupy.uint8,
                memptr=_cuda_mod.cupy.cuda.MemoryPointer(
                    _cuda_mod.cupy.cuda.UnownedMemory(device_ptr.value, size, device_ptr), 0
                ),
            )
            arr = src.get()
            return coerce_bytes(arr.tobytes())

        if self.cuda_lib:
            host_data = ctypes.create_string_buffer(size)
            res = self.cuda_lib.cuMemcpyDtoH_v2(host_data, device_ptr, size)
            if res != 0:
                raise RuntimeError(f"cuMemcpyDtoH failed with error code {res}")
            return host_data.raw
        return b"\x00" * size

    def free_buffer(self, device_ptr: ctypes.c_void_p | None) -> None:
        """Free device memory.

        Args:
            device_ptr (ctypes.c_void_p | None): Device pointer.
        """
        if device_ptr is None or not (
            adapt_backend_module(device_ptr).get_symbol("value", None)
            if adapt_backend_module(device_ptr).has_symbol("value")
            else device_ptr
        ):
            return

        if self.mode == "cupy":
            pass
        elif self.cuda_lib:
            self.cuda_lib.cuMemFree_v2(device_ptr)

    def compile_cuda_to_ptx(
        self,
        cuda_source: str,
        source_filename: str = "kernel.cu",
        options: list[str] | None = None,
    ) -> str:
        """Compile CUDA source code to PTX string.

        Args:
            cuda_source (str): CUDA C++ source code.
            source_filename (str): Virtual filename.
            options (Optional[list[str]]): Compiler options.

        Returns:
            str: Compiled PTX code.

        Raises:
            RuntimeError: If execution or validation fails in `compile_cuda_to_ptx`.
        """
        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            try:
                opts = tuple(options) if options else ()
                if adapt_backend_module(_cuda_mod.cupy.cuda.nvrtc).has_symbol("create_program"):
                    _cuda_mod.cupy.cuda.nvrtc.create_program(cuda_source, source_filename)
                ptx_bytes = _cuda_mod.cupy.cuda.nvrtc.get_ptx(cuda_source, options=opts)
                if isinstance(ptx_bytes, bytes):
                    return ptx_bytes.decode("utf-8")
                return str(ptx_bytes)
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

        try:
            nvrtc_path = ctypes.util.find_library("nvrtc") or "libnvrtc.so"
            nvrtc_lib = ctypes.cdll.LoadLibrary(nvrtc_path)
        except (OSError, AttributeError, TypeError, ValueError, RuntimeError, ctypes.ArgumentError) as err:
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
            if adapt_backend_module(nvrtc_lib).has_symbol("nvrtcDestroyProgram"):
                nvrtc_lib.nvrtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"NVRTC compilation failed: {log_msg}")

        ptx_size = ctypes.c_size_t()
        nvrtc_lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        nvrtc_lib.nvrtcGetPTX(prog, ptx_buf)
        if adapt_backend_module(nvrtc_lib).has_symbol("nvrtcDestroyProgram"):
            nvrtc_lib.nvrtcDestroyProgram(ctypes.byref(prog))
        return ptx_buf.value.decode("utf-8", errors="replace")

    def compile_and_dispatch(
        self,
        ptx_source: str,
        entry_point: str,
        grid_size: list[int],
        block_size: list[int],
        args: list[KernelArgType] | None = None,
        shared_memory_bytes: int = 0,
        stream: ctypes.c_void_p | None = None,
    ) -> None:
        """Compile PTX modules and launch kernels with dynamic shared memory and stream support.

        Args:
            ptx_source (str): PTX source code.
            entry_point (str): Kernel entry point.
            grid_size (list[int]): Grid sizing.
            block_size (list[int]): Block sizing.
            args (Optional[list[KernelArgType]]): Kernel arguments to marshall to device.
            shared_memory_bytes (int): Dynamic shared memory allocation size in bytes.
            stream (Optional[ctypes.c_void_p]): Asynchronous CUDA stream.

        Raises:
            RuntimeError: If execution or validation fails in `compile_and_dispatch`.
        """
        if self.mode == "cupy" and _cuda_mod.cupy is not None:
            module = _cuda_mod.cupy.RawModule(code=ptx_source)
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

            h_stream = stream if stream is not None else ctypes.c_void_p(0)
            res = self.cuda_lib.cuLaunchKernel(
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
                raise RuntimeError(f"cuLaunchKernel failed with error code {res}")
