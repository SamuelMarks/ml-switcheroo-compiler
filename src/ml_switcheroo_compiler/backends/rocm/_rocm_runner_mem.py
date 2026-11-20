"""ROCm HIP device memory management and hiprtc compilation layer."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os

import yaml

import ml_switcheroo_compiler.backends.rocm.rocm as _rocm_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_bytes,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.rocm._rocm_helpers import (
    KernelArgType,
    _marshall_kernel_params,
)


class _ROCmRunnerMemory:
    """HIP buffer allocation, host-device transfer, and kernel compilation support."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if ROCm HIP driver and accelerator hardware are functional.

        Returns:
            bool: True if ROCm HIP is functional, False otherwise.
        """
        if _rocm_mod.cupy is not None:
            try:
                return bool(
                    adapt_backend_module(_rocm_mod.cupy.cuda).has_symbol("is_hip") and _rocm_mod.cupy.cuda.is_hip
                )
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        try:
            hip_path = ctypes.util.find_library("amdhip64") or ctypes.util.find_library("hip_hcc")
            if hip_path:
                lib = ctypes.cdll.LoadLibrary(hip_path)
                if adapt_backend_module(lib).has_symbol("hipInit") and lib.hipInit(0) == 0:
                    count = ctypes.c_int()
                    if (
                        adapt_backend_module(lib).has_symbol("hipGetDeviceCount")
                        and lib.hipGetDeviceCount(ctypes.byref(count)) == 0
                    ):
                        return count.value > 0
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return False

    def __init__(self) -> None:
        """Initialize ROCmRunner and load hardware limits."""
        self.limits: dict[str, int | list[int] | str | None] = {}
        limits_path: str = os.path.join(os.path.dirname(__file__), "hardware_limits.yaml")
        if os.path.exists(limits_path):
            with open(limits_path) as f:
                self.limits = yaml.safe_load(f) or {}

        self.rocm_lib: ctypes.CDLL | None = None
        self._ctx: ctypes.c_void_p | None = None

        if _rocm_mod.cupy is not None and adapt_backend_module(
            adapt_backend_module(_rocm_mod.cupy).get_symbol("cuda", None)
        ).get_symbol("is_hip", False):
            self.mode = "cupy"
        else:
            self.mode = "ctypes"
            try:
                rocm_lib_path = ctypes.util.find_library("amdhip64")
                if rocm_lib_path:
                    self.rocm_lib = ctypes.cdll.LoadLibrary(rocm_lib_path)
                    self.rocm_lib.hipInit(0)
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
        if self.mode == "cupy" and _rocm_mod.cupy is not None:
            mem = _rocm_mod.cupy.cuda.alloc(size)
            return ctypes.c_void_p(mem.ptr)

        if not self.rocm_lib:
            return ctypes.c_void_p(0)

        dptr = ctypes.c_void_p()
        res = self.rocm_lib.hipMalloc(ctypes.byref(dptr), size)
        if res != 0:
            raise RuntimeError(f"hipMalloc failed with error code {res}")
        return dptr

    def write_buffer(self, device_ptr: ctypes.c_void_p | None, host_data: bytes) -> None:
        """Write to device buffer.

        Args:
            device_ptr (ctypes.c_void_p | None): Device pointer.
            host_data (bytes): Data to write.

        Raises:
            RuntimeError: If execution or validation fails in `write_buffer`.
        """
        if device_ptr is None or not adapt_backend_module(device_ptr).get_symbol("value", device_ptr):
            return

        if self.mode == "cupy" and _rocm_mod.cupy is not None:
            arr = _rocm_mod.cupy.asarray(bytearray(host_data), dtype="uint8")
            dst = _rocm_mod.cupy.ndarray(
                arr.shape,
                dtype=arr.dtype,
                memptr=_rocm_mod.cupy.cuda.MemoryPointer(
                    _rocm_mod.cupy.cuda.UnownedMemory(device_ptr.value, arr.nbytes, device_ptr), 0
                ),
            )
            dst.set(arr)
            return

        if self.rocm_lib:
            res = self.rocm_lib.hipMemcpyHtoD(device_ptr, host_data, len(host_data))
            if res != 0:
                raise RuntimeError(f"hipMemcpyHtoD failed with error code {res}")

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
        if device_ptr is None or not adapt_backend_module(device_ptr).get_symbol("value", device_ptr):
            return b"\x00" * size

        if self.mode == "cupy" and _rocm_mod.cupy is not None:
            src = _rocm_mod.cupy.ndarray(
                (size,),
                dtype=_rocm_mod.cupy.uint8,
                memptr=_rocm_mod.cupy.cuda.MemoryPointer(
                    _rocm_mod.cupy.cuda.UnownedMemory(device_ptr.value, size, device_ptr), 0
                ),
            )
            arr = src.get()
            return coerce_bytes(arr.tobytes())

        if self.rocm_lib:
            host_data = ctypes.create_string_buffer(size)
            res = self.rocm_lib.hipMemcpyDtoH(host_data, device_ptr, size)
            if res != 0:
                raise RuntimeError(f"hipMemcpyDtoH failed with error code {res}")
            return host_data.raw
        return b"\x00" * size

    def free_buffer(self, device_ptr: ctypes.c_void_p | None) -> None:
        """Free device memory.

        Args:
            device_ptr (ctypes.c_void_p | None): Device pointer.
        """
        if device_ptr is None or not adapt_backend_module(device_ptr).get_symbol("value", device_ptr):
            return

        if self.mode == "cupy":
            pass
        elif self.rocm_lib:
            self.rocm_lib.hipFree(device_ptr)

    def compile_hip_to_code(
        self,
        hip_source: str,
        source_filename: str = "kernel.hip",
        options: list[str] | None = None,
    ) -> bytes:
        """Compile HIP source code to executable code object bytes.

        Args:
            hip_source (str): HIP C++ source code.
            source_filename (str): Virtual source filename.
            options (Optional[list[str]]): Compiler flags.

        Returns:
            bytes: Compiled binary code object.

        Raises:
            RuntimeError: If execution or validation fails in `compile_hip_to_code`.
        """
        try:
            hiprtc_path = ctypes.util.find_library("hiprtc") or "libhiprtc.so"
            hiprtc_lib = ctypes.cdll.LoadLibrary(hiprtc_path)
        except (OSError, AttributeError, TypeError, ValueError, RuntimeError, ctypes.ArgumentError) as err:
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
            if adapt_backend_module(hiprtc_lib).has_symbol("hiprtcDestroyProgram"):
                hiprtc_lib.hiprtcDestroyProgram(ctypes.byref(prog))
            raise RuntimeError(f"HIPRTC compilation failed: {log_msg}")

        code_size = ctypes.c_size_t()
        hiprtc_lib.hiprtcGetCodeSize(prog, ctypes.byref(code_size))
        code_buf = ctypes.create_string_buffer(code_size.value)
        hiprtc_lib.hiprtcGetCode(prog, code_buf)
        if adapt_backend_module(hiprtc_lib).has_symbol("hiprtcDestroyProgram"):
            hiprtc_lib.hiprtcDestroyProgram(ctypes.byref(prog))
        return code_buf.raw

    def load_and_dispatch(
        self,
        binary_path: str,
        entry_point: str,
        grid_size: list[int],
        block_size: list[int],
        args: list[KernelArgType] | None = None,
        shared_memory_bytes: int = 0,
        stream: ctypes.c_void_p | None = None,
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

        Raises:
            RuntimeError: If execution or validation fails in `load_and_dispatch`.
        """
        if self.mode == "cupy" and _rocm_mod.cupy is not None:
            with open(binary_path, "rb") as f:
                code_bytes = f.read()
            module = _rocm_mod.cupy.RawModule(code=code_bytes)
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
