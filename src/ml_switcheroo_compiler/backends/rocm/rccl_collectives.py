"""Ctypes-based RCCL collective driver bindings for ROCm/HIP."""

from __future__ import annotations

import ctypes
import logging
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_int
from ml_switcheroo_compiler.backends.protocols import get_backend_member, has_backend_member
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

# RCCL Data types
RCCL_FLOAT32: int = 7
RCCL_FLOAT64: int = 8
RCCL_INT32: int = 2
RCCL_INT64: int = 3

# RCCL Reduce operations
RCCL_SUM: int = 0
RCCL_PROD: int = 1
RCCL_MAX: int = 2
RCCL_MIN: int = 3


class _RcclUniqueId(ctypes.Structure):
    """Internal 128-byte ctypes Structure representing rcclUniqueId / ncclUniqueId."""

    _fields_ = [("internal", ctypes.c_char * 128)]


_CArgT = typing.TypeVar("_CArgT")


class _CtypesFuncProtocol(typing.Protocol):
    """Protocol for ctypes C function objects."""

    argtypes: EagerValue
    restype: EagerValue

    def __call__(self, *args: _CArgT) -> EagerValue:
        """Invoke ctypes C function.

        Args:
            *args (_CArgT): C function arguments.

        Returns:
            EagerValue: Return value.
        """
        ...


def _as_rccl_fn(val: EagerValue) -> _CtypesFuncProtocol | None:
    """Narrow `val` to `_CtypesFuncProtocol | None`.

    Args:
        val (EagerValue): Candidate C function symbol.

    Returns:
        _CtypesFuncProtocol | None: Typed ctypes function or `None`.
    """

    def _guard(v: EagerValue) -> typing.TypeGuard[_CtypesFuncProtocol]:
        """Type guard narrowing `v` to `_CtypesFuncProtocol`.

        Args:
            v (EagerValue): Candidate value.

        Returns:
            bool: True if non-None.
        """
        return v is not None

    return val if _guard(val) else None


class RCCLDriver:
    """Dynamic C-library loader and caller for AMD ROCm RCCL."""

    def __init__(self, lib_path: str | None = None) -> None:
        """Initialize RCCL driver loading librccl.so if available.

        Args:
            lib_path (str | None): Custom path to librccl.so or None for discovery.
        """
        self.rccl_lib: ctypes.CDLL | None = None
        self._allocated_buffers: dict[int, EagerValue] = {}
        self.hip_rt: ctypes.CDLL | None = None
        for rt in ["libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so", "libhip_hcc.so", "hip.dll"]:
            try:
                self.hip_rt = ctypes.CDLL(rt)
                break
            except (OSError, Exception):
                continue

        paths: list[str] = [lib_path] if lib_path is not None else ["librccl.so.1", "librccl.so", "librccl.dylib", "rccl.dll"]
        for p in paths:
            if not p:
                continue
            try:
                self.rccl_lib = ctypes.CDLL(p)
                break
            except (OSError, Exception):
                continue

    def is_available(self) -> bool:
        """Check if RCCL library was successfully loaded.

        Returns:
            bool: True if librccl is linked and loaded.
        """
        return self.rccl_lib is not None

    def allocate_buffer(self, size_bytes: int) -> int:
        """Allocate an accelerator memory buffer for collective execution.

        Args:
            size_bytes (int): Total number of bytes to allocate.

        Returns:
            int: Memory address pointer value.
        """
        if self.hip_rt is not None and has_backend_member(self.hip_rt, "hipMalloc"):
            ptr = ctypes.c_void_p()
            fn = self.hip_rt.hipMalloc
            fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            fn.restype = ctypes.c_int
            status = int(ensure_int(fn(ctypes.byref(ptr), max(1, size_bytes))))
            if status == 0 and ptr.value is not None:
                return int(ptr.value)

        buf = (ctypes.c_char * max(1, size_bytes))()
        ptr_val = ctypes.addressof(buf)
        self._allocated_buffers[ptr_val] = buf
        return ptr_val

    def free_buffer(self, ptr: int) -> None:
        """Release an allocated accelerator memory buffer.

        Args:
            ptr (int): Pointer address to free.
        """
        if self.hip_rt is not None and has_backend_member(self.hip_rt, "hipFree") and ptr not in self._allocated_buffers:
            try:
                fn = self.hip_rt.hipFree
                fn.argtypes = [ctypes.c_void_p]
                fn.restype = ctypes.c_int
                fn(ctypes.c_void_p(ptr))
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        else:
            self._allocated_buffers.pop(ptr, None)

    def copy_host_to_device(self, host_array: EagerValue, dev_ptr: int) -> None:
        """Transfer memory buffer from host NumPy array to accelerator device pointer.

        Args:
            host_array (EagerValue): Source host NumPy array or buffer.
            dev_ptr (int): Destination accelerator device pointer address.
        """
        raw_ct = get_backend_member(host_array, "ctypes", None)
        raw_dp = get_backend_member(raw_ct, "data", None) if raw_ct is not None else None
        buf_ptr: int | None = int(ensure_int(raw_dp)) if raw_dp is not None else None
        nbytes = int(ensure_int(get_backend_member(host_array, "nbytes", 0)))
        if self.hip_rt is not None and has_backend_member(self.hip_rt, "hipMemcpy") and dev_ptr not in self._allocated_buffers:
            try:
                fn = self.hip_rt.hipMemcpy
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                fn.restype = ctypes.c_int
                # hipMemcpyHostToDevice = 1
                fn(ctypes.c_void_p(dev_ptr), ctypes.c_void_p(buf_ptr), nbytes, 1)
                return
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        ensure_callable(get_backend_member(ctypes, "memmove"))(dev_ptr, buf_ptr, nbytes)

    def copy_device_to_host(self, dev_ptr: int, out_array: EagerValue) -> None:
        """Transfer memory buffer from accelerator device pointer to destination host NumPy array.

        Args:
            dev_ptr (int): Source accelerator device pointer address.
            out_array (EagerValue): Destination host NumPy array or buffer.
        """
        raw_ct = get_backend_member(out_array, "ctypes", None)
        raw_dp = get_backend_member(raw_ct, "data", None) if raw_ct is not None else None
        buf_ptr: int | None = int(ensure_int(raw_dp)) if raw_dp is not None else None
        nbytes = int(ensure_int(get_backend_member(out_array, "nbytes", 0)))
        if self.hip_rt is not None and has_backend_member(self.hip_rt, "hipMemcpy") and dev_ptr not in self._allocated_buffers:
            try:
                fn = self.hip_rt.hipMemcpy
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                fn.restype = ctypes.c_int
                # hipMemcpyDeviceToHost = 2
                fn(ctypes.c_void_p(buf_ptr), ctypes.c_void_p(dev_ptr), nbytes, 2)
                return
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        ensure_callable(get_backend_member(ctypes, "memmove"))(buf_ptr, dev_ptr, nbytes)

    def stream_synchronize(self, stream: int | None = None) -> int:
        """Synchronize HIP compute stream.

        Args:
            stream (int | None): HIP stream pointer to synchronize.

        Returns:
            int: Status code (0 = success).
        """
        if stream is None:
            return 0
        if self.hip_rt is not None and has_backend_member(self.hip_rt, "hipStreamSynchronize"):
            try:
                fn = self.hip_rt.hipStreamSynchronize
                fn.argtypes = [ctypes.c_void_p]
                fn.restype = ctypes.c_int
                return int(ensure_int(fn(ctypes.c_void_p(stream))))
            except Exception:
                return -1
        return 0

    def _check_available(self) -> None:
        """Verify that RCCL hardware and shared library are available.

        Raises:
            BackendNotSupportedError: When librccl is unavailable.
        """
        if not self.is_available() or self.rccl_lib is None:
            raise BackendNotSupportedError("RCCL library is not available on this system.")

    def all_reduce(
        self,
        sendbuff: int,
        recvbuff: int,
        count: int,
        datatype: int = RCCL_FLOAT32,
        op: int = RCCL_SUM,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclAllReduce / rcclAllReduce over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            count (int): Number of elements.
            datatype (int): RCCL data type code.
            op (int): RCCL reduction operation code.
            comm (int | None): Communicator pointer.
            stream (int | None): HIP stream pointer.

        Returns:
            int: Result return status (0 = success).
        """
        self._check_available()
        try:
            fn = _as_rccl_fn(get_backend_member(self.rccl_lib, "ncclAllReduce", get_backend_member(self.rccl_lib, "rcclAllReduce", None)))
            if fn is None:
                return -1
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int
            return int(ensure_int(fn(sendbuff, recvbuff, count, datatype, op, comm or 0, stream or 0)))
        except Exception:
            return -1

    def all_gather(
        self,
        sendbuff: int,
        recvbuff: int,
        sendcount: int,
        datatype: int = RCCL_FLOAT32,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclAllGather / rcclAllGather over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            sendcount (int): Number of elements contributed by this rank.
            datatype (int): RCCL data type code.
            comm (int | None): Communicator pointer.
            stream (int | None): HIP stream pointer.

        Returns:
            int: Result status.
        """
        self._check_available()
        try:
            fn = _as_rccl_fn(get_backend_member(self.rccl_lib, "ncclAllGather", get_backend_member(self.rccl_lib, "rcclAllGather", None)))
            if fn is None:
                return -1
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int
            return int(ensure_int(fn(sendbuff, recvbuff, sendcount, datatype, comm or 0, stream or 0)))
        except Exception:
            return -1

    def broadcast(
        self,
        sendbuff: int,
        recvbuff: int,
        count: int,
        datatype: int = RCCL_FLOAT32,
        root: int = 0,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclBroadcast / rcclBroadcast over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            count (int): Number of elements.
            datatype (int): RCCL data type code.
            root (int): Root leader rank.
            comm (int | None): Communicator pointer.
            stream (int | None): HIP stream pointer.

        Returns:
            int: Result status.
        """
        self._check_available()
        try:
            fn = _as_rccl_fn(get_backend_member(self.rccl_lib, "ncclBroadcast", get_backend_member(self.rccl_lib, "rcclBroadcast", None)))
            if fn is None:
                return -1
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int
            return int(ensure_int(fn(sendbuff, recvbuff, count, datatype, root, comm or 0, stream or 0)))
        except Exception:
            return -1

    def reduce_scatter(
        self,
        sendbuff: int,
        recvbuff: int,
        recvcount: int,
        datatype: int = RCCL_FLOAT32,
        op: int = RCCL_SUM,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclReduceScatter / rcclReduceScatter over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            recvcount (int): Number of elements received by this rank.
            datatype (int): RCCL data type code.
            op (int): RCCL reduction operator code.
            comm (int | None): Communicator pointer.
            stream (int | None): HIP stream pointer.

        Returns:
            int: Result status.
        """
        self._check_available()
        try:
            fn = _as_rccl_fn(get_backend_member(self.rccl_lib, "ncclReduceScatter", get_backend_member(self.rccl_lib, "rcclReduceScatter", None)))
            if fn is None:
                return -1
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int
            return int(ensure_int(fn(sendbuff, recvbuff, recvcount, datatype, op, comm or 0, stream or 0)))
        except Exception:
            return -1

    def all_to_all(
        self,
        sendbuff: int,
        recvbuff: int,
        count: int,
        datatype: int = RCCL_FLOAT32,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclAllToAll / rcclAllToAll over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            count (int): Number of elements per rank.
            datatype (int): RCCL data type code.
            comm (int | None): Communicator pointer.
            stream (int | None): HIP stream pointer.

        Returns:
            int: Result status.
        """
        self._check_available()
        try:
            fn = _as_rccl_fn(get_backend_member(self.rccl_lib, "ncclAllToAll", get_backend_member(self.rccl_lib, "rcclAllToAll", None)))
            if fn is not None:
                fn.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
                fn.restype = ctypes.c_int
                return int(ensure_int(fn(sendbuff, recvbuff, count, datatype, comm or 0, stream or 0)))
            return 0
        except Exception:
            return -1

    def get_unique_id(self) -> bytes:
        """Generate a new RCCL unique communicator ID.

        Returns:
            bytes: 128-byte raw rcclUniqueId buffer.

        Raises:
            RuntimeError: If unique ID retrieval fails.
        """
        self._check_available()
        uid: _RcclUniqueId = _RcclUniqueId()
        fn = _as_rccl_fn(get_backend_member(self.rccl_lib, "ncclGetUniqueId", get_backend_member(self.rccl_lib, "rcclGetUniqueId", None)))
        if fn is None:
            raise RuntimeError("ncclGetUniqueId not found in rccl library")
        fn.argtypes = [ctypes.POINTER(_RcclUniqueId)]
        fn.restype = ctypes.c_int
        status: int = int(ensure_int(fn(ctypes.byref(uid))))
        if status != 0:
            raise RuntimeError(f"rcclGetUniqueId failed with status {status}")
        return bytes(uid.internal)

    def init_rank(self, nranks: int, comm_id: bytes, rank: int) -> int:
        """Initialize a new RCCL communicator for a rank.

        Args:
            nranks (int): Total number of participating ranks.
            comm_id (bytes): Unique ID generated by root via get_unique_id().
            rank (int): Rank index of this process.

        Returns:
            int: Pointer value to communicator.

        Raises:
            RuntimeError: If comm init rank fails.
        """
        self._check_available()
        uid: _RcclUniqueId = _RcclUniqueId()
        ctypes.memmove(ctypes.byref(uid), comm_id[:128], min(128, len(comm_id)))

        comm = ctypes.c_void_p()
        fn = _as_rccl_fn(get_backend_member(self.rccl_lib, "ncclCommInitRank", get_backend_member(self.rccl_lib, "rcclCommInitRank", None)))
        if fn is None:
            raise RuntimeError("ncclCommInitRank not found in rccl library")
        fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _RcclUniqueId, ctypes.c_int]
        fn.restype = ctypes.c_int
        status: int = int(ensure_int(fn(ctypes.byref(comm), nranks, uid, rank)))
        if status != 0:
            raise RuntimeError(f"rcclCommInitRank failed with status {status}")
        return int(comm.value or 0)

    def comm_destroy(self, comm: int) -> int:
        """Destroy an allocated RCCL communicator.

        Args:
            comm (int): Pointer value to communicator.

        Returns:
            int: Result status.
        """
        self._check_available()
        try:
            fn = _as_rccl_fn(get_backend_member(self.rccl_lib, "ncclCommDestroy", get_backend_member(self.rccl_lib, "rcclCommDestroy", None)))
            if fn is None:
                return -1
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
            return int(ensure_int(fn(ctypes.c_void_p(comm))))
        except Exception:
            return -1
