# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Ctypes-based NCCL collective driver bindings for CUDA."""

from __future__ import annotations

import ctypes
from typing import Optional

from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

# NCCL Data types
NCCL_FLOAT32: int = 7
NCCL_FLOAT64: int = 8
NCCL_INT32: int = 2
NCCL_INT64: int = 3

# NCCL Reduce operations
NCCL_SUM: int = 0
NCCL_PROD: int = 1
NCCL_MAX: int = 2
NCCL_MIN: int = 3


class _NcclUniqueId(ctypes.Structure):
    """Internal 128-byte ctypes Structure representing ncclUniqueId."""

    _fields_ = [("internal", ctypes.c_char * 128)]


class NCCLDriver:
    """Dynamic C-library loader and caller for NVIDIA NCCL."""

    def __init__(self, lib_path: str | None = None) -> None:
        """Initialize NCCL driver loading libnccl.so if available.

        Args:
            lib_path (str | None): Custom path to libnccl.so or None for discovery.
        """
        self.nccl_lib: ctypes.CDLL | None = None
        self._allocated_buffers: dict[int, object] = {}
        self.cuda_rt: ctypes.CDLL | None = None
        for rt in ["libcudart.so.12", "libcudart.so.11", "libcudart.so", "nvcuda.dll", "libcuda.dylib"]:
            try:
                self.cuda_rt = ctypes.CDLL(rt)
                break
            except (OSError, Exception):
                continue

        paths: list[str] = [lib_path] if lib_path is not None else ["libnccl.so.2", "libnccl.so", "libnccl.dylib", "nccl.dll"]
        for p in paths:
            if not p:
                continue
            try:
                self.nccl_lib = ctypes.CDLL(p)
                break
            except (OSError, Exception):
                continue

    def is_available(self) -> bool:
        """Check if NCCL library was successfully loaded.

        Returns:
            bool: True if libnccl is linked and loaded.
        """
        return self.nccl_lib is not None

    def allocate_buffer(self, size_bytes: int) -> int:
        """Allocate an accelerator memory buffer for collective execution.

        Args:
            size_bytes (int): Total number of bytes to allocate.

        Returns:
            int: Memory address pointer value.
        """
        if self.cuda_rt is not None and hasattr(self.cuda_rt, "cudaMalloc"):
            ptr = ctypes.c_void_p()
            fn = self.cuda_rt.cudaMalloc
            fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            fn.restype = ctypes.c_int
            status = int(fn(ctypes.byref(ptr), max(1, size_bytes)))
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
        if self.cuda_rt is not None and hasattr(self.cuda_rt, "cudaFree") and ptr not in self._allocated_buffers:
            try:
                fn = self.cuda_rt.cudaFree
                fn.argtypes = [ctypes.c_void_p]
                fn.restype = ctypes.c_int
                fn(ctypes.c_void_p(ptr))
            except Exception:
                pass
        else:
            self._allocated_buffers.pop(ptr, None)

    def copy_host_to_device(self, host_array: object, dev_ptr: int) -> None:
        """Transfer memory buffer from host NumPy array to accelerator device pointer.

        Args:
            host_array (object): Source host NumPy array or buffer.
            dev_ptr (int): Destination accelerator device pointer address.
        """
        buf_ptr = getattr(getattr(host_array, "ctypes", None), "data", None)
        nbytes = int(getattr(host_array, "nbytes", 0))
        if self.cuda_rt is not None and hasattr(self.cuda_rt, "cudaMemcpy") and dev_ptr not in self._allocated_buffers:
            try:
                fn = self.cuda_rt.cudaMemcpy
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                fn.restype = ctypes.c_int
                # cudaMemcpyHostToDevice = 1
                fn(ctypes.c_void_p(dev_ptr), ctypes.c_void_p(buf_ptr), nbytes, 1)
                return
            except Exception:
                pass
        ctypes.memmove(dev_ptr, buf_ptr, nbytes)

    def copy_device_to_host(self, dev_ptr: int, out_array: object) -> None:
        """Transfer memory buffer from accelerator device pointer to destination host NumPy array.

        Args:
            dev_ptr (int): Source accelerator device pointer address.
            out_array (object): Destination host NumPy array or buffer.
        """
        buf_ptr = getattr(getattr(out_array, "ctypes", None), "data", None)
        nbytes = int(getattr(out_array, "nbytes", 0))
        if self.cuda_rt is not None and hasattr(self.cuda_rt, "cudaMemcpy") and dev_ptr not in self._allocated_buffers:
            try:
                fn = self.cuda_rt.cudaMemcpy
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                fn.restype = ctypes.c_int
                # cudaMemcpyDeviceToHost = 2
                fn(ctypes.c_void_p(buf_ptr), ctypes.c_void_p(dev_ptr), nbytes, 2)
                return
            except Exception:
                pass
        ctypes.memmove(buf_ptr, dev_ptr, nbytes)

    def _check_available(self) -> None:
        """Verify that NCCL hardware and shared library are available.

        Raises:
            BackendNotSupportedError: When libnccl is unavailable.
        """
        if not self.is_available() or self.nccl_lib is None:
            raise BackendNotSupportedError("NCCL library is not available on this system.")

    def all_reduce(
        self,
        sendbuff: int,
        recvbuff: int,
        count: int,
        datatype: int = NCCL_FLOAT32,
        op: int = NCCL_SUM,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclAllReduce over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            count (int): Number of elements.
            datatype (int): NCCL data type code.
            op (int): NCCL reduction operation code.
            comm (int | None): ncclComm_t communicator pointer.
            stream (int | None): cudaStream_t pointer.

        Returns:
            int: ncclResult_t return status (0 = success).
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclAllReduce
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
            return int(fn(sendbuff, recvbuff, count, datatype, op, comm or 0, stream or 0))
        except Exception:
            return -1

    def all_gather(
        self,
        sendbuff: int,
        recvbuff: int,
        sendcount: int,
        datatype: int = NCCL_FLOAT32,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclAllGather over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            sendcount (int): Number of elements contributed by this rank.
            datatype (int): NCCL data type code.
            comm (int | None): ncclComm_t pointer.
            stream (int | None): cudaStream_t pointer.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclAllGather
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int
            return int(fn(sendbuff, recvbuff, sendcount, datatype, comm or 0, stream or 0))
        except Exception:
            return -1

    def broadcast(
        self,
        sendbuff: int,
        recvbuff: int,
        count: int,
        datatype: int = NCCL_FLOAT32,
        root: int = 0,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclBroadcast over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            count (int): Number of elements.
            datatype (int): NCCL data type code.
            root (int): Root leader rank.
            comm (int | None): ncclComm_t pointer.
            stream (int | None): cudaStream_t pointer.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclBroadcast
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
            return int(fn(sendbuff, recvbuff, count, datatype, root, comm or 0, stream or 0))
        except Exception:
            return -1

    def reduce_scatter(
        self,
        sendbuff: int,
        recvbuff: int,
        recvcount: int,
        datatype: int = NCCL_FLOAT32,
        op: int = NCCL_SUM,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclReduceScatter over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            recvcount (int): Number of elements received by this rank.
            datatype (int): NCCL data type code.
            op (int): NCCL reduction operator code.
            comm (int | None): ncclComm_t pointer.
            stream (int | None): cudaStream_t pointer.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclReduceScatter
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
            return int(fn(sendbuff, recvbuff, recvcount, datatype, op, comm or 0, stream or 0))
        except Exception:
            return -1

    def all_to_all(
        self,
        sendbuff: int,
        recvbuff: int,
        count: int,
        datatype: int = NCCL_FLOAT32,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclAllToAll over device pointers.

        Args:
            sendbuff (int): Device pointer to send buffer.
            recvbuff (int): Device pointer to receive buffer.
            count (int): Number of elements per rank.
            datatype (int): NCCL data type code.
            comm (int | None): ncclComm_t pointer.
            stream (int | None): cudaStream_t pointer.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            if hasattr(self.nccl_lib, "ncclAllToAll"):
                fn = self.nccl_lib.ncclAllToAll
                fn.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
                fn.restype = ctypes.c_int
                return int(fn(sendbuff, recvbuff, count, datatype, comm or 0, stream or 0))
            return 0
        except Exception:
            return -1

    def send(
        self,
        sendbuff: int,
        count: int,
        datatype: int = NCCL_FLOAT32,
        peer: int = 0,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclSend point-to-point transmission.

        Args:
            sendbuff (int): Device pointer to send buffer.
            count (int): Number of elements to send.
            datatype (int): NCCL data type code.
            peer (int): Destination peer rank.
            comm (int | None): ncclComm_t pointer.
            stream (int | None): cudaStream_t pointer.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclSend
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int
            return int(fn(sendbuff, count, datatype, peer, comm or 0, stream or 0))
        except Exception:
            return -1

    def recv(
        self,
        recvbuff: int,
        count: int,
        datatype: int = NCCL_FLOAT32,
        peer: int = 0,
        comm: int | None = None,
        stream: int | None = None,
    ) -> int:
        """Invoke ncclRecv point-to-point reception.

        Args:
            recvbuff (int): Device pointer to receive buffer.
            count (int): Number of elements to receive.
            datatype (int): NCCL data type code.
            peer (int): Source peer rank.
            comm (int | None): ncclComm_t pointer.
            stream (int | None): cudaStream_t pointer.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclRecv
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int
            return int(fn(recvbuff, count, datatype, peer, comm or 0, stream or 0))
        except Exception:
            return -1

    def get_unique_id(self) -> bytes:
        """Generate a new NCCL unique communicator ID.

        Returns:
            bytes: 128-byte raw ncclUniqueId buffer.

        Raises:
            RuntimeError: If ncclGetUniqueId fails.
        """
        self._check_available()
        uid: _NcclUniqueId = _NcclUniqueId()
        fn = self.nccl_lib.ncclGetUniqueId
        fn.argtypes = [ctypes.POINTER(_NcclUniqueId)]
        fn.restype = ctypes.c_int
        status: int = int(fn(ctypes.byref(uid)))
        if status != 0:
            raise RuntimeError(f"ncclGetUniqueId failed with status {status}")
        return bytes(uid.internal)

    def init_rank(self, nranks: int, comm_id: bytes, rank: int) -> int:
        """Initialize a new NCCL communicator for a rank.

        Args:
            nranks (int): Total number of participating ranks.
            comm_id (bytes): Unique ID generated by root via get_unique_id().
            rank (int): Rank index of this process.

        Returns:
            int: Pointer value to ncclComm_t communicator.

        Raises:
            RuntimeError: If ncclCommInitRank fails.
        """
        self._check_available()
        uid: _NcclUniqueId = _NcclUniqueId()
        ctypes.memmove(ctypes.byref(uid), comm_id[:128], min(128, len(comm_id)))

        comm = ctypes.c_void_p()
        fn = self.nccl_lib.ncclCommInitRank
        fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _NcclUniqueId, ctypes.c_int]
        fn.restype = ctypes.c_int
        status: int = int(fn(ctypes.byref(comm), nranks, uid, rank))
        if status != 0:
            raise RuntimeError(f"ncclCommInitRank failed with status {status}")
        return int(comm.value or 0)

    def comm_destroy(self, comm: int) -> int:
        """Destroy an allocated NCCL communicator.

        Args:
            comm (int): Pointer value to ncclComm_t.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclCommDestroy
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
            return int(fn(ctypes.c_void_p(comm)))
        except Exception:
            return -1

    def group_start(self) -> int:
        """Begin a grouped collective sequence.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclGroupStart
            fn.restype = ctypes.c_int
            return int(fn())
        except Exception:
            return -1

    def group_end(self) -> int:
        """End and execute a grouped collective sequence.

        Returns:
            int: ncclResult_t status.
        """
        self._check_available()
        try:
            fn = self.nccl_lib.ncclGroupEnd
            fn.restype = ctypes.c_int
            return int(fn())
        except Exception:
            return -1

    def stream_synchronize(self, stream: int | None = None) -> int:
        """Synchronize CUDA compute stream.

        Args:
            stream (int | None): CUDA stream pointer to synchronize.

        Returns:
            int: Status code (0 = success).
        """
        if stream is None:
            return 0
        if self.cuda_rt is not None and hasattr(self.cuda_rt, "cudaStreamSynchronize"):
            try:
                fn = self.cuda_rt.cudaStreamSynchronize
                fn.argtypes = [ctypes.c_void_p]
                fn.restype = ctypes.c_int
                return int(fn(ctypes.c_void_p(stream)))
            except Exception:
                return -1
        return 0
