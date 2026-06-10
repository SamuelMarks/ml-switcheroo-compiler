"""Linear Algebra Operations."""

import uuid
from typing import Union
from collections.abc import Sequence
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def _emit_linalg_node(
    op_type: str,
    inputs: Sequence[Tensor],
    attrs: dict,
    out_shapes: Sequence[Sequence[int]],
    out_dtypes: Sequence[DType],
) -> Union[Tensor, tuple[Tensor, ...]]:
    """Emit a linear algebra node to the IR graph."""
    if not _tracer.is_tracing:
        raise RuntimeError(f"Cannot emit {op_type} node outside of a tracing context.")

    out_ids = [str(uuid.uuid4()) for _ in out_shapes]
    # Simple broadcast fallback for shape_metadata
    shape_meta = (
        tuple(out_shapes[0])
        if len(out_shapes) == 1
        else tuple(tuple(s) for s in out_shapes)
    )

    node = LogicalNode(
        id=out_ids[0],
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=shape_meta,
    )
    _tracer.add_node(node)

    tensors = []
    for _i, (out_id, shape, dtype) in enumerate(zip(out_ids, out_shapes, out_dtypes)):
        proxy = ProxyTensor(id=out_id, shape=tuple(shape), dtype=dtype.value)
        tensors.append(
            Tensor(data=proxy, shape=tuple(shape), dtype=dtype, device=inputs[0].device)
        )

    return tensors[0] if len(tensors) == 1 else tuple(tensors)


def matmul(input: Tensor, other: Tensor) -> Tensor:
    """Computes matmul."""
    if config.eager_mode:
        data = np.matmul(input.data, other.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Matmul", [input, other], {}, [()], [input.dtype])


def dot(input: Tensor, other: Tensor) -> Tensor:
    """Computes dot."""
    if config.eager_mode:
        data = np.dot(input.data, other.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Dot", [input, other], {}, [()], [input.dtype])


def tensordot(
    a: Tensor, b: Tensor, axes: Union[int, tuple[Sequence[int], Sequence[int]]] = 2
) -> Tensor:
    """Computes tensordot."""
    if config.eager_mode:
        data = np.tensordot(a.data, b.data, axes=axes)
        return Tensor(data, data.shape, a.dtype, a.device)
    else:
        return _emit_linalg_node("Tensordot", [a, b], {"axes": axes}, [()], [a.dtype])


def vdot(input: Tensor, other: Tensor) -> Tensor:
    """Computes vdot."""
    if config.eager_mode:
        data = np.vdot(input.data, other.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Vdot", [input, other], {}, [()], [input.dtype])


def inner(input: Tensor, other: Tensor) -> Tensor:
    """Computes inner."""
    if config.eager_mode:
        data = np.inner(input.data, other.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Inner", [input, other], {}, [()], [input.dtype])


def outer(input: Tensor, other: Tensor) -> Tensor:
    """Computes outer."""
    if config.eager_mode:
        data = np.outer(input.data, other.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Outer", [input, other], {}, [()], [input.dtype])


def einsum(equation: str, *operands: Tensor) -> Tensor:
    """Computes einsum."""
    if config.eager_mode:
        data = np.einsum(equation, *[op.data for op in operands])
        return Tensor(data, data.shape, operands[0].dtype, operands[0].device)
    else:
        return _emit_linalg_node(
            "Einsum", operands, {"equation": equation}, [()], [operands[0].dtype]
        )


def cholesky(input: Tensor) -> Tensor:
    """Computes cholesky."""
    if config.eager_mode:
        data = np.linalg.cholesky(input.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Cholesky", [input], {}, [()], [input.dtype])


def svd(
    input: Tensor, full_matrices: bool = True, compute_uv: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes svd."""
    if config.eager_mode:
        u, s, vh = np.linalg.svd(
            input.data, full_matrices=full_matrices, compute_uv=compute_uv
        )
        return (
            Tensor(u, u.shape, input.dtype, input.device),
            Tensor(s, s.shape, input.dtype, input.device),
            Tensor(vh, vh.shape, input.dtype, input.device),
        )
    else:
        return _emit_linalg_node(
            "Svd",
            [input],
            {"full_matrices": full_matrices, "compute_uv": compute_uv},
            [(), (), ()],
            [input.dtype] * 3,
        )


def qr(input: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    """Computes qr."""
    if config.eager_mode:
        q, r = np.linalg.qr(input.data, mode=mode)
        return (
            Tensor(q, q.shape, input.dtype, input.device),
            Tensor(r, r.shape, input.dtype, input.device),
        )
    else:
        return _emit_linalg_node(
            "Qr", [input], {"mode": mode}, [(), ()], [input.dtype] * 2
        )


def inv(input: Tensor) -> Tensor:
    """Computes inv."""
    if config.eager_mode:
        data = np.linalg.inv(input.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Inv", [input], {}, [()], [input.dtype])


def pinv(input: Tensor, rcond: float = 1e-15) -> Tensor:
    """Computes pinv."""
    if config.eager_mode:
        data = np.linalg.pinv(input.data, rcond=rcond)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Pinv", [input], {"rcond": rcond}, [()], [input.dtype])


def det(input: Tensor) -> Tensor:
    """Computes det."""
    if config.eager_mode:
        data = np.linalg.det(input.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("Det", [input], {}, [()], [input.dtype])


def slogdet(input: Tensor) -> tuple[Tensor, Tensor]:
    """Computes slogdet."""
    if config.eager_mode:
        sign, logdet = np.linalg.slogdet(input.data)
        return (
            Tensor(np.array(sign), np.array(sign).shape, input.dtype, input.device),
            Tensor(np.array(logdet), np.array(logdet).shape, input.dtype, input.device),
        )
    else:
        return _emit_linalg_node("Slogdet", [input], {}, [(), ()], [input.dtype] * 2)


def eigh(input: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Computes eigh."""
    if config.eager_mode:
        w, v = np.linalg.eigh(input.data, UPLO=UPLO)
        return (
            Tensor(w, w.shape, input.dtype, input.device),
            Tensor(v, v.shape, input.dtype, input.device),
        )
    else:
        return _emit_linalg_node(
            "Eigh", [input], {"UPLO": UPLO}, [(), ()], [input.dtype] * 2
        )


def eigvalsh(input: Tensor, UPLO: str = "L") -> Tensor:
    """Computes eigvalsh."""
    if config.eager_mode:
        data = np.linalg.eigvalsh(input.data, UPLO=UPLO)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_linalg_node(
            "Eigvalsh", [input], {"UPLO": UPLO}, [()], [input.dtype]
        )


def matrix_power(input: Tensor, n: int) -> Tensor:
    """Computes matrix_power."""
    if config.eager_mode:
        data = np.linalg.matrix_power(input.data, n)
        return Tensor(data, data.shape, input.dtype, input.device)
    else:
        return _emit_linalg_node("MatrixPower", [input], {"n": n}, [()], [input.dtype])
