ops = [
    "matmul",
    "dot",
    "tensordot",
    "vdot",
    "inner",
    "outer",
    "einsum",
    "cholesky",
    "svd",
    "qr",
    "inv",
    "pinv",
    "det",
    "slogdet",
    "eigh",
    "eigvalsh",
    "matrix_power",
]

content = """\"\"\"Linear Algebra Operations.\"\"\"
import uuid
from typing import Sequence, Union, Optional, Tuple
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode

def _emit_linalg_node(op_type: str, inputs: Sequence[Tensor], attrs: dict, out_shapes: Sequence[Sequence[int]], out_dtypes: Sequence[DType]) -> Union[Tensor, Tuple[Tensor, ...]]:
    \"\"\"Emit a linear algebra node to the IR graph.\"\"\"
    if config.eager_mode:
        raise RuntimeError("Cannot emit node in eager mode.")
    if not _tracer.is_tracing:
        raise RuntimeError(f"Cannot emit {op_type} node outside of a tracing context.")
        
    out_ids = [str(uuid.uuid4()) for _ in out_shapes]
    # Simple broadcast fallback for shape_metadata
    shape_meta = tuple(out_shapes[0]) if len(out_shapes) == 1 else tuple(tuple(s) for s in out_shapes)
    
    node = LogicalNode(
        id=out_ids[0],
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=shape_meta,
    )
    _tracer.add_node(node)
    
    tensors = []
    for i, (out_id, shape, dtype) in enumerate(zip(out_ids, out_shapes, out_dtypes)):
        proxy = ProxyTensor(id=out_id, shape=tuple(shape), dtype=dtype.value)
        tensors.append(Tensor(data=proxy, shape=tuple(shape), dtype=dtype, device=inputs[0].device))
        
    return tensors[0] if len(tensors) == 1 else tuple(tensors)

"""

for op in ops:
    if op == "tensordot":
        sig = "(a: Tensor, b: Tensor, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> Tensor:"
        eager_body = "data = np.tensordot(a.data, b.data, axes=axes)\n        return Tensor(data, data.shape, a.dtype, a.device)"
        trace_body = "return _emit_linalg_node('Tensordot', [a, b], {'axes': axes}, [()], [a.dtype])"
    elif op == "einsum":
        sig = "(equation: str, *operands: Tensor) -> Tensor:"
        eager_body = "data = np.einsum(equation, *[op.data for op in operands])\n        return Tensor(data, data.shape, operands[0].dtype, operands[0].device)"
        trace_body = "return _emit_linalg_node('Einsum', operands, {'equation': equation}, [()], [operands[0].dtype])"
    elif op in ["svd", "qr", "slogdet", "eigh", "eigvalsh"]:
        if op == "svd":
            sig = "(input: Tensor, full_matrices: bool = True, compute_uv: bool = True) -> Tuple[Tensor, Tensor, Tensor]:"
            eager_body = "u, s, vh = np.linalg.svd(input.data, full_matrices=full_matrices, compute_uv=compute_uv)\n        return (Tensor(u, u.shape, input.dtype, input.device), Tensor(s, s.shape, input.dtype, input.device), Tensor(vh, vh.shape, input.dtype, input.device))"
            trace_body = "return _emit_linalg_node('Svd', [input], {'full_matrices': full_matrices, 'compute_uv': compute_uv}, [(), (), ()], [input.dtype]*3)"
        elif op == "qr":
            sig = "(input: Tensor, mode: str = 'reduced') -> Tuple[Tensor, Tensor]:"
            eager_body = "q, r = np.linalg.qr(input.data, mode=mode)\n        return (Tensor(q, q.shape, input.dtype, input.device), Tensor(r, r.shape, input.dtype, input.device))"
            trace_body = "return _emit_linalg_node('Qr', [input], {'mode': mode}, [(), ()], [input.dtype]*2)"
        elif op == "slogdet":
            sig = "(input: Tensor) -> Tuple[Tensor, Tensor]:"
            eager_body = "sign, logdet = np.linalg.slogdet(input.data)\n        return (Tensor(np.array(sign), np.array(sign).shape, input.dtype, input.device), Tensor(np.array(logdet), np.array(logdet).shape, input.dtype, input.device))"
            trace_body = "return _emit_linalg_node('Slogdet', [input], {}, [(), ()], [input.dtype]*2)"
        elif op == "eigh":
            sig = "(input: Tensor, UPLO: str = 'L') -> Tuple[Tensor, Tensor]:"
            eager_body = "w, v = np.linalg.eigh(input.data, UPLO=UPLO)\n        return (Tensor(w, w.shape, input.dtype, input.device), Tensor(v, v.shape, input.dtype, input.device))"
            trace_body = "return _emit_linalg_node('Eigh', [input], {'UPLO': UPLO}, [(), ()], [input.dtype]*2)"
        else:
            sig = "(input: Tensor, UPLO: str = 'L') -> Tensor:"
            eager_body = "data = np.linalg.eigvalsh(input.data, UPLO=UPLO)\n        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)"
            trace_body = "return _emit_linalg_node('Eigvalsh', [input], {'UPLO': UPLO}, [()], [input.dtype])"
    elif op == "pinv":
        sig = "(input: Tensor, rcond: float = 1e-15) -> Tensor:"
        eager_body = "data = np.linalg.pinv(input.data, rcond=rcond)\n        return Tensor(data, data.shape, input.dtype, input.device)"
        trace_body = "return _emit_linalg_node('Pinv', [input], {'rcond': rcond}, [()], [input.dtype])"
    elif op == "matrix_power":
        sig = "(input: Tensor, n: int) -> Tensor:"
        eager_body = "data = np.linalg.matrix_power(input.data, n)\n        return Tensor(data, data.shape, input.dtype, input.device)"
        trace_body = "return _emit_linalg_node('MatrixPower', [input], {'n': n}, [()], [input.dtype])"
    elif op in ["matmul", "dot", "vdot", "inner", "outer"]:
        sig = "(input: Tensor, other: Tensor) -> Tensor:"
        eager_body = f"data = np.{op}(input.data, other.data)\n        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)"
        trace_body = f"return _emit_linalg_node('{op.capitalize()}', [input, other], {{}}, [()], [input.dtype])"
    else:  # cholesky, inv, det
        sig = "(input: Tensor) -> Tensor:"
        eager_body = f"data = np.linalg.{op}(input.data)\n        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)"
        trace_body = f"return _emit_linalg_node('{op.capitalize()}', [input], {{}}, [()], [input.dtype])"

    content += f"""
def {op}{sig}
    \"\"\"Computes {op}.\"\"\"
    if config.eager_mode:
        {eager_body}
    else:
        {trace_body}
"""

with open("src/ml_switcheroo/ops/linalg.py", "w") as f:
    f.write(content)
