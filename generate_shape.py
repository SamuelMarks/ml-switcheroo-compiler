ops = {
    "reshape": (
        "(input: Tensor, shape: Sequence[int])",
        "np.reshape(input.data, shape)",
    ),
    "flatten": (
        "(input: Tensor, start_dim: int = 0, end_dim: int = -1)",
        "np.reshape(input.data, -1)",
    ),  # simplification
    "squeeze": (
        "(input: Tensor, dim: Optional[Union[int, Sequence[int]]] = None)",
        "np.squeeze(input.data, axis=dim)",
    ),
    "unsqueeze": ("(input: Tensor, dim: int)", "np.expand_dims(input.data, axis=dim)"),
    "expand": (
        "(input: Tensor, size: Sequence[int])",
        "np.broadcast_to(input.data, size)",
    ),
    "broadcast_to": (
        "(input: Tensor, size: Sequence[int])",
        "np.broadcast_to(input.data, size)",
    ),
    "transpose": (
        "(input: Tensor, dim0: int, dim1: int)",
        "np.swapaxes(input.data, dim0, dim1)",
    ),
    "permute": (
        "(input: Tensor, dims: Sequence[int])",
        "np.transpose(input.data, dims)",
    ),
    "swapaxes": (
        "(input: Tensor, axis1: int, axis2: int)",
        "np.swapaxes(input.data, axis1, axis2)",
    ),
    "moveaxis": (
        "(input: Tensor, source: Union[int, Sequence[int]], destination: Union[int, Sequence[int]])",
        "np.moveaxis(input.data, source, destination)",
    ),
    "roll": (
        "(input: Tensor, shifts: Union[int, Sequence[int]], dims: Optional[Union[int, Sequence[int]]] = None)",
        "np.roll(input.data, shifts, axis=dims)",
    ),
    "slice": (
        "(input: Tensor, dim: int, start: Optional[int] = None, end: Optional[int] = None, step: int = 1)",
        "None # special",
    ),
    "dynamic_slice": (
        "(input: Tensor, start_indices: Sequence[Tensor], slice_sizes: Sequence[int])",
        "None",
    ),
    "update_slice": (
        "(input: Tensor, update: Tensor, start_indices: Sequence[int])",
        "None",
    ),
    "strided_slice": (
        "(input: Tensor, begin: Sequence[int], end: Sequence[int], strides: Sequence[int])",
        "None",
    ),
    "concatenate": (
        "(tensors: Sequence[Tensor], dim: int = 0)",
        "np.concatenate([t.data for t in tensors], axis=dim)",
    ),
    "stack": (
        "(tensors: Sequence[Tensor], dim: int = 0)",
        "np.stack([t.data for t in tensors], axis=dim)",
    ),
    "split": (
        "(input: Tensor, split_size_or_sections: Union[int, Sequence[int]], dim: int = 0) -> Sequence[Tensor]",
        "np.split(input.data, split_size_or_sections, axis=dim)",
    ),
    "unstack": (
        "(input: Tensor, dim: int = 0) -> Sequence[Tensor]",
        "np.unstack(input.data, axis=dim) if hasattr(np, 'unstack') else np.moveaxis(input.data, dim, 0)",
    ),
    "tile": ("(input: Tensor, reps: Sequence[int])", "np.tile(input.data, reps)"),
    "repeat": (
        "(input: Tensor, repeats: Union[int, Sequence[int]], dim: Optional[int] = None)",
        "np.repeat(input.data, repeats, axis=dim)",
    ),
    "gather": (
        "(input: Tensor, dim: int, index: Tensor)",
        "np.take_along_axis(input.data, index.data, axis=dim)",
    ),
    "gather_nd": ("(input: Tensor, indices: Tensor)", "None"),
    "scatter": ("(input: Tensor, dim: int, index: Tensor, src: Tensor)", "None"),
    "scatter_nd": ("(indices: Tensor, updates: Tensor, shape: Sequence[int])", "None"),
    "scatter_add": ("(input: Tensor, dim: int, index: Tensor, src: Tensor)", "None"),
    "take": ("(input: Tensor, indices: Tensor)", "np.take(input.data, indices.data)"),
    "where": (
        "(condition: Tensor, input: Tensor, other: Tensor)",
        "np.where(condition.data, input.data, other.data)",
    ),
    "triu": ("(input: Tensor, diagonal: int = 0)", "np.triu(input.data, k=diagonal)"),
    "tril": ("(input: Tensor, diagonal: int = 0)", "np.tril(input.data, k=diagonal)"),
    "meshgrid": (
        "(*tensors: Tensor, indexing: str = 'ij') -> Sequence[Tensor]",
        "np.meshgrid(*[t.data for t in tensors], indexing=indexing)",
    ),
}

content = """\"\"\"Shape, Memory, and Movement Ops.\"\"\"
import uuid
from typing import Sequence, Union, Optional, Tuple
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.core.errors import UnimplementedMathError
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode

def _emit_shape_node(op_type: str, inputs: Sequence[Tensor], attrs: dict, out_shape: tuple, out_dtype: DType) -> Tensor:
    if not _tracer.is_tracing:
        raise RuntimeError(f"Cannot emit {op_type} node outside of a tracing context.")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    device = inputs[0].device if len(inputs) > 0 else config.default_device
    return Tensor(data=proxy, shape=out_shape, dtype=out_dtype, device=device)

"""

for op, (sig, np_impl) in ops.items():
    if "->" not in sig:
        sig += " -> Tensor"
    op_type = "".join(x.title() for x in op.split("_"))

    if "Sequence[Tensor]" in sig and op not in ["meshgrid"]:
        inputs_code = (
            "inputs = list(tensors)" if "tensors" in sig else "inputs = [input]"
        )
    elif op == "meshgrid":
        inputs_code = "inputs = list(tensors)"
    elif op == "where":
        inputs_code = "inputs = [condition, input, other]"
    elif op == "scatter_nd":
        inputs_code = "inputs = [indices, updates]"
    elif op in ["dynamic_slice"]:
        inputs_code = "inputs = [input] + list(start_indices)"
    elif op in ["update_slice", "scatter", "scatter_add"]:
        inputs_code = (
            "inputs = [input, update]"
            if op == "update_slice"
            else "inputs = [input, index, src]"
        )
    elif op in ["gather", "gather_nd", "take"]:
        inputs_code = (
            "inputs = [input, index]" if op == "gather" else "inputs = [input, indices]"
        )
    else:
        inputs_code = "inputs = [input]"

    if np_impl == "None":
        eager_body = f"raise UnimplementedMathError('No direct numpy for {op}')"
    elif op == "slice":
        eager_body = "sl = [slice(None)] * len(input.shape)\n        sl[dim] = slice(start, end, step)\n        data = input.data[tuple(sl)]\n        return Tensor(data, data.shape, input.dtype, input.device)"
    elif "Sequence[Tensor]" in sig or op == "meshgrid":
        if op in ["split", "unstack"]:
            eager_body = f"datas = {np_impl}\n        return tuple(Tensor(d, d.shape, input.dtype, input.device) for d in datas)"
        elif op == "meshgrid":
            eager_body = f"datas = {np_impl}\n        return tuple(Tensor(d, d.shape, tensors[0].dtype, tensors[0].device) for d in datas)"
        else:
            eager_body = f"data = {np_impl}\n        return Tensor(data, data.shape, tensors[0].dtype, tensors[0].device)"
    else:
        eager_body = f"data = {np_impl}\n        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)"

    content += f"""
def {op}{sig}:
    \"\"\"{op}\"\"\"
    if config.eager_mode:
        {eager_body}
    else:
        {inputs_code}
        # shape calculation placeholder
        out_shape = inputs[0].shape if len(inputs) > 0 else ()
        if "{op}" in ["split", "unstack"]:
            return ( _emit_shape_node('{op_type}', inputs, {{}}, out_shape, inputs[0].dtype), )
        elif "{op}" == "meshgrid":
            return tuple(_emit_shape_node('{op_type}', inputs, {{}}, out_shape, inputs[0].dtype) for _ in inputs)
        else:
            return _emit_shape_node('{op_type}', inputs, {{}}, out_shape, inputs[0].dtype if len(inputs) > 0 else DType.Float32)
"""

with open("src/ml_switcheroo/ops/shape.py", "w") as f:
    f.write(content)
