"""Tiling, element repetition, triangular masking, and coordinate meshgrid operations."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import (
    coerce_int,
    coerce_operand_list,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue, extract_shape, extract_symbolic_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import SymNode
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _compute_tile_shape(
    in_shape: tuple[int | None, ...],
    reps: tuple[int | None, ...],
) -> tuple[int | None, ...]:
    """Compute the tiled output shape from aligned input dimensions and repetition factors.

    Args:
        in_shape (tuple[int | None, ...]): Input tensor shape with optional unknown dimensions.
        reps (tuple[int | None, ...]): Per-axis repetition counts.

    Returns:
        tuple[int | None, ...]: Broadcast-aligned tiled output shape.
    """
    d = len(reps)
    c = len(in_shape)
    if c < d:
        in_shape = (1,) * (d - c) + in_shape
    elif c > d:
        reps = (1,) * (c - d) + reps
    return tuple(s * r if s is not None and r is not None else None for s, r in zip(in_shape, reps))


@register_op("Tile")
class Tile(OpDef):
    """Schema definition constructing an array by repeating an operand `reps` times."""

    op_name = "Tile"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for a `Tile` operation.

        Args:
            *args (OpValue): Input array followed by repetition counts `reps`.
            **kwargs (OpValue): Unused keyword operation attributes.

        Returns:
            OpValue: Inferred tiled output shape tuple.
        """
        del kwargs
        a = args[0] if len(args) > 0 else None
        reps = args[1] if len(args) > 1 else None
        if reps is None:
            return ()
        in_shape = tuple(d if isinstance(d, int) else None for d in extract_symbolic_shape(a))
        if isinstance(reps, int):
            reps_tuple: tuple[int | None, ...] = (reps,)
        else:
            reps_tuple = tuple(
                r if isinstance(r, int) and not isinstance(r, bool) else None for r in coerce_operand_list(reps)
            )
        return _compute_tile_shape(in_shape, reps_tuple)


def tile(input: Tensor, reps: Sequence[int]) -> OpValue:
    """Construct a tensor by repeating `input` according to per-dimension `reps`.

    Args:
        input (Tensor): Source tensor to tile.
        reps (Sequence[int]): Number of repetitions along each axis.

    Returns:
        OpValue: Tiled tensor matching repeated dimensions.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    if config.eager_mode:
        backend = _pt_mod.get_active_backend()
        data = backend.execute_op("Tile", (input.data if type(input).__name__ == "Tensor" else input), reps)
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), input.dtype, input.device))
    inputs = [input]
    raw_in_shape = extract_shape(inputs[0])
    in_shape = list(raw_in_shape) if raw_in_shape else [1]
    rep_list = list(reps) if isinstance(reps, (list, tuple)) else [int(coerce_int(reps))]
    while len(in_shape) < len(rep_list):
        in_shape.insert(0, 1)
    while len(rep_list) < len(in_shape):
        rep_list.insert(0, 1)
    out_shape = tuple(d * int(r) for d, r in zip(in_shape, rep_list))
    return _pt_mod._emit_shape_node(
        "Tile",
        inputs,
        {"reps": reps},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def _repeat_infer_no_axis(
    in_shape: tuple[int | None, ...],
    repeats: OpValue,
) -> tuple[int | SymNode | None, ...]:
    """Infer flattened 1D output shape for `Repeat` when `axis` is `None`.

    Args:
        in_shape (tuple[int | None, ...]): Input tensor dimensions.
        repeats (OpValue): Scalar or per-element repetition counts.

    Returns:
        tuple[int | SymNode | None, ...]: Single-element 1D shape tuple (concrete or symbolic).
    """
    from ml_switcheroo_compiler.ir.shape_system import SymVar

    size: int | None = 1
    for s in in_shape:
        if s is None:
            size = None
            break
        size = int(coerce_int(size)) * s
    if isinstance(repeats, int) and size is not None:
        return (size * repeats,)
    if isinstance(repeats, (list, tuple)):
        return (sum(int(coerce_int(r)) for r in repeats),)
    return (SymVar("repeat_size"),)


@register_op("Repeat")
class Repeat(OpDef):
    """Schema definition repeating elements of an array along an optional axis."""

    op_name = "Repeat"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for a `Repeat` operation.

        Args:
            *args (OpValue): Input tensor followed by `repeats` specification.
            **kwargs (OpValue): Keyword attributes including optional `axis`.

        Returns:
            OpValue: Inferred repeated output shape tuple.
        """
        a = args[0] if len(args) > 0 else None
        repeats = args[1] if len(args) > 1 else None
        axis = kwargs.get("axis", None)
        in_shape = tuple(d if isinstance(d, int) else None for d in extract_symbolic_shape(a))
        if axis is None:
            return _repeat_infer_no_axis(in_shape, repeats)
        out_shape: list[int | None] = list(in_shape)
        ax = int(coerce_int(axis))
        if isinstance(repeats, int):
            cur_dim = out_shape[ax]
            out_shape[ax] = cur_dim * repeats if cur_dim is not None else None
        elif isinstance(repeats, (list, tuple)):
            out_shape[ax] = int(sum(int(coerce_int(r)) for r in repeats))
        else:
            out_shape[ax] = None
        return tuple(out_shape)


def repeat(
    input: Tensor,
    repeats: int | Sequence[int],
    axis: int | None = None,
) -> OpValue:
    """Repeat elements of `input` along `axis` (or flattened when `axis` is `None`).

    Args:
        input (Tensor): Source tensor whose elements are repeated.
        repeats (int | Sequence[int]): Uniform or per-element repetition counts.
        axis (int | None): Optional axis along which values are repeated.

    Returns:
        OpValue: Tensor containing repeated elements.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    if config.eager_mode:
        backend = _pt_mod.get_active_backend()
        data = backend.execute_op(
            "Repeat", (input.data if type(input).__name__ == "Tensor" else input), repeats, axis=axis
        )
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), input.dtype, input.device))
    inputs = [input]
    base_shape = extract_shape(inputs[0])
    if axis is None:
        num_elems = 1
        for dim in base_shape:
            num_elems *= dim
        if isinstance(repeats, int):
            out_shape: tuple[int, ...] = (num_elems * repeats,)
        elif isinstance(repeats, (list, tuple)):
            out_shape = (sum(int(r) for r in repeats),)
        else:
            out_shape = (num_elems,)
    else:
        rank = len(base_shape)
        norm_axis = axis if axis >= 0 else axis + rank
        if isinstance(repeats, int):
            out_shape = tuple(d * repeats if i == norm_axis else d for i, d in enumerate(base_shape))
        elif isinstance(repeats, (list, tuple)):
            out_shape = tuple(sum(int(r) for r in repeats) if i == norm_axis else d for i, d in enumerate(base_shape))
        else:
            out_shape = base_shape
    return _pt_mod._emit_shape_node(
        "Repeat",
        inputs,
        {"repeats": repeats, "axis": axis},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def triu(input: Tensor, diagonal: int = 0) -> OpValue:
    """Return the upper triangular portion of a matrix or batch of matrices.

    Args:
        input (Tensor): Input matrix or batch of 2D matrices.
        diagonal (int): Diagonal offset below which entries are zeroed.

    Returns:
        OpValue: Upper triangular tensor preserving `input` shape.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    if config.eager_mode:
        backend = _pt_mod.get_active_backend()
        data = backend.execute_op("Triu", (input.data if type(input).__name__ == "Tensor" else input), k=diagonal)
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), input.dtype, input.device))
    inputs = [input]
    out_shape = inputs[0].shape
    return _pt_mod._emit_shape_node(
        "Triu",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def tril(input: Tensor, diagonal: int = 0) -> OpValue:
    """Return the lower triangular portion of a matrix or batch of matrices.

    Args:
        input (Tensor): Input matrix or batch of 2D matrices.
        diagonal (int): Diagonal offset above which entries are zeroed.

    Returns:
        OpValue: Lower triangular tensor preserving `input` shape.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    if config.eager_mode:
        backend = _pt_mod.get_active_backend()
        data = backend.execute_op("Tril", (input.data if type(input).__name__ == "Tensor" else input), k=diagonal)
        return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), input.dtype, input.device))
    inputs = [input]
    out_shape = inputs[0].shape
    return _pt_mod._emit_shape_node(
        "Tril",
        inputs,
        {},
        out_shape,
        inputs[0].dtype if len(inputs) > 0 else DType.Float32,
    )


def _compute_meshgrid_shape(inputs: list[Tensor], indexing: str) -> tuple[int, ...]:
    """Compute output grid dimensions from 1D coordinate tensors and indexing mode.

    Args:
        inputs (list[Tensor]): Coordinate vector tensors defining each grid axis.
        indexing (str): Grid indexing convention (`'ij'` matrix or `'xy'` Cartesian).

    Returns:
        tuple[int, ...]: Resulting grid tensor shape.
    """
    if not inputs:
        return ()
    out_shape = tuple(extract_shape(t)[0] if t.shape else 1 for t in inputs)
    if indexing == "xy" and len(inputs) >= 2:
        return (out_shape[1], out_shape[0]) + out_shape[2:]
    return out_shape


def meshgrid(*tensors: Tensor, indexing: str = "ij") -> Sequence[Tensor]:
    """Construct N-D coordinate matrices from 1D coordinate vector tensors.

    Args:
        *tensors (Tensor): 1D coordinate tensors representing grid axes.
        indexing (str): Grid indexing convention (`'ij'` or `'xy'`).

    Returns:
        Sequence[Tensor]: Tuple of coordinate grid tensors, one per input vector.
    """
    import ml_switcheroo_compiler.ops.shape.pad_and_tile as _pt_mod

    if config.eager_mode:
        backend = _pt_mod.get_active_backend()
        datas = backend.execute_op(
            "Meshgrid", *[(t.data if type(t).__name__ == "Tensor" else t) for t in tensors], indexing=indexing
        )
        return tuple(
            Tensor(d, TensorConfig(extract_shape(d), tensors[0].dtype, tensors[0].device))
            for d in coerce_operand_list(datas)
        )
    inputs = list(tensors)
    out_shape = _compute_meshgrid_shape(inputs, indexing)
    dtype = inputs[0].dtype if inputs else DType.Float32
    return tuple(_pt_mod._emit_shape_node("Meshgrid", inputs, {"indexing": indexing}, out_shape, dtype) for _ in inputs)
