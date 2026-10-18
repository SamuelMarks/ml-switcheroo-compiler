"""Tensor creation, windowing, and index generation operation definitions."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.constants import MIN_CONV_RANK
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ops.base import OpArgValue, OpDef, register_op


def _coerce_shape(val: OpArgValue) -> tuple[int, ...] | None:
    """Coerce a raw shape argument into a tuple of integers or None.

    Args:
        val (OpArgValue): Raw shape specification (tuple, list, int, or None).

    Returns:
        tuple[int, ...] | None: Coerced shape tuple or None.
    """
    return (val,) if isinstance(val, int) else (tuple(int(x) for x in val if isinstance(x, int)) if isinstance(val, (tuple, list)) else None)


class CreationOp(OpDef):
    """Base class for tensor creation operations that produce a tensor of a target shape."""

    op_name: str = ""

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer the output shape of the tensor creation operation.

        Args:
            *args (OpArgValue): Positional arguments containing the target shape.
            **kwargs (OpArgValue): Keyword arguments containing optional shape specification.

        Returns:
            tuple[int, ...] | None: Inferred output tensor shape.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        return _coerce_shape(shape)


@register_op("Zeros")
class Zeros(CreationOp):
    """Create a tensor of a specified shape filled with zeros."""

    op_name = "Zeros"


@register_op("Ones")
class Ones(CreationOp):
    """Create a tensor of a specified shape filled with ones."""

    op_name = "Ones"


@register_op("Full")
class Full(CreationOp):
    """Create a tensor of a specified shape filled with a constant scalar value."""

    op_name = "Full"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer the output shape of the `Full` operation.

        Args:
            *args (OpArgValue): Positional arguments containing shape and fill_value.
            **kwargs (OpArgValue): Keyword arguments containing optional shape specification.

        Returns:
            tuple[int, ...] | None: Inferred output tensor shape.
        """
        shape = args[0] if len(args) > 0 else kwargs.get("shape")
        return _coerce_shape(shape)


@register_op("Arange")
class Arange(OpDef):
    """Create a 1-D tensor containing evenly spaced values within a half-open interval."""

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Return None because `Arange` output length depends on runtime start/stop/step values.

        Args:
            *args (OpArgValue): Positional arguments (`start`, `stop`, `step`).
            **kwargs (OpArgValue): Keyword arguments.

        Returns:
            tuple[int, ...] | None: None.
        """
        del args, kwargs
        return None


@register_op("Rand")
class Rand(CreationOp):
    """Create a tensor with random samples from a uniform distribution over `[0, 1)`."""

    op_name = "Rand"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer the output shape of the `Rand` operation.

        Args:
            *args (OpArgValue): Dimensions or shape tuple.
            **kwargs (OpArgValue): Keyword arguments containing optional `size`.

        Returns:
            tuple[int, ...]: Inferred output shape tuple.
        """
        if "size" in kwargs:
            return _coerce_shape(kwargs["size"]) or ()
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return _coerce_shape(args[0]) or ()
        return _coerce_shape(args) or ()


@register_op("Randn")
class Randn(Rand):
    """Create a tensor with random samples from a standard normal distribution."""

    op_name = "Randn"


@register_op("Randint")
class Randint(CreationOp):
    """Create a tensor with random integers sampled uniformly from `[low, high)`."""

    op_name = "Randint"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer the output shape of the `Randint` operation.

        Args:
            *args (OpArgValue): Positional arguments (`low`, `high`, `size`).
            **kwargs (OpArgValue): Keyword arguments containing optional `size`.

        Returns:
            tuple[int, ...]: Inferred output shape tuple.
        """
        if "size" in kwargs:
            return _coerce_shape(kwargs["size"]) or ()
        if len(args) == MIN_CONV_RANK:
            return _coerce_shape(args[2]) or ()
        return ()


@register_op("ManualSeed")
class ManualSeed(OpDef):
    """Set the random number generator seed."""

    op_name = "ManualSeed"

    def infer_shape(self, seed: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer scalar `()` shape for `ManualSeed`.

        Args:
            seed (EagerTensorProtocol): Seed integer or tensor.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: Empty shape tuple `()`.
        """
        _ = (seed, kwargs)
        return ()


@register_op("ConstantOfShape")
class ConstantOfShape(OpDef):
    """Generate a tensor of a given shape filled with a constant value."""

    def infer_shape(self, *shapes: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape from the first shape argument.

        Args:
            *shapes (EagerTensorProtocol): Target shape tuple arguments.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: Inferred output shape.
        """
        _ = kwargs
        return shapes[0] if shapes else ()


@register_op("Range")
class Range(OpDef):
    """Generate a 1D range sequence tensor."""

    def infer_shape(self, *shapes: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Range`.

        Args:
            *shapes (EagerTensorProtocol): Input bound arguments.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: Inferred shape tuple.
        """
        _ = kwargs
        return shapes[0] if shapes else ()


@register_op("Blackman")
class Blackman(OpDef):
    """Return the Blackman window of length `M`."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D `(M,)` output shape for `Blackman`.

        Args:
            M (EagerTensorProtocol): Number of points in the output window.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: 1-tuple `(M,)`.
        """
        _ = kwargs
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Bartlett")
class Bartlett(OpDef):
    """Return the Bartlett triangular window of length `M`."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D `(M,)` output shape for `Bartlett`.

        Args:
            M (EagerTensorProtocol): Number of points in the output window.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: 1-tuple `(M,)`.
        """
        _ = kwargs
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Hamming")
class Hamming(OpDef):
    """Return the Hamming window of length `M`."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D `(M,)` output shape for `Hamming`.

        Args:
            M (EagerTensorProtocol): Number of points in the output window.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: 1-tuple `(M,)`.
        """
        _ = kwargs
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Hanning")
class Hanning(OpDef):
    """Return the Hanning (Hann) window of length `M`."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D `(M,)` output shape for `Hanning`.

        Args:
            M (EagerTensorProtocol): Number of points in the output window.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: 1-tuple `(M,)`.
        """
        _ = kwargs
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("Kaiser")
class Kaiser(OpDef):
    """Return the Kaiser window of length `M`."""

    def infer_shape(self, M: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D `(M,)` output shape for `Kaiser`.

        Args:
            M (EagerTensorProtocol): Number of points in the output window.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: 1-tuple `(M,)`.
        """
        _ = kwargs
        return (M,) if isinstance(M, int) else (M.item(),)


@register_op("TrilIndices")
class TrilIndices(OpDef):
    """Return the indices for the lower-triangle of an `(n, m)` array."""

    op_name = "TrilIndices"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for `TrilIndices`.

        Args:
            *args (OpArgValue): Positional dimension arguments.
            **kwargs (OpArgValue): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred shape tuple.
        """
        del kwargs
        return (_coerce_shape(args[0]) or ()) if args else ()


@register_op("TrilIndicesFrom")
class TrilIndicesFrom(OpDef):
    """Return the indices for the lower-triangle of a given 2D array."""

    op_name = "TrilIndicesFrom"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for `TrilIndicesFrom`.

        Args:
            *args (OpArgValue): Positional input array arguments.
            **kwargs (OpArgValue): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred shape tuple.
        """
        del kwargs
        return (_coerce_shape(args[0]) or ()) if args else ()


@register_op("TriuIndices")
class TriuIndices(OpDef):
    """Return the indices for the upper-triangle of an `(n, m)` array."""

    op_name = "TriuIndices"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for `TriuIndices`.

        Args:
            *args (OpArgValue): Positional dimension arguments.
            **kwargs (OpArgValue): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred shape tuple.
        """
        del kwargs
        return (_coerce_shape(args[0]) or ()) if args else ()


@register_op("TriuIndicesFrom")
class TriuIndicesFrom(OpDef):
    """Return the indices for the upper-triangle of a given 2D array."""

    op_name = "TriuIndicesFrom"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer shape for `TriuIndicesFrom`.

        Args:
            *args (OpArgValue): Positional input array arguments.
            **kwargs (OpArgValue): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred shape tuple.
        """
        del kwargs
        return (_coerce_shape(args[0]) or ()) if args else ()


@register_op("Logspace")
class Logspace(CreationOp):
    """Return numbers spaced evenly on a log scale."""

    op_name = "Logspace"

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...]:
        """Infer 1D `(num,)` output shape for `Logspace`.

        Args:
            *args (OpArgValue): Positional arguments (`start`, `stop`, `num`).
            **kwargs (OpArgValue): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred 1-D output shape.
        """
        num_val = kwargs.get("num", args[2] if len(args) > 2 and isinstance(args[2], int) else 50)
        return (int(num_val) if isinstance(num_val, int) else 50,)


@register_op("Frombuffer")
class Frombuffer(OpDef):
    """Interpret a buffer as a 1-dimensional array."""

    def infer_shape(self, *args: OpArgValue, **kwargs: OpArgValue) -> tuple[int, ...] | None:
        """Infer 1D `(count,)` shape for `Frombuffer` if `count` is specified.

        Args:
            *args (OpArgValue): Positional buffer arguments.
            **kwargs (OpArgValue): Keyword arguments containing optional `count`.

        Returns:
            tuple[int, ...] | None: Inferred 1-D shape if `count != -1`, else None.
        """
        del args
        count_val = kwargs.get("count", -1)
        if isinstance(count_val, int) and count_val != -1:
            return (count_val,)
        return None
