"""Haskell-style parameterized capability protocols (typeclasses) and static dictionaries.

Provides compile-time structural verification (`mypy --strict`) that backend implementations
satisfy closed, typed operation contracts via explicit dictionary records rather than
untyped runtime reflection.
"""

from __future__ import annotations

import dataclasses
import enum
import typing
from collections.abc import Callable, Mapping, Sequence

from ml_switcheroo_ir import Eq, Invalid, Valid, Validation

from ml_switcheroo_compiler.backends.eager_registry.coercion import (
    coerce_opt_int_seq,
    coerce_opt_str,
    coerce_shape,
    require_callable,
)
from ml_switcheroo_compiler.backends.eager_registry.protocols import EagerBoundCallable
from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand
from ml_switcheroo_compiler.core.shape import VariadicCallableProtocol

TensorT = typing.TypeVar("TensorT")
TensorT_co = typing.TypeVar("TensorT_co", covariant=True)


class CapabilityKind(str, enum.Enum):
    """Static capability traits declared by registered compiler backends."""

    CODE_EMIT = "code_emit"
    CORE_EAGER = "core_eager"
    DOMAIN_EAGER = "domain_eager"
    AOT_COMPILE = "aot_compile"
    HARDWARE_KERNEL = "hardware_kernel"
    SERIALIZED_ARTIFACT = "serialized_artifact"


CORE_EAGER_REQUIRED_OPS: frozenset[str] = frozenset(
    {
        "zeros",
        "ones",
        "array",
        "asarray",
        "abs",
        "exp",
        "log",
        "sqrt",
        "sin",
        "cos",
        "tanh",
        "add",
        "subtract",
        "multiply",
        "divide",
        "matmul",
        "reshape",
        "transpose",
        "sum",
        "mean",
    }
)


@typing.runtime_checkable
class CreationTypeclass(typing.Protocol[TensorT_co]):
    """Typeclass contract for tensor allocation and host-data ingestion."""

    def zeros(self, shape: tuple[int, ...], dtype: str | None = None) -> TensorT_co:
        """Allocate a zero-filled tensor with `shape` and optional `dtype`.

        Args:
            shape (tuple[int, ...]): Target tensor dimensions.
            dtype (str | None): Optional canonical dtype name.

        Returns:
            TensorT_co: Zero-initialized backend tensor.
        """
        ...

    def ones(self, shape: tuple[int, ...], dtype: str | None = None) -> TensorT_co:
        """Allocate a one-filled tensor with `shape` and optional `dtype`.

        Args:
            shape (tuple[int, ...]): Target tensor dimensions.
            dtype (str | None): Optional canonical dtype name.

        Returns:
            TensorT_co: One-initialized backend tensor.
        """
        ...

    def array(self, data: BackendOperand, dtype: str | None = None) -> TensorT_co:
        """Construct a backend tensor from `data`.

        Args:
            data (BackendOperand): Host scalar, nested sequence, or array payload.
            dtype (str | None): Optional canonical dtype name.

        Returns:
            TensorT_co: Materialized backend tensor.
        """
        ...

    def asarray(self, data: BackendOperand, dtype: str | None = None) -> TensorT_co:
        """Convert `data` into a backend tensor without unnecessary copying.

        Args:
            data (BackendOperand): Host scalar, sequence, or existing backend tensor.
            dtype (str | None): Optional canonical dtype name.

        Returns:
            TensorT_co: Backend tensor view or copy.
        """
        ...


@typing.runtime_checkable
class ElementwiseUnaryTypeclass(typing.Protocol[TensorT]):
    """Typeclass contract for unary transcendental and magnitude operations."""

    def abs(self, x: TensorT) -> TensorT:
        """Compute elementwise absolute value of `x`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Elementwise magnitude tensor.
        """
        ...

    def exp(self, x: TensorT) -> TensorT:
        """Compute elementwise natural exponential of `x`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Exponential tensor.
        """
        ...

    def log(self, x: TensorT) -> TensorT:
        """Compute elementwise natural logarithm of `x`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Natural logarithm tensor.
        """
        ...

    def sqrt(self, x: TensorT) -> TensorT:
        """Compute elementwise principal square root of `x`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Square-root tensor.
        """
        ...

    def sin(self, x: TensorT) -> TensorT:
        """Compute elementwise sine of `x`.

        Args:
            x (TensorT): Input tensor in radians.

        Returns:
            TensorT: Sine tensor.
        """
        ...

    def cos(self, x: TensorT) -> TensorT:
        """Compute elementwise cosine of `x`.

        Args:
            x (TensorT): Input tensor in radians.

        Returns:
            TensorT: Cosine tensor.
        """
        ...

    def tanh(self, x: TensorT) -> TensorT:
        """Compute elementwise hyperbolic tangent of `x`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Hyperbolic tangent tensor.
        """
        ...


@typing.runtime_checkable
class ElementwiseBinaryTypeclass(typing.Protocol[TensorT]):
    """Typeclass contract for binary arithmetic ring/field operations."""

    def add(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute elementwise sum `lhs + rhs`.

        Args:
            lhs (TensorT): Left operand tensor.
            rhs (TensorT): Right operand tensor.

        Returns:
            TensorT: Sum tensor.
        """
        ...

    def subtract(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute elementwise difference `lhs - rhs`.

        Args:
            lhs (TensorT): Left operand tensor.
            rhs (TensorT): Right operand tensor.

        Returns:
            TensorT: Difference tensor.
        """
        ...

    def multiply(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute elementwise product `lhs * rhs`.

        Args:
            lhs (TensorT): Left operand tensor.
            rhs (TensorT): Right operand tensor.

        Returns:
            TensorT: Product tensor.
        """
        ...

    def divide(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute elementwise quotient `lhs / rhs`.

        Args:
            lhs (TensorT): Numerator tensor.
            rhs (TensorT): Denominator tensor.

        Returns:
            TensorT: Quotient tensor.
        """
        ...


@typing.runtime_checkable
class LinearAlgebraTypeclass(typing.Protocol[TensorT]):
    """Typeclass contract for core contraction and structural shape operations."""

    def matmul(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute matrix or batched tensor product `lhs @ rhs`.

        Args:
            lhs (TensorT): Left matrix/tensor operand.
            rhs (TensorT): Right matrix/tensor operand.

        Returns:
            TensorT: Contracted output tensor.
        """
        ...

    def reshape(self, x: TensorT, shape: tuple[int, ...]) -> TensorT:
        """Reshape `x` to `shape` preserving element count.

        Args:
            x (TensorT): Input tensor.
            shape (tuple[int, ...]): Target shape dimensions.

        Returns:
            TensorT: Reshaped tensor.
        """
        ...

    def transpose(self, x: TensorT, axes: Sequence[int] | None = None) -> TensorT:
        """Permute dimensions of `x` according to `axes`.

        Args:
            x (TensorT): Input tensor.
            axes (Sequence[int] | None): Optional permutation order.

        Returns:
            TensorT: Permuted tensor.
        """
        ...


@typing.runtime_checkable
class ReductionTypeclass(typing.Protocol[TensorT]):
    """Typeclass contract for axis-wise aggregation operations."""

    def sum(
        self,
        x: TensorT,
        axis: Sequence[int] | int | None = None,
        keepdims: bool = False,
    ) -> TensorT:
        """Sum elements of `x` over `axis`.

        Args:
            x (TensorT): Input tensor.
            axis (Sequence[int] | int | None): Axis or axes to reduce.
            keepdims (bool): Whether to retain reduced dimensions of size 1.

        Returns:
            TensorT: Reduced sum tensor.
        """
        ...

    def mean(
        self,
        x: TensorT,
        axis: Sequence[int] | int | None = None,
        keepdims: bool = False,
    ) -> TensorT:
        """Average elements of `x` over `axis`.

        Args:
            x (TensorT): Input tensor.
            axis (Sequence[int] | int | None): Axis or axes to reduce.
            keepdims (bool): Whether to retain reduced dimensions of size 1.

        Returns:
            TensorT: Reduced mean tensor.
        """
        ...


@typing.runtime_checkable
class CoreEagerTypeclass(
    CreationTypeclass[TensorT],
    ElementwiseUnaryTypeclass[TensorT],
    ElementwiseBinaryTypeclass[TensorT],
    LinearAlgebraTypeclass[TensorT],
    ReductionTypeclass[TensorT],
    typing.Protocol[TensorT],
):
    """Composite Haskell-style prelude typeclass for eager array backends."""

    def zeros(self, shape: tuple[int, ...], dtype: str | None = None) -> TensorT:
        """Allocate a zero-filled tensor with `shape` and optional `dtype`.

        Args:
            shape (tuple[int, ...]): Target tensor dimensions.
            dtype (str | None): Optional canonical dtype name.

        Returns:
            TensorT: Zero-initialized backend tensor.
        """
        ...


@typing.runtime_checkable
class CodeEmitterTypeclass(typing.Protocol):
    """Typeclass contract for IR-to-target source or artifact code emitters."""

    def generate(self) -> str:
        """Emit target source code or serialized representation for the bound IR graph.

        Returns:
            str: Generated target source text.
        """
        ...


@typing.runtime_checkable
class AOTCompilerTypeclass(typing.Protocol):
    """Typeclass contract for ahead-of-time executable graph compilers."""

    def compile_aot(self) -> VariadicCallableProtocol:
        """Compile the bound IR graph into an executable callable.

        Returns:
            VariadicCallableProtocol: Compiled graph runner callable.
        """
        ...


@dataclasses.dataclass(frozen=True)
class CreationDictionary(typing.Generic[TensorT]):
    """Explicit GHC-style method dictionary for `CreationTypeclass`."""

    zeros_fn: Callable[[tuple[int, ...], str | None], TensorT]
    ones_fn: Callable[[tuple[int, ...], str | None], TensorT]
    array_fn: Callable[[BackendOperand, str | None], TensorT]
    asarray_fn: Callable[[BackendOperand, str | None], TensorT]


@dataclasses.dataclass(frozen=True)
class ElementwiseUnaryDictionary(typing.Generic[TensorT]):
    """Explicit GHC-style method dictionary for `ElementwiseUnaryTypeclass`."""

    abs_fn: Callable[[TensorT], TensorT]
    exp_fn: Callable[[TensorT], TensorT]
    log_fn: Callable[[TensorT], TensorT]
    sqrt_fn: Callable[[TensorT], TensorT]
    sin_fn: Callable[[TensorT], TensorT]
    cos_fn: Callable[[TensorT], TensorT]
    tanh_fn: Callable[[TensorT], TensorT]


@dataclasses.dataclass(frozen=True)
class ElementwiseBinaryDictionary(typing.Generic[TensorT]):
    """Explicit GHC-style method dictionary for `ElementwiseBinaryTypeclass`."""

    add_fn: Callable[[TensorT, TensorT], TensorT]
    subtract_fn: Callable[[TensorT, TensorT], TensorT]
    multiply_fn: Callable[[TensorT, TensorT], TensorT]
    divide_fn: Callable[[TensorT, TensorT], TensorT]


@dataclasses.dataclass(frozen=True)
class LinearAlgebraDictionary(typing.Generic[TensorT]):
    """Explicit GHC-style method dictionary for `LinearAlgebraTypeclass`."""

    matmul_fn: Callable[[TensorT, TensorT], TensorT]
    reshape_fn: Callable[[TensorT, tuple[int, ...]], TensorT]
    transpose_fn: Callable[[TensorT, Sequence[int] | None], TensorT]


@dataclasses.dataclass(frozen=True)
class ReductionDictionary(typing.Generic[TensorT]):
    """Explicit GHC-style method dictionary for `ReductionTypeclass`."""

    sum_fn: Callable[[TensorT, Sequence[int] | int | None, bool], TensorT]
    mean_fn: Callable[[TensorT, Sequence[int] | int | None, bool], TensorT]


def _wrap_unary_bound(fn: Callable[[BackendOperand], BackendOperand]) -> EagerBoundCallable:
    """Adapt a typed unary function into an `EagerBoundCallable` for O(1) adapter dispatch.

    Args:
        fn (Callable[[BackendOperand], BackendOperand]): Typed unary operation.

    Returns:
        EagerBoundCallable: Variadic adapter callable.
    """

    def _bound(*args: BackendOperand, **kwargs: BackendOperand) -> BackendOperand:
        """Invoke typed unary op on first positional operand.

        Args:
            *args (BackendOperand): Positional operands (`x`,).
            **kwargs (BackendOperand): Unused keyword arguments.

        Returns:
            BackendOperand: Unary operation result.
        """
        del kwargs
        return fn(args[0])

    return require_callable(_bound)


def _wrap_binary_bound(fn: Callable[[BackendOperand, BackendOperand], BackendOperand]) -> EagerBoundCallable:
    """Adapt a typed binary function into an `EagerBoundCallable` for O(1) adapter dispatch.

    Args:
        fn (Callable[[BackendOperand, BackendOperand], BackendOperand]): Typed binary operation.

    Returns:
        EagerBoundCallable: Variadic adapter callable.
    """

    def _bound(*args: BackendOperand, **kwargs: BackendOperand) -> BackendOperand:
        """Invoke typed binary op on `(lhs, rhs)`.

        Args:
            *args (BackendOperand): Positional operands (`lhs`, `rhs`).
            **kwargs (BackendOperand): Unused keyword arguments.

        Returns:
            BackendOperand: Binary operation result.
        """
        del kwargs
        return fn(args[0], args[1])

    return require_callable(_bound)


def _wrap_shape_alloc_bound(
    fn: Callable[[tuple[int, ...], str | None], BackendOperand],
) -> EagerBoundCallable:
    """Adapt a typed allocation function `(shape, dtype)` into an `EagerBoundCallable`.

    Args:
        fn (Callable[[tuple[int, ...], str | None], BackendOperand]): Allocation function.

    Returns:
        EagerBoundCallable: Variadic adapter callable.
    """

    def _bound(*args: BackendOperand, **kwargs: BackendOperand) -> BackendOperand:
        """Invoke typed allocation op coercing shape and optional dtype.

        Args:
            *args (BackendOperand): Positional arguments (`shape`, optional `dtype`).
            **kwargs (BackendOperand): Optional `dtype` keyword argument.

        Returns:
            BackendOperand: Allocated tensor.
        """
        shape = coerce_shape(args[0]) if args else ()
        raw_dt = args[1] if len(args) > 1 else kwargs.get("dtype")
        return fn(shape, coerce_opt_str(raw_dt))

    return require_callable(_bound)


def _wrap_array_ingest_bound(
    fn: Callable[[BackendOperand, str | None], BackendOperand],
) -> EagerBoundCallable:
    """Adapt a typed array ingestion function `(data, dtype)` into an `EagerBoundCallable`.

    Args:
        fn (Callable[[BackendOperand, str | None], BackendOperand]): Array conversion function.

    Returns:
        EagerBoundCallable: Variadic adapter callable.
    """

    def _bound(*args: BackendOperand, **kwargs: BackendOperand) -> BackendOperand:
        """Invoke typed array ingestion op with optional `dtype`.

        Args:
            *args (BackendOperand): Positional arguments (`data`, optional `dtype`).
            **kwargs (BackendOperand): Optional `dtype` keyword argument.

        Returns:
            BackendOperand: Materialized tensor.
        """
        data: BackendOperand = args[0] if args else None
        raw_dt = args[1] if len(args) > 1 else kwargs.get("dtype")
        return fn(data, coerce_opt_str(raw_dt))

    return require_callable(_bound)


def _wrap_reduction_bound(
    fn: Callable[[BackendOperand, Sequence[int] | int | None, bool], BackendOperand],
) -> EagerBoundCallable:
    """Adapt a typed reduction function `(x, axis, keepdims)` into an `EagerBoundCallable`.

    Args:
        fn (Callable[[BackendOperand, Sequence[int] | int | None, bool], BackendOperand]): Reduction function.

    Returns:
        EagerBoundCallable: Variadic adapter callable.
    """

    def _bound(*args: BackendOperand, **kwargs: BackendOperand) -> BackendOperand:
        """Invoke typed reduction op normalizing `axis` and `keepdims`.

        Args:
            *args (BackendOperand): Positional arguments (`x`, optional `axis`, optional `keepdims`).
            **kwargs (BackendOperand): Optional `axis` and `keepdims` keyword arguments.

        Returns:
            BackendOperand: Reduced tensor.
        """
        raw_axis = args[1] if len(args) > 1 else kwargs.get("axis")
        axis: Sequence[int] | int | None
        if isinstance(raw_axis, int) and not isinstance(raw_axis, bool):
            axis = raw_axis
        else:
            axis = coerce_opt_int_seq(raw_axis)
        raw_kd = args[2] if len(args) > 2 else kwargs.get("keepdims", False)
        return fn(args[0], axis, bool(raw_kd))

    return require_callable(_bound)


@dataclasses.dataclass(frozen=True)
class CoreEagerDictionary(typing.Generic[TensorT]):
    """Composite GHC-style superclass dictionary implementing `CoreEagerTypeclass[TensorT]`."""

    creation: CreationDictionary[TensorT]
    unary: ElementwiseUnaryDictionary[TensorT]
    binary: ElementwiseBinaryDictionary[TensorT]
    linalg: LinearAlgebraDictionary[TensorT]
    reduction: ReductionDictionary[TensorT]

    def zeros(self, shape: tuple[int, ...], dtype: str | None = None) -> TensorT:
        """Allocate zero tensor via `creation.zeros_fn`.

        Args:
            shape (tuple[int, ...]): Target shape.
            dtype (str | None): Optional dtype string.

        Returns:
            TensorT: Zero tensor.
        """
        return self.creation.zeros_fn(shape, dtype)

    def ones(self, shape: tuple[int, ...], dtype: str | None = None) -> TensorT:
        """Allocate ones tensor via `creation.ones_fn`.

        Args:
            shape (tuple[int, ...]): Target shape.
            dtype (str | None): Optional dtype string.

        Returns:
            TensorT: Ones tensor.
        """
        return self.creation.ones_fn(shape, dtype)

    def array(self, data: BackendOperand, dtype: str | None = None) -> TensorT:
        """Construct tensor via `creation.array_fn`.

        Args:
            data (BackendOperand): Host payload.
            dtype (str | None): Optional dtype string.

        Returns:
            TensorT: Backend tensor.
        """
        return self.creation.array_fn(data, dtype)

    def asarray(self, data: BackendOperand, dtype: str | None = None) -> TensorT:
        """Convert `data` via `creation.asarray_fn`.

        Args:
            data (BackendOperand): Host or tensor payload.
            dtype (str | None): Optional dtype string.

        Returns:
            TensorT: Backend tensor.
        """
        return self.creation.asarray_fn(data, dtype)

    def abs(self, x: TensorT) -> TensorT:
        """Compute `abs(x)` via `unary.abs_fn`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Magnitude tensor.
        """
        return self.unary.abs_fn(x)

    def exp(self, x: TensorT) -> TensorT:
        """Compute `exp(x)` via `unary.exp_fn`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Exponential tensor.
        """
        return self.unary.exp_fn(x)

    def log(self, x: TensorT) -> TensorT:
        """Compute `log(x)` via `unary.log_fn`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Natural logarithm tensor.
        """
        return self.unary.log_fn(x)

    def sqrt(self, x: TensorT) -> TensorT:
        """Compute `sqrt(x)` via `unary.sqrt_fn`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Square-root tensor.
        """
        return self.unary.sqrt_fn(x)

    def sin(self, x: TensorT) -> TensorT:
        """Compute `sin(x)` via `unary.sin_fn`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Sine tensor.
        """
        return self.unary.sin_fn(x)

    def cos(self, x: TensorT) -> TensorT:
        """Compute `cos(x)` via `unary.cos_fn`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Cosine tensor.
        """
        return self.unary.cos_fn(x)

    def tanh(self, x: TensorT) -> TensorT:
        """Compute `tanh(x)` via `unary.tanh_fn`.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Hyperbolic tangent tensor.
        """
        return self.unary.tanh_fn(x)

    def add(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute `lhs + rhs` via `binary.add_fn`.

        Args:
            lhs (TensorT): Left operand.
            rhs (TensorT): Right operand.

        Returns:
            TensorT: Sum tensor.
        """
        return self.binary.add_fn(lhs, rhs)

    def subtract(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute `lhs - rhs` via `binary.subtract_fn`.

        Args:
            lhs (TensorT): Left operand.
            rhs (TensorT): Right operand.

        Returns:
            TensorT: Difference tensor.
        """
        return self.binary.subtract_fn(lhs, rhs)

    def multiply(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute `lhs * rhs` via `binary.multiply_fn`.

        Args:
            lhs (TensorT): Left operand.
            rhs (TensorT): Right operand.

        Returns:
            TensorT: Product tensor.
        """
        return self.binary.multiply_fn(lhs, rhs)

    def divide(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute `lhs / rhs` via `binary.divide_fn`.

        Args:
            lhs (TensorT): Numerator operand.
            rhs (TensorT): Denominator operand.

        Returns:
            TensorT: Quotient tensor.
        """
        return self.binary.divide_fn(lhs, rhs)

    def matmul(self, lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute `lhs @ rhs` via `linalg.matmul_fn`.

        Args:
            lhs (TensorT): Left matrix/tensor.
            rhs (TensorT): Right matrix/tensor.

        Returns:
            TensorT: Contracted tensor.
        """
        return self.linalg.matmul_fn(lhs, rhs)

    def reshape(self, x: TensorT, shape: tuple[int, ...]) -> TensorT:
        """Reshape `x` via `linalg.reshape_fn`.

        Args:
            x (TensorT): Input tensor.
            shape (tuple[int, ...]): Target shape.

        Returns:
            TensorT: Reshaped tensor.
        """
        return self.linalg.reshape_fn(x, shape)

    def transpose(self, x: TensorT, axes: Sequence[int] | None = None) -> TensorT:
        """Permute `x` via `linalg.transpose_fn`.

        Args:
            x (TensorT): Input tensor.
            axes (Sequence[int] | None): Optional permutation axes.

        Returns:
            TensorT: Transposed tensor.
        """
        return self.linalg.transpose_fn(x, axes)

    def sum(
        self,
        x: TensorT,
        axis: Sequence[int] | int | None = None,
        keepdims: bool = False,
    ) -> TensorT:
        """Reduce sum via `reduction.sum_fn`.

        Args:
            x (TensorT): Input tensor.
            axis (Sequence[int] | int | None): Axis or axes to sum.
            keepdims (bool): Retain reduced dimensions when True.

        Returns:
            TensorT: Summed tensor.
        """
        return self.reduction.sum_fn(x, axis, keepdims)

    def mean(
        self,
        x: TensorT,
        axis: Sequence[int] | int | None = None,
        keepdims: bool = False,
    ) -> TensorT:
        """Reduce mean via `reduction.mean_fn`.

        Args:
            x (TensorT): Input tensor.
            axis (Sequence[int] | int | None): Axis or axes to average.
            keepdims (bool): Retain reduced dimensions when True.

        Returns:
            TensorT: Mean tensor.
        """
        return self.reduction.mean_fn(x, axis, keepdims)


def build_typeclass_op_table(
    dictionary: CoreEagerTypeclass[BackendOperand],
) -> dict[str, EagerBoundCallable]:
    """Project a verified `CoreEagerTypeclass` instance into an O(1) static dispatch table.

    Args:
        dictionary (CoreEagerTypeclass[BackendOperand]): Typechecked capability instance.

    Returns:
        dict[str, EagerBoundCallable]: Mapping from canonical op names to bound callables.
    """

    def _reshape_bound(*args: BackendOperand, **kwargs: BackendOperand) -> BackendOperand:
        """Invoke typed `reshape` coercing target shape.

        Args:
            *args (BackendOperand): Positional arguments (`x`, `shape`).
            **kwargs (BackendOperand): Optional `shape` keyword argument.

        Returns:
            BackendOperand: Reshaped tensor.
        """
        raw_shape = args[1] if len(args) > 1 else kwargs.get("shape", ())
        return dictionary.reshape(args[0], coerce_shape(raw_shape))

    def _transpose_bound(*args: BackendOperand, **kwargs: BackendOperand) -> BackendOperand:
        """Invoke typed `transpose` coercing optional `axes`.

        Args:
            *args (BackendOperand): Positional arguments (`x`, optional `axes`).
            **kwargs (BackendOperand): Optional `axes` keyword argument.

        Returns:
            BackendOperand: Transposed tensor.
        """
        raw_axes = args[1] if len(args) > 1 else kwargs.get("axes")
        return dictionary.transpose(args[0], coerce_opt_int_seq(raw_axes))

    return {
        "zeros": _wrap_shape_alloc_bound(dictionary.zeros),
        "ones": _wrap_shape_alloc_bound(dictionary.ones),
        "array": _wrap_array_ingest_bound(dictionary.array),
        "asarray": _wrap_array_ingest_bound(dictionary.asarray),
        "abs": _wrap_unary_bound(dictionary.abs),
        "exp": _wrap_unary_bound(dictionary.exp),
        "log": _wrap_unary_bound(dictionary.log),
        "sqrt": _wrap_unary_bound(dictionary.sqrt),
        "sin": _wrap_unary_bound(dictionary.sin),
        "cos": _wrap_unary_bound(dictionary.cos),
        "tanh": _wrap_unary_bound(dictionary.tanh),
        "add": _wrap_binary_bound(dictionary.add),
        "subtract": _wrap_binary_bound(dictionary.subtract),
        "multiply": _wrap_binary_bound(dictionary.multiply),
        "divide": _wrap_binary_bound(dictionary.divide),
        "matmul": _wrap_binary_bound(dictionary.matmul),
        "reshape": require_callable(_reshape_bound),
        "transpose": require_callable(_transpose_bound),
        "sum": _wrap_reduction_bound(dictionary.sum),
        "mean": _wrap_reduction_bound(dictionary.mean),
    }


def verify_core_eager_typeclass(
    instance: CoreEagerTypeclass[TensorT],
) -> CoreEagerTypeclass[TensorT]:
    """Witness compile-time (`mypy --strict`) conformance of `instance` to `CoreEagerTypeclass[TensorT]`.

    Args:
        instance (CoreEagerTypeclass[TensorT]): Candidate typeclass implementation.

    Returns:
        CoreEagerTypeclass[TensorT]: The verified typeclass instance unchanged.
    """
    return instance


def build_core_eager_dictionary_from_ops(
    ops_map: Mapping[str, EagerBoundCallable],
) -> CoreEagerTypeclass[BackendOperand]:
    """Synthesize a verified `CoreEagerDictionary[BackendOperand]` from a complete `ops_map`.

    Args:
        ops_map (Mapping[str, EagerBoundCallable]): Bound operation map containing `CORE_EAGER_REQUIRED_OPS`.

    Returns:
        CoreEagerTypeclass[BackendOperand]: Verified typeclass dictionary instance.
    """
    return verify_core_eager_typeclass(
        CoreEagerDictionary[BackendOperand](
            creation=CreationDictionary[BackendOperand](
                zeros_fn=lambda shape, dtype: (
                    ops_map["zeros"](shape, dtype=dtype) if dtype is not None else ops_map["zeros"](shape)
                ),
                ones_fn=lambda shape, dtype: (
                    ops_map["ones"](shape, dtype=dtype) if dtype is not None else ops_map["ones"](shape)
                ),
                array_fn=lambda data, dtype: (
                    ops_map["array"](data, dtype=dtype) if dtype is not None else ops_map["array"](data)
                ),
                asarray_fn=lambda data, dtype: (
                    ops_map["asarray"](data, dtype=dtype) if dtype is not None else ops_map["asarray"](data)
                ),
            ),
            unary=ElementwiseUnaryDictionary[BackendOperand](
                abs_fn=ops_map["abs"],
                exp_fn=ops_map["exp"],
                log_fn=ops_map["log"],
                sqrt_fn=ops_map["sqrt"],
                sin_fn=ops_map["sin"],
                cos_fn=ops_map["cos"],
                tanh_fn=ops_map["tanh"],
            ),
            binary=ElementwiseBinaryDictionary[BackendOperand](
                add_fn=ops_map["add"],
                subtract_fn=ops_map["subtract"],
                multiply_fn=ops_map["multiply"],
                divide_fn=ops_map["divide"],
            ),
            linalg=LinearAlgebraDictionary[BackendOperand](
                matmul_fn=ops_map["matmul"],
                reshape_fn=lambda x, shape: ops_map["reshape"](x, shape),
                transpose_fn=lambda x, axes: (
                    ops_map["transpose"](x, axes) if axes is not None else ops_map["transpose"](x)
                ),
            ),
            reduction=ReductionDictionary[BackendOperand](
                sum_fn=lambda x, axis, keepdims: ops_map["sum"](x, axis=axis, keepdims=keepdims),
                mean_fn=lambda x, axis, keepdims: ops_map["mean"](x, axis=axis, keepdims=keepdims),
            ),
        )
    )


@dataclasses.dataclass(frozen=True)
class MinimalCoreEagerPrimitives(typing.Generic[TensorT]):
    """Haskell-style Minimal Complete Definition (MCD) basis for ``CoreEagerTypeclass``.

    Supplies irreducible creation/ring/transcendental/linalg/reduction primitives alongside
    Haskell ``Fractional``-style ``divide_fn | reciprocal_fn`` and optional overrides for
    algebraically derivable operations (``asarray``, ``subtract``, ``divide``, ``sqrt``, ``abs``, ``tanh``, ``mean``).
    """

    zeros_fn: Callable[[tuple[int, ...], str | None], TensorT]
    ones_fn: Callable[[tuple[int, ...], str | None], TensorT]
    array_fn: Callable[[BackendOperand, str | None], TensorT]
    exp_fn: Callable[[TensorT], TensorT]
    log_fn: Callable[[TensorT], TensorT]
    sin_fn: Callable[[TensorT], TensorT]
    cos_fn: Callable[[TensorT], TensorT]
    add_fn: Callable[[TensorT, TensorT], TensorT]
    multiply_fn: Callable[[TensorT, TensorT], TensorT]
    matmul_fn: Callable[[TensorT, TensorT], TensorT]
    reshape_fn: Callable[[TensorT, tuple[int, ...]], TensorT]
    transpose_fn: Callable[[TensorT, Sequence[int] | None], TensorT]
    sum_fn: Callable[[TensorT, Sequence[int] | int | None, bool], TensorT]
    divide_fn: Callable[[TensorT, TensorT], TensorT] | None = None
    reciprocal_fn: Callable[[TensorT], TensorT] | None = None
    asarray_fn: Callable[[BackendOperand, str | None], TensorT] | None = None
    subtract_fn: Callable[[TensorT, TensorT], TensorT] | None = None
    sqrt_fn: Callable[[TensorT], TensorT] | None = None
    abs_fn: Callable[[TensorT], TensorT] | None = None
    tanh_fn: Callable[[TensorT], TensorT] | None = None
    mean_fn: Callable[[TensorT, Sequence[int] | int | None, bool], TensorT] | None = None


def _resolve_divide_primitive(
    primitives: MinimalCoreEagerPrimitives[TensorT],
) -> Callable[[TensorT, TensorT], TensorT]:
    """Resolve ``divide`` from ``divide_fn`` or ``reciprocal_fn`` (Haskell ``Fractional`` MCD).

    Args:
        primitives (MinimalCoreEagerPrimitives[TensorT]): Minimal primitive basis.

    Returns:
        Callable[[TensorT, TensorT], TensorT]: Concrete or reciprocal-derived binary division function.

    Raises:
        ValueError: If neither ``divide_fn`` nor ``reciprocal_fn`` is provided.
    """
    if primitives.divide_fn is not None:
        return primitives.divide_fn
    if primitives.reciprocal_fn is not None:
        recip = primitives.reciprocal_fn
        return lambda lhs, rhs: primitives.multiply_fn(lhs, recip(rhs))
    raise ValueError("MinimalCoreEagerPrimitives requires at least one of divide_fn or reciprocal_fn.")


def derive_core_eager_dictionary(
    primitives: MinimalCoreEagerPrimitives[TensorT],
) -> CoreEagerDictionary[TensorT]:
    """Derive a complete ``CoreEagerDictionary[TensorT]`` from a ``MinimalCoreEagerPrimitives`` basis.

    Args:
        primitives (MinimalCoreEagerPrimitives[TensorT]): Minimal primitive basis with optional overrides.

    Returns:
        CoreEagerDictionary[TensorT]: Complete typeclass dictionary satisfying ``CoreEagerTypeclass[TensorT]``.
    """
    divide_impl = _resolve_divide_primitive(primitives)
    asarray_impl: Callable[[BackendOperand, str | None], TensorT] = (
        primitives.asarray_fn if primitives.asarray_fn is not None else primitives.array_fn
    )

    def _default_subtract(lhs: TensorT, rhs: TensorT) -> TensorT:
        """Compute ``lhs - rhs`` as ``lhs + (rhs * -1.0)``.

        Args:
            lhs (TensorT): Left operand.
            rhs (TensorT): Right operand.

        Returns:
            TensorT: Elementwise difference.
        """
        neg_one = primitives.array_fn(-1.0, None)
        return primitives.add_fn(lhs, primitives.multiply_fn(rhs, neg_one))

    subtract_impl: Callable[[TensorT, TensorT], TensorT] = (
        primitives.subtract_fn if primitives.subtract_fn is not None else _default_subtract
    )

    def _default_sqrt(x: TensorT) -> TensorT:
        """Compute ``sqrt(x)`` as ``exp(0.5 * log(x))``.

        Args:
            x (TensorT): Non-negative input tensor.

        Returns:
            TensorT: Principal square root tensor.
        """
        half = primitives.array_fn(0.5, None)
        return primitives.exp_fn(primitives.multiply_fn(half, primitives.log_fn(x)))

    sqrt_impl: Callable[[TensorT], TensorT] = primitives.sqrt_fn if primitives.sqrt_fn is not None else _default_sqrt

    def _default_abs(x: TensorT) -> TensorT:
        """Compute ``abs(x)`` as ``sqrt(x * x)``.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Elementwise magnitude tensor.
        """
        return sqrt_impl(primitives.multiply_fn(x, x))

    abs_impl: Callable[[TensorT], TensorT] = primitives.abs_fn if primitives.abs_fn is not None else _default_abs

    def _default_tanh(x: TensorT) -> TensorT:
        """Compute ``tanh(x)`` as ``(exp(x) - exp(-x)) / (exp(x) + exp(-x))``.

        Args:
            x (TensorT): Input tensor.

        Returns:
            TensorT: Hyperbolic tangent tensor.
        """
        neg_x = primitives.multiply_fn(x, primitives.array_fn(-1.0, None))
        ep = primitives.exp_fn(x)
        em = primitives.exp_fn(neg_x)
        return divide_impl(subtract_impl(ep, em), primitives.add_fn(ep, em))

    tanh_impl: Callable[[TensorT], TensorT] = primitives.tanh_fn if primitives.tanh_fn is not None else _default_tanh

    def _default_mean(
        x: TensorT,
        axis: Sequence[int] | int | None = None,
        keepdims: bool = False,
    ) -> TensorT:
        """Compute ``mean(x, axis, keepdims)`` algebraically via ``sum(x) / sum(ones_like(x))``.

        Args:
            x (TensorT): Input tensor.
            axis (Sequence[int] | int | None): Reduction axis or axes.
            keepdims (bool): Whether to retain reduced dimensions.

        Returns:
            TensorT: Reduced mean tensor.
        """
        zero_s = primitives.array_fn(0.0, None)
        one_s = primitives.array_fn(1.0, None)
        ones_like_x = primitives.add_fn(primitives.multiply_fn(x, zero_s), one_s)
        denom = primitives.sum_fn(ones_like_x, axis, keepdims)
        return divide_impl(primitives.sum_fn(x, axis, keepdims), denom)

    mean_impl: Callable[[TensorT, Sequence[int] | int | None, bool], TensorT] = (
        primitives.mean_fn if primitives.mean_fn is not None else _default_mean
    )

    return CoreEagerDictionary[TensorT](
        creation=CreationDictionary[TensorT](
            zeros_fn=primitives.zeros_fn,
            ones_fn=primitives.ones_fn,
            array_fn=primitives.array_fn,
            asarray_fn=asarray_impl,
        ),
        unary=ElementwiseUnaryDictionary[TensorT](
            abs_fn=abs_impl,
            exp_fn=primitives.exp_fn,
            log_fn=primitives.log_fn,
            sqrt_fn=sqrt_impl,
            sin_fn=primitives.sin_fn,
            cos_fn=primitives.cos_fn,
            tanh_fn=tanh_impl,
        ),
        binary=ElementwiseBinaryDictionary[TensorT](
            add_fn=primitives.add_fn,
            subtract_fn=subtract_impl,
            multiply_fn=primitives.multiply_fn,
            divide_fn=divide_impl,
        ),
        linalg=LinearAlgebraDictionary[TensorT](
            matmul_fn=primitives.matmul_fn,
            reshape_fn=primitives.reshape_fn,
            transpose_fn=primitives.transpose_fn,
        ),
        reduction=ReductionDictionary[TensorT](
            sum_fn=primitives.sum_fn,
            mean_fn=mean_impl,
        ),
    )


def verify_core_eager_algebraic_laws(
    eager_dict: CoreEagerTypeclass[TensorT],
    approx_eq: Eq[TensorT],
    label: str = "CoreEager",
) -> Validation[tuple[str, ...], None]:
    """Verify ring, inverse, matrix-associativity, transpose-involution, and trig/exp laws on ``eager_dict``.

    Args:
        eager_dict (CoreEagerTypeclass[TensorT]): Parameterized eager tensor dictionary witness.
        approx_eq (Eq[TensorT]): Approximate numerical equality witness over ``TensorT``.
        label (str): Human-readable backend identifier for error messages.

    Returns:
        Validation[tuple[str, ...], None]: ``Valid(None)`` when all laws hold, else ``Invalid`` tuple.
    """
    z = eager_dict.zeros((2, 2), None)
    o = eager_dict.ones((2, 2), None)
    a = eager_dict.reshape(eager_dict.array([1.0, 2.0, 3.0, 4.0], None), (2, 2))
    b = eager_dict.reshape(eager_dict.array([0.5, -1.0, 2.0, 1.5], None), (2, 2))
    c = eager_dict.reshape(eager_dict.array([2.0, 0.5, 1.0, 3.0], None), (2, 2))
    eye2 = eager_dict.reshape(eager_dict.array([1.0, 0.0, 0.0, 1.0], None), (2, 2))

    obligations: tuple[tuple[str, TensorT, TensorT], ...] = (
        ("add_zero_identity", eager_dict.add(z, a), a),
        ("add_commutativity", eager_dict.add(a, b), eager_dict.add(b, a)),
        ("add_associativity", eager_dict.add(eager_dict.add(a, b), c), eager_dict.add(a, eager_dict.add(b, c))),
        ("mul_one_identity", eager_dict.multiply(o, a), a),
        (
            "left_distributivity",
            eager_dict.multiply(a, eager_dict.add(b, c)),
            eager_dict.add(eager_dict.multiply(a, b), eager_dict.multiply(a, c)),
        ),
        ("subtract_inverse", eager_dict.add(eager_dict.subtract(a, b), b), a),
        ("divide_inverse", eager_dict.multiply(eager_dict.divide(a, c), c), a),
        ("matmul_left_identity", eager_dict.matmul(eye2, a), a),
        (
            "matmul_associativity",
            eager_dict.matmul(eager_dict.matmul(a, b), c),
            eager_dict.matmul(a, eager_dict.matmul(b, c)),
        ),
        ("transpose_involution", eager_dict.transpose(eager_dict.transpose(a, (1, 0)), (1, 0)), a),
        ("sqrt_square", eager_dict.sqrt(eager_dict.multiply(c, c)), c),
        ("log_exp_inverse", eager_dict.log(eager_dict.exp(b)), b),
        (
            "pythagorean_trig",
            eager_dict.add(
                eager_dict.multiply(eager_dict.sin(b), eager_dict.sin(b)),
                eager_dict.multiply(eager_dict.cos(b), eager_dict.cos(b)),
            ),
            o,
        ),
    )
    violations = [f"{label}: failed {name}" for name, lhs, rhs in obligations if not approx_eq.eq(lhs, rhs)]
    if violations:
        return Invalid(tuple(violations))
    return Valid(None)


@dataclasses.dataclass(frozen=True)
class CodeEmitterDictionary:
    """Explicit GHC-style capability dictionary implementing ``CodeEmitterTypeclass``."""

    generate_fn: Callable[[], str]

    def generate(self) -> str:
        """Emit target source code using the bound ``generate_fn`` witness.

        Returns:
            str: Generated target source text.
        """
        return self.generate_fn()


@dataclasses.dataclass(frozen=True)
class AOTCompilerDictionary:
    """Explicit GHC-style capability dictionary implementing ``AOTCompilerTypeclass``."""

    compile_aot_fn: Callable[[], VariadicCallableProtocol]

    def compile_aot(self) -> VariadicCallableProtocol:
        """Compile bound IR graph into an executable callable via ``compile_aot_fn``.

        Returns:
            VariadicCallableProtocol: Compiled runner callable.
        """
        return self.compile_aot_fn()


@dataclasses.dataclass(frozen=True)
class BackendCapabilityBundle(typing.Generic[TensorT]):
    """Composed dictionary-passing record bundling codegen, AOT, and eager typeclass witnesses."""

    backend_name: str
    code_emitter: CodeEmitterTypeclass | None = None
    aot_compiler: AOTCompilerTypeclass | None = None
    core_eager: CoreEagerTypeclass[TensorT] | None = None

    def capabilities(self) -> frozenset[CapabilityKind]:
        """Derive ``frozenset[CapabilityKind]`` directly from populated capability dictionaries.

        Returns:
            frozenset[CapabilityKind]: Set of active backend capabilities.
        """
        caps: set[CapabilityKind] = set()
        if self.code_emitter is not None:
            caps.add(CapabilityKind.CODE_EMIT)
        if self.aot_compiler is not None:
            caps.add(CapabilityKind.AOT_COMPILE)
        if self.core_eager is not None:
            caps.add(CapabilityKind.CORE_EAGER)
        return frozenset(caps)


__all__ = [
    "AOTCompilerDictionary",
    "AOTCompilerTypeclass",
    "BackendCapabilityBundle",
    "CORE_EAGER_REQUIRED_OPS",
    "CapabilityKind",
    "CodeEmitterDictionary",
    "CodeEmitterTypeclass",
    "CoreEagerDictionary",
    "CoreEagerTypeclass",
    "CreationDictionary",
    "CreationTypeclass",
    "ElementwiseBinaryDictionary",
    "ElementwiseBinaryTypeclass",
    "ElementwiseUnaryDictionary",
    "ElementwiseUnaryTypeclass",
    "LinearAlgebraDictionary",
    "LinearAlgebraTypeclass",
    "MinimalCoreEagerPrimitives",
    "ReductionDictionary",
    "ReductionTypeclass",
    "TensorT",
    "TensorT_co",
    "build_core_eager_dictionary_from_ops",
    "build_typeclass_op_table",
    "derive_core_eager_dictionary",
    "verify_core_eager_algebraic_laws",
    "verify_core_eager_typeclass",
]
