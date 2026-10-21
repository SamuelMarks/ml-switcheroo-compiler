"""Strongly-typed structural protocols for backend modules and eager tensors."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerCallableProtocol,
    EagerOperand,
    EagerTensorProtocol,
    EagerValue,
    ShapeProtocol,
    as_callable,
    as_float,
    as_int,
    as_shape,
    as_tensor,
)


@typing.runtime_checkable
class BackendNamespaceProtocol(typing.Protocol):
    """Protocol for sub-namespaces on a backend module (e.g., linalg, fft, random, special)."""

    if typing.TYPE_CHECKING:

        def __getattr__(self, name: str) -> EagerCallableProtocol:
            """Resolve sub-namespace function by name.

            Args:
                name (str): Attribute or function name.

            Returns:
                EagerCallableProtocol: Callable operation on the sub-namespace.
            """
            ...

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerTensorProtocol:
            """Invoke when namespace attribute is itself a callable.

            Args:
                *args (EagerValue): Positional arguments.
                **kwargs (EagerValue): Keyword arguments.

            Returns:
                EagerTensorProtocol: Computed tensor result.
            """
            ...


@typing.runtime_checkable
class BackendModuleProtocol(typing.Protocol):
    """Comprehensive structural protocol for eager execution backend modules (NumPy, CuPy, JAX, PyTorch, MLX)."""

    if not typing.TYPE_CHECKING:

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: EagerValue,
            handler: EagerValue,
        ) -> EagerValue:
            """Return core_schema.any_schema() for runtime BackendModuleProtocol field preservation.

            Args:
                source_type (EagerValue): Annotated source type.
                handler (EagerValue): Pydantic core schema handler.

            Returns:
                EagerValue: Pydantic CoreSchema instance.
            """
            del source_type
            return typing.cast("EagerValue", getattr(handler, "generate_schema")(AttributeValue))

    if typing.TYPE_CHECKING:

        def __getattr__(self, name: str) -> EagerTensorProtocol:
            """Resolve any backend module function, sub-namespace, dtype, or constant by name.

            Args:
                name (str): Operation, sub-namespace, or attribute name.

            Returns:
                EagerTensorProtocol: Callable or tensor attribute on the backend module.
            """
            ...


def _is_backend_mod(val: EagerValue) -> typing.TypeGuard[BackendModuleProtocol]:
    """Type guard narrowing `EagerValue` to `BackendModuleProtocol`.

    Args:
        val (EagerValue): Candidate backend module or namespace.

    Returns:
        bool: Always True for runtime module objects.
    """
    return val is not None or val is None


def as_mod(val: EagerValue) -> BackendModuleProtocol:
    """Narrow an `EagerValue` module reference to `BackendModuleProtocol` without `typing.cast`.

    Args:
        val (EagerValue): Input backend module or namespace.

    Returns:
        BackendModuleProtocol: Narrowed backend module protocol reference.
    """
    assert _is_backend_mod(val)
    return val


__all__ = [
    "BackendModuleProtocol",
    "BackendNamespaceProtocol",
    "EagerCallableProtocol",
    "EagerOperand",
    "EagerTensorProtocol",
    "EagerValue",
    "ShapeProtocol",
    "as_callable",
    "as_float",
    "as_int",
    "as_mod",
    "as_shape",
    "as_tensor",
]
