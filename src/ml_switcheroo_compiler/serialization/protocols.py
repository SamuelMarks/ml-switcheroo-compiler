"""Structural typing protocols and shared metadata helpers for serialization."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.shape import OpValue as AttributeValue


@typing.runtime_checkable
class WeightedModelProtocol(typing.Protocol):
    """Protocol for models exposing a `weights` iterable."""

    weights: Iterable[OpValue]


@typing.runtime_checkable
class _NamedWeightProtocol(typing.Protocol):
    """Protocol for weight variables exposing a `name` attribute."""

    name: str | None


@typing.runtime_checkable
class _AssignableWeightProtocol(typing.Protocol):
    """Protocol for weight variables supporting `.assign(value)`."""

    def assign(self, value: AttributeValue) -> None:
        """Assign a restored weight array into this variable.

        Args:
            value (AttributeValue): Restored weight array.
        """
        ...


@typing.runtime_checkable
class _ShapedVariableProtocol(typing.Protocol):
    """Protocol for variables or arrays exposing `.shape`."""

    shape: Sequence[int]


@typing.runtime_checkable
class _OptimizerVariablesProtocol(typing.Protocol):
    """Protocol for optimizers exposing `.variables`."""

    variables: Iterable[EagerTensorProtocol]


@typing.runtime_checkable
class _OptimizerMomentumsProtocol(typing.Protocol):
    """Protocol for optimizers exposing `.momentums`."""

    momentums: Iterable[EagerTensorProtocol]


@typing.runtime_checkable
class _OptimizerModelProtocol(typing.Protocol):
    """Protocol for models exposing an `.optimizer` attribute."""

    optimizer: OpValue


@typing.runtime_checkable
class _NonTrainableModelProtocol(typing.Protocol):
    """Protocol for models exposing `.non_trainable_variables`."""

    non_trainable_variables: Iterable[OpValue]


@typing.runtime_checkable
class _EmaModelProtocol(typing.Protocol):
    """Protocol for models exposing `.ema_variables`."""

    ema_variables: Iterable[OpValue]


@typing.runtime_checkable
class _ConfigProviderProtocol(typing.Protocol):
    """Protocol for objects exposing `.get_config()`."""

    def get_config(self) -> dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]]:
        """Return configuration dictionary.

        Returns:
            dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]]: Configuration
                dictionary.
        """
        ...


@typing.runtime_checkable
class _FromConfigProtocol(typing.Protocol):
    """Protocol for classes exposing `.from_config(config)`."""

    def from_config(self, config: AttributeValue) -> AttributeValue:
        """Instantiate object from configuration payload.

        Args:
            config (AttributeValue): Configuration payload.

        Returns:
            AttributeValue: Reconstructed object.
        """
        ...


@typing.runtime_checkable
class _GraphProviderProtocol(typing.Protocol):
    """Protocol for models exposing `.graph`."""

    graph: IRGraph


@typing.runtime_checkable
class _ToJsonProviderProtocol(typing.Protocol):
    """Protocol for models exposing `.to_json()`."""

    def to_json(self) -> str:
        """Serialize model topology to JSON string.

        Returns:
            str: JSON string.
        """
        ...


def _weight_name(w: OpValue, default: str | None = None) -> str | None:
    """Extract the `.name` string from a weight variable if present.

    Args:
        w (OpValue): Weight variable object.
        default (str | None): Fallback name if `.name` is missing.

    Returns:
        str | None: Weight variable name or default.
    """
    cand_w: OpValue | _NamedWeightProtocol = w
    if isinstance(cand_w, _NamedWeightProtocol) and cand_w.name is not None:
        return str(cand_w.name)
    return default


def _extract_shape_tuple(val: AttributeValue) -> tuple[int, ...]:
    """Extract a shape tuple from a variable or array if `.shape` is present.

    Args:
        val (AttributeValue): Variable or array.

    Returns:
        tuple[int, ...]: Extracted shape tuple.
    """
    return tuple(int(d) for d in (val.shape if isinstance(val, _ShapedVariableProtocol) else ()))


def graph_to_json(graph: IRGraph) -> str:
    """Convert graph to JSON.

    Args:
        graph (IRGraph): Input `graph` argument (graph).

    Returns:
        str: JSON representation of the graph.
    """
    return str(graph.to_json())
