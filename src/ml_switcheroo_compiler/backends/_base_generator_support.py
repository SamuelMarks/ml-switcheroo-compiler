"""Supporting protocols, graph walker, formatter proxy, and descriptor helpers for BaseGenerator."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Callable, Mapping

import pydantic

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core.shape import IRAttrValue as AttributeValue
from ml_switcheroo_compiler.core.shape import OpValue as OpValue
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.formatters import CodeFormatter

from .generator_mixins import GeneratorLifecycleMixin


@typing.runtime_checkable
class VisitorProtocol(typing.Protocol):
    """Protocol for AST and IR visitor delegates."""

    def __repr__(self) -> str:
        """Return string representation of visitor.

        Returns:
            str: String representation.
        """
        ...


@typing.runtime_checkable
class CleanableResourceProtocol(typing.Protocol):
    """Protocol for temporary directory or build resources supporting cleanup."""

    def cleanup(self) -> None:
        """Release or clean up associated resources."""
        ...


def _is_cleanable_resource(val: CleanableResourceProtocol | BackendArg) -> typing.TypeGuard[CleanableResourceProtocol]:
    """Check whether `val` exposes `.cleanup()`.

    Args:
        val (CleanableResourceProtocol | BackendArg): Candidate resource handle.

    Returns:
        typing.TypeGuard[CleanableResourceProtocol]: `True` when `val` exposes `.cleanup()`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, CleanableResourceProtocol)
    return found or is_mock_with_attrs(val, "cleanup")


def _resolve_visitor_method(
    target: VisitorProtocol,
    method_name: str,
) -> TypedVariadicCallable[BackendArg] | None:
    """Resolve a bound visitor method on `target` without dynamic reflection builtins.

    Args:
        target (VisitorProtocol): Visitor or generator instance.
        method_name (str): Name of the `visit_<Op>` method to look up.

    Returns:
        TypedVariadicCallable[BackendArg] | None: Bound callable if present, otherwise None.
    """
    found, raw_proto = adapt_backend_module(target).resolve_symbol(method_name)
    raw: BackendArg = raw_proto
    if not found or raw is None or not callable(raw):
        return None
    return raw


class CompiledArtifact(pydantic.BaseModel):
    """Standardized ahead-of-time (AOT) compiled executable artifact container.

    Attributes:
        callable_fn: Executable callable function or entry point runner.
        binary_bytes: Raw compiled binary bytes (e.g. PTX, HSACO, metallib, bytecode).
        source_code: Emitted source code in the target backend language.
        metadata: Backend-specific compilation metadata, options, and schemas.
        temp_dir: Optional temporary directory managing compiled build artifacts.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    callable_fn: BackendArg | None = None
    binary_bytes: bytes = b""
    source_code: str = ""
    metadata: dict[str, BackendArg] | None = None
    temp_dir: BackendArg | CleanableResourceProtocol = None

    def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute the compiled artifact.

        Args:
            *args: Positional arguments to pass to the compiled callable.
            **kwargs: Keyword arguments to pass to the compiled callable.

        Returns:
            BackendArg: Computation output or tuple of outputs.

        Raises:
            RuntimeError: If no executable callable_fn is bound to this artifact.
        """
        if self.callable_fn is None:
            raise RuntimeError("CompiledArtifact has no executable callable bound.")
        return require_callable(self.callable_fn)(*args, **kwargs)

    def cleanup(self) -> None:
        """Clean up associated temporary directory resources if active."""
        if _is_cleanable_resource(self.temp_dir):
            self.temp_dir.cleanup()
            self.temp_dir = None

    def __getitem__(self, item: str) -> BackendArg:
        """Allow dictionary-style access to metadata and artifact attributes.

        Args:
            item (str): Key name to retrieve.

        Returns:
            BackendArg: Value from metadata dictionary or artifact attributes.

        Raises:
            KeyError: If key is not found in metadata or attributes.
        """
        if self.metadata is not None and item in self.metadata:
            return self.metadata[item]
        fields: dict[str, BackendArg] = {
            "callable_fn": self.callable_fn,
            "binary_bytes": self.binary_bytes,
            "source_code": self.source_code,
            "metadata": self.metadata,
            "temp_dir": str(self.temp_dir) if isinstance(self.temp_dir, CleanableResourceProtocol) else self.temp_dir,
        }
        if item in fields:
            return fields[item]
        raise KeyError(item)


class IRGraphWalker:
    """Help class to encapsulate IR graph traversal logic."""

    def __init__(self, generator: BaseGenerator | GeneratorLifecycleMixin) -> None:
        """Initialize the walker.

        Args:
            generator (BaseGenerator | GeneratorLifecycleMixin): The generator instance.
        """
        self.generator = generator

    def walk(self, input_prefix: str = "args") -> None:
        """Walk the graph and generate code.

        Args:
            input_prefix (str): The prefix for input args.
        """
        import ml_switcheroo_compiler.backends.base_generator as _bg_mod

        visitor = _bg_mod.CodeGeneratorVisitor(self.generator)
        visitor.generate_body(input_prefix)


class InputContext(pydantic.BaseModel):
    """Provide context for input assignment."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    var_name: str
    node: IRNode
    input_prefix: str
    input_idx: int


class FormatterProxyMixin:
    """Provide mixin for proxying formatter methods."""

    formatter: CodeFormatter

    @property
    def var_names(self) -> dict[str, str]:
        """Provide proxy property for formatter var_names.

        Returns:
            dict[str, str]: The variable names map.
        """
        return self.formatter.var_names

    @var_names.setter
    def var_names(self, value: dict[str, str]) -> None:
        """Set formatter var_names.

        Args:
            value (dict[str, str]): The variable names map.
        """
        self.formatter.var_names = value

    @property
    def code(self) -> list[str]:
        """Provide proxy property for formatter code.

        Returns:
            list[str]: The generated code list.
        """
        return self.formatter.code

    @code.setter
    def code(self, value: list[str]) -> None:
        """Set formatter code.

        Args:
            value (list[str]): The generated code list.
        """
        self.formatter.code = value

    @property
    def indent_level(self) -> int:
        """Provide proxy property for formatter indent_level.

        Returns:
            int: The current indent level.
        """
        return self.formatter.indent_level

    @indent_level.setter
    def indent_level(self, value: int) -> None:
        """Set formatter indent_level.

        Args:
            value (int): The current indent level.
        """
        self.formatter.indent_level = value

    @property
    def header(self) -> str:
        """Provide proxy property for formatter header.

        Returns:
            str: The header string.
        """
        return self.formatter.header

    @header.setter
    def header(self, value: str) -> None:
        """Set formatter header.

        Args:
            value (str): The header string.
        """
        self.formatter.header = value

    def get_indent(self) -> str:
        """Get the indentation string.

        Returns:
            str: Indentation string.
        """
        return self.formatter.get_indent()

    def add_line(self, line: str) -> None:
        """Add a line of code.

        Args:
            line (str): The line to add.
        """
        self.formatter.add_line(line)

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        """Assign a variable name.

        Args:
            node_id (str): The node ID.
            prefix (str): The prefix to use.

        Returns:
            str: The assigned variable name.
        """
        return self.formatter.assign_var_name(node_id, prefix)


@typing.runtime_checkable
class _GraphHolderProtocol(typing.Protocol):
    """Protocol for generator mixins holding an `IRGraph` attribute."""

    @property
    def graph(self) -> IRGraph | None:
        """Return the associated IRGraph if present.

        Returns:
            IRGraph | None: Associated computation graph.
        """
        ...


@typing.runtime_checkable
class _NodeAttributesProtocol(typing.Protocol):
    """Protocol for IR nodes exposing `.attributes`."""

    @property
    def attributes(self) -> Mapping[str, AttributeValue] | None:
        """Return node attributes mapping.

        Returns:
            Mapping[str, AttributeValue] | None: Node attributes.
        """
        ...


def _extract_node_attributes(node: IRNode | BackendArg) -> Mapping[str, AttributeValue]:
    """Return `node.attributes` if defined on `node`, or an empty mapping.

    Args:
        node (IRNode | BackendArg): Target IR node or mock node.

    Returns:
        Mapping[str, AttributeValue]: Node attributes mapping.
    """
    if isinstance(node, _NodeAttributesProtocol) and isinstance(node.attributes, Mapping):
        return node.attributes
    return {}


class EmitUtilsMixin:
    """Provide mixin for emit utilities."""

    formatter: CodeFormatter
    add_line: Callable[[str], None]

    def _emit_body_return(self, returns: list[str]) -> None:
        """Emit the final return statement.

        Args:
            returns: List of return variable names.
        """
        if returns:
            if len(returns) == 1:
                self.add_line(f"return {returns[0]}")
            else:
                self.add_line(f"return ({', '.join(returns)})")
        elif isinstance(self, _GraphHolderProtocol) and self.graph is not None and self.graph.outputs:
            out_vars: list[str] = [self.formatter.var_names.get(out_id, out_id) for out_id in self.graph.outputs]
            if len(out_vars) == 1:
                self.add_line(f"return {out_vars[0]}")
            else:
                self.add_line(f"return ({', '.join(out_vars)})")
        else:
            self.add_line("return None")

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Override in subclasses to emit framework-specific constant arrays.

        Args:
            var_name (str): The var_name parameter for the operation.
            val_repr (str): The val_repr parameter for the operation.
        """
        self.add_line(f"{var_name} = {val_repr}")
