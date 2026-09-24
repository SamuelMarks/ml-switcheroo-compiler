# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define base generator for emitting backend code from IR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.device import Device

from ml_switcheroo_compiler.backends.formatters import CodeFormatter, FormatterContext, OpFormatter
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

from .generator_mixins import EagerExecutionMixin, GeneratorLifecycleMixin


@dataclass
class CompiledArtifact:
    """Standardized ahead-of-time (AOT) compiled executable artifact container.

    Attributes:
        callable_fn: Executable callable function or entry point runner.
        binary_bytes: Raw compiled binary bytes (e.g. PTX, HSACO, metallib, bytecode).
        source_code: Emitted source code in the target backend language.
        metadata: Backend-specific compilation metadata, options, and schemas.
        temp_dir: Optional temporary directory managing compiled build artifacts.
    """

    callable_fn: Any | None = None
    binary_bytes: bytes = b""
    source_code: str = ""
    metadata: dict[str, Any] | None = None
    temp_dir: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the compiled artifact.

        Args:
            *args: Positional arguments to pass to the compiled callable.
            **kwargs: Keyword arguments to pass to the compiled callable.

        Returns:
            Any: Computation output or tuple of outputs.

        Raises:
            RuntimeError: If no executable callable_fn is bound to this artifact.
        """
        if self.callable_fn is None:
            raise RuntimeError("CompiledArtifact has no executable callable bound.")
        return self.callable_fn(*args, **kwargs)

    def cleanup(self) -> None:
        """Clean up associated temporary directory resources if active."""
        if self.temp_dir is not None and hasattr(self.temp_dir, "cleanup"):
            self.temp_dir.cleanup()
            self.temp_dir = None

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style access to metadata and artifact attributes.

        Args:
            item (str): Key name to retrieve.

        Returns:
            Any: Value from metadata dictionary or artifact attributes.

        Raises:
            KeyError: If key is not found in metadata or attributes.
        """
        if self.metadata is not None and item in self.metadata:
            return self.metadata[item]
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


class IRGraphWalker:
    """Help class to encapsulate IR graph traversal logic."""

    def __init__(self, generator: BaseGenerator) -> None:
        """Initialize the walker.

        Args:
            generator (BaseGenerator): The generator instance.
        """
        self.generator = generator

    def walk(self, input_prefix: str = "args") -> None:
        """Walk the graph and generate code.

        Args:
            input_prefix (str): The prefix for input args.
        """
        visitor = CodeGeneratorVisitor(self.generator)
        visitor.generate_body(input_prefix)


@dataclass
class InputContext:
    """Provide context for input assignment."""

    var_name: str
    node: IRNode
    input_prefix: str
    input_idx: int


class FormatterProxyMixin:
    """FormatterProxyMixin class."""

    """Provide mixin for proxying formatter methods."""

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


class EmitUtilsMixin:
    """EmitUtilsMixin class."""

    """Provide mixin for emit utilities."""

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
        elif hasattr(self, "graph") and getattr(self.graph, "outputs", None):
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


class class_or_instance_method:
    """Descriptor supporting invocation on either a class or an instance."""

    def __init__(self, fn: object) -> None:
        """Initialize descriptor with wrapped function.

        Args:
            fn (object): The callable method to wrap.
        """
        self.fn = fn

    def __get__(self, obj: object, cls: type | None = None) -> object:
        """Bind method to instance if present, else to class.

        Args:
            obj (object): Target instance or None.
            cls (type | None): Owning class.

        Returns:
            object: Bound callable.
        """
        target = obj if obj is not None else cls
        return lambda *args, **kwargs: self.fn(target, *args, **kwargs)


class BaseGenerator(FormatterProxyMixin, EmitUtilsMixin, GeneratorLifecycleMixin, EagerExecutionMixin):
    """Abstract base class for backend code generation."""

    def __init__(self, graph: IRGraph, delegates=None) -> None:
        """Initialize the object.

        Args:
            graph (IRGraph): The graph to process.
            delegates: The visitor delegates.
        """
        self.graph = graph
        self.sorted_nodes = topological_sort(graph)
        self.formatter = CodeFormatter()
        self.visitors = [self] + (delegates or [])

    def get_language(self) -> str:
        """Get the target programming language for code emission.

        Returns:
            str: Target language string ('python').
        """
        return "python"

    def emit_constant(self, node: IRNode) -> str:
        """Emit code for the constant backend.

        Args:
            node (IRNode): The node to process.

        Returns:
            str: The computed result.
        """
        val = node.attributes.get("value")
        return repr(val)

    def visit_OpNode(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generic visitor for IRNode driven by declarative mapping YAML templates.

        Args:
            node (IRNode): Target IRNode.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional node attributes and context.

        Returns:
            str: Generated code line or expression.
        """
        backend_name = getattr(self, "backend_name", None) or self.get_fallback_prefix()
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        op_name = getattr(node, "op_type", "")
        ast_template = None
        kwarg_map: dict[str, str | None] = {}
        target_api = None

        try:
            schema = load_backend_mappings(backend_name)
            if op_name in schema.operations:
                op_mapping = schema.operations[op_name]
                ast_template = op_mapping.ast_template
                kwarg_map = dict(op_mapping.kwarg_map or op_mapping.kwarg_translations)
                target_api = op_mapping.target_api
        except Exception:
            pass

        all_attrs: dict[str, object] = dict(getattr(node, "attributes", {}) or {})
        all_attrs.update(kwargs)

        translated_kwargs: dict[str, object] = {}
        for k, v in all_attrs.items():
            if k in kwarg_map:
                target_k = kwarg_map[k]
                if target_k is not None:
                    translated_kwargs[target_k] = v
            else:
                translated_kwargs[k] = v

        if ast_template:
            out_var = str(all_attrs.get("out_var") or "")
            subs: dict[str, str] = {
                "out": out_var or self.assign_var_name(getattr(node, "id", "")),
            }
            for i, iv in enumerate(input_vars):
                subs[f"in{i}"] = iv
            subs["inputs"] = ", ".join(input_vars)
            for k, v in translated_kwargs.items():
                subs[k] = str(v)

            result = ast_template
            for k, val in subs.items():
                result = result.replace(f"{{{k}}}", str(val))
            return result

        if target_api:
            call_args = list(input_vars)
            kw_pairs = [f"{k}={v}" for k, v in translated_kwargs.items() if not str(k).startswith("_")]
            call_str = ", ".join(call_args + kw_pairs)
            return f"{target_api}({call_str})"

        return self.generic_visit(node, input_vars, **kwargs)

    def visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Visit a node and return the formatted code string for the operation.

        Args:
            node (IRNode): The node parameter for the operation.
            input_vars (list[str]): The input_vars parameter for the operation.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The computed result.
        """
        op_type = getattr(node, "op_type", "")
        method_name = f"visit_{op_type}"
        for visitor in getattr(self, "visitors", []):
            if hasattr(visitor, method_name):
                method = getattr(visitor, method_name)
                return method(node, input_vars, **kwargs)
        if hasattr(self, method_name) and method_name != "visit_OpNode":
            method = getattr(self, method_name)
            return method(node, input_vars, **kwargs)
        if hasattr(self, "visit_OpNode"):
            return self.visit_OpNode(node, input_vars, **kwargs)
        return self.generic_visit(node, input_vars, **kwargs)

    def get_ops_map(self, kwargs) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        from ml_switcheroo_compiler.ops.registry import backend_mapping_registry

        ops = {}
        prefix = self.get_fallback_prefix()
        for op_name in backend_mapping_registry.operations.keys():
            fmt = backend_mapping_registry.get_generator_mapping(prefix, op_name)
            if fmt is not None:
                ops[op_name] = fmt

        if "OverlapAndAdd" not in ops:
            ops["OverlapAndAdd"] = "tf.signal.overlap_and_add({0})"

        return ops

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
            str: The prefix.
        """
        return "np"

    def get_fallback_axis_kwarg(self) -> str:
        """Get the fallback axis keyword argument name.

        Returns:
            str: The axis keyword.
        """
        return "axis"

    @class_or_instance_method
    def compile_aot(self, graph: IRGraph | None = None, **kwargs: object) -> object:
        """Compile an IRGraph into an ahead-of-time binary artifact or callable executable.

        Can be invoked either as an instance method (e.g. generator.compile_aot())
        or as a class method (e.g. GeneratorClass.compile_aot(graph)).

        Args:
            graph (IRGraph | None): Target computation graph to compile.
            **kwargs (object): Compiler options, optimization levels, or destination paths.

        Returns:
            object: Compiled binary artifact, path, or callable execution wrapper.

        Raises:
            NotImplementedError: If AOT compilation is not implemented for this backend.
        """
        target_graph = graph if graph is not None else getattr(self, "graph", None)
        if target_graph is None:
            target_graph = IRGraph()

        if isinstance(self, type):
            instance = self(target_graph)
            return instance.compile_aot(target_graph, **kwargs)

        if not hasattr(self, "_aot_cache") or getattr(self, "_aot_cache", None) is None:
            self._aot_cache: dict[tuple[int, tuple[tuple[str, str], ...]], object] = {}

        cache_key = (id(target_graph), tuple(sorted((str(k), str(v)) for k, v in kwargs.items())))
        if cache_key in self._aot_cache:
            return self._aot_cache[cache_key]

        artifact = self._compile_aot_impl(target_graph, **kwargs)
        self._aot_cache[cache_key] = artifact
        return artifact

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Internal backend-specific AOT compilation hook.

        Args:
            graph (IRGraph): The computation graph to compile.
            **kwargs (object): Backend-specific compiler arguments.

        Returns:
            object: Resulting compiled artifact or callable.
        """
        msg = f"AOT compilation is not implemented for {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def get_fallback_keepdims_kwarg(self) -> str:
        """Get the fallback keepdims keyword argument name.

        Returns:
            str: The keepdims keyword.
        """
        return "keepdims"

    @classmethod
    def get_logical_devices(cls, device_type: str | None = None) -> list[Device]:
        """Discover logical devices available for this backend.

        Args:
            device_type (str | None): Optional device type filter (e.g., 'cpu', 'gpu', 'webgpu').

        Returns:
            list[Device]: Available logical devices.
        """
        from ml_switcheroo_compiler.core.device import Device, DeviceType

        devices: list[Device] = [Device(DeviceType.CPU, 0)]
        cls_name = cls.__name__.lower()
        if "webgpu" in cls_name or "wgsl" in cls_name:
            devices.append(Device(DeviceType.WEBGPU, 0))
        elif "cuda" in cls_name or "gpu" in cls_name or "rocm" in cls_name or "metal" in cls_name:
            devices.append(Device(DeviceType.GPU, 0))

        if device_type is not None:
            dtype_lower = str(device_type).lower()
            return [d for d in devices if d.device_type.value == dtype_lower or (dtype_lower in ("cuda", "rocm", "metal") and d.device_type == DeviceType.GPU)]
        return devices

    @classmethod
    def get_physical_devices(cls, device_type: str | None = None) -> list[Device]:
        """Discover physical hardware accelerators available for this backend.

        Args:
            device_type (str | None): Optional device type filter.

        Returns:
            list[Device]: Available physical devices.
        """
        return cls.get_logical_devices(device_type)

    @classmethod
    def get_memory_info(cls, device: str | None = None) -> dict[str, int]:
        """Retrieve memory allocation statistics for the specified device.

        Args:
            device (str | None): Optional target device identifier.

        Returns:
            dict[str, int]: Dictionary containing 'current' and 'peak' memory in bytes.
        """
        del device
        return {"current": 0, "peak": 0}

    @classmethod
    def initialize_distributed(cls, *args: object, **kwargs: object) -> None:
        """Initialize distributed execution context for multi-device operations.

        Args:
            *args (object): Distributed initialization parameters.
            **kwargs (object): Additional distributed execution options.
        """
        del args, kwargs

    @classmethod
    def export_function(cls, *args: object, **kwargs: object) -> None:
        """Export a computational graph or trace for unified persistence.

        Args:
            *args (object): Export arguments.
            **kwargs (object): Export configuration parameters.
        """
        del args, kwargs

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs) -> str:
        """Fallback visit method for operations not explicitly handled.

        Args:
            node (IRNode): The node.
            input_vars (list[str]): The inputs.
            **kwargs: Additional attributes.

        Returns:
            str: The code string.
        """
        op_type = getattr(node, "op_type", "")
        ops_map = self.get_ops_map(kwargs)
        if op_type in ops_map:
            fmt = ops_map[op_type]
            fmt = OpFormatter.format_backend_string(fmt, input_vars, kwargs)
            fmt = re.sub(", \\w+=\\{[^\\}]+\\}", "", fmt)
            return fmt

        ctx = FormatterContext(
            prefix=self.get_fallback_prefix(),
            op_type=op_type,
            input_vars=input_vars,
            kwargs=kwargs,
            axis_kwarg=self.get_fallback_axis_kwarg(),
            keepdims_kwarg=self.get_fallback_keepdims_kwarg(),
        )
        return OpFormatter.format_generic_fallback(ctx)

    def _emit_input_assignment(self, var_name: str, node: IRNode, input_prefix: str, input_idx: int) -> None:
        """Override in subclasses to handle custom input logic (e.g. keras.Input).

        Args:
            var_name (str): The var_name parameter for the operation.
            node (IRNode): The node parameter for the operation.
            input_prefix (str): The input_prefix parameter for the operation.
            input_idx (int): The input_idx parameter for the operation.
        """
        self.add_line(f"{var_name} = {input_prefix}[{input_idx}]")

    def _emit_output_assignment(self, node: IRNode, input_vars: list[str], returns: str) -> None:
        """Override in subclasses to handle custom output logic.

        Args:
            node (IRNode): The node parameter for the operation.
            input_vars (list[str]): The input_vars parameter for the operation.
            returns (str): The returns parameter for the operation.
        """
        if not hasattr(self, "_output_returns"):
            self._output_returns = []
        self._output_returns.append(returns)


class PythonStringGenerator(BaseGenerator):
    """Provide mixin for python string generators to avoid DRY issues in generate()."""

    _import_header: str | tuple[str, ...] = ""
    _func_name: str = "evaluate"

    def generate(self) -> str:
        """Generate the complete script.

        Returns:
            str: The generated script.
        """
        self.code = [self.header]
        if isinstance(self._import_header, str):
            self.add_line(self._import_header)
        elif isinstance(self._import_header, (tuple, list)):
            self.add_line("\n".join(self._import_header))
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)


class ClassBasedGenerator(BaseGenerator):
    """ClassBasedGenerator class."""

    """Provide mixin for class-based string generators to avoid DRY issues in generate()."""

    _forward_method_name: str = "forward"
    _base_class_name: str = ""

    def _get_prefix_code(self) -> list[str]:
        """Return the code to be inserted before the class definition.

        Returns:
            list[str]: The prefix code lines.
        """
        return []

    def _emit_init_body(self) -> bool:
        """Emit initialization code.

        Returns:
            bool: True if params were emitted, False otherwise.
        """
        return False

    def generate(self) -> str:
        """Generate the complete script.

        Returns:
            str: The generated script.
        """
        self.code = [self.header]
        self.code.extend(self._get_prefix_code())
        base_class = f"({self._base_class_name})" if self._base_class_name else ""
        self.add_line(f"class CompiledModel{base_class}:")
        self.indent_level = 1
        self.add_line("def __init__(self, *args, **kwargs) -> None:")
        self.indent_level += 1
        if self._base_class_name:
            self.add_line("super().__init__()")
        has_params = self._emit_init_body()
        if not has_params:
            self.add_line("pass" if self.get_language() == "python" else "")
        self.add_line("")
        self.indent_level -= 1
        self.add_line(f"def {self._forward_method_name}(self, *args, **kwargs):")
        self.indent_level += 1
        self._generate_body()
        return "\n".join(self.code)
