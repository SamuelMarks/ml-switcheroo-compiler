"""Declarative AST template formatting and generic fallback mixin for BaseGenerator."""

from __future__ import annotations

import logging
import re
import typing

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.formatters import FormatterContext, OpFormatter


class _AssignVarNameProto(typing.Protocol):
    """Protocol for ``assign_var_name`` method on generator mixins."""

    def __call__(self, node_id: str, prefix: str = "tensor") -> str:
        """Assign variable name for ``node_id``.

        Args:
            node_id (str): Logical node identifier.
            prefix (str): Variable name prefix.

        Returns:
            str: Assigned variable name.
        """
        ...


class BaseGeneratorTemplateMixin:
    """Mixin providing declarative YAML op mapping loading, AST templates, and generic fallback."""

    backend_name: str | None
    assign_var_name: _AssignVarNameProto

    def _load_op_mapping_info(
        self,
        backend_name: str,
        op_name: str,
    ) -> tuple[str | None, dict[str, str | None], str | None]:
        """Load AST template, keyword mapping, and target API for an operation.

        Args:
            backend_name (str): Target backend name.
            op_name (str): Operation type name.

        Returns:
            tuple[str | None, dict[str, str | None], str | None]: Tuple of (ast_template, kwarg_map, target_api).
        """
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        try:
            schema = load_backend_mappings(backend_name)
            if op_name in schema.operations:
                op_mapping = schema.operations[op_name]
                return (
                    op_mapping.ast_template,
                    dict(op_mapping.kwarg_map or op_mapping.kwarg_translations),
                    op_mapping.target_api,
                )
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return None, {}, None

    def _translate_node_kwargs(
        self,
        all_attrs: dict[str, BackendArg],
        kwarg_map: dict[str, str | None],
    ) -> dict[str, BackendArg]:
        """Translate node attributes according to backend keyword mappings.

        Args:
            all_attrs (dict[str, BackendArg]): Raw node and call attributes.
            kwarg_map (dict[str, str | None]): Mapping from source kwarg to target kwarg.

        Returns:
            dict[str, BackendArg]: Translated keyword arguments dictionary.
        """
        translated_kwargs: dict[str, BackendArg] = {}
        for k, v in all_attrs.items():
            if k in kwarg_map:
                target_k = kwarg_map[k]
                if target_k is not None:
                    translated_kwargs[target_k] = v
            else:
                translated_kwargs[k] = v
        return translated_kwargs

    def _format_ast_template(
        self,
        ast_template: str,
        node: IRNode,
        input_vars: list[str],
        all_attrs: dict[str, BackendArg],
        translated_kwargs: dict[str, BackendArg],
    ) -> str:
        """Substitute variables and attributes into an AST template string.

        Args:
            ast_template (str): Template string with placeholders.
            node (IRNode): Target IR node.
            input_vars (list[str]): Input variable names.
            all_attrs (dict[str, BackendArg]): Combined node attributes.
            translated_kwargs (dict[str, BackendArg]): Translated keyword arguments.

        Returns:
            str: Formatted code string.
        """
        out_var = str(all_attrs.get("out_var") or "")
        subs: dict[str, str] = {
            "out": out_var or self.assign_var_name(node.id),
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

    def get_ops_map(self, kwargs: dict[str, BackendArg] | None = None) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs (dict[str, BackendArg] | None): Optional operation keyword arguments.

        Returns:
            dict[str, str]: Dictionary mapping operation type to format string.
        """
        del kwargs
        from ml_switcheroo_compiler.ops.registry import backend_mapping_registry

        ops: dict[str, str] = {}
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

    def get_fallback_keepdims_kwarg(self) -> str:
        """Get the fallback keepdims keyword argument name.

        Returns:
            str: The keepdims keyword.
        """
        return "keepdims"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Fallback visit method for operations not explicitly handled.

        Args:
            node (IRNode): The node.
            input_vars (list[str]): The inputs.
            **kwargs (BackendArg): Additional attributes.

        Returns:
            str: The code string.
        """
        op_type = node.op_type
        ops_map = self.get_ops_map(kwargs)
        if op_type in ops_map:
            fmt = ops_map[op_type]
            fmt = OpFormatter.format_backend_string(fmt, input_vars, kwargs)
            fmt = re.sub(r", \w+=\{[^\}]+\}", "", fmt)
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
