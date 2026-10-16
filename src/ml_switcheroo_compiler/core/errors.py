"""Error classes for the ml-switcheroo compiler."""

from __future__ import annotations

import os
from typing import Optional

import yaml

_ERROR_TEMPLATES: dict[str, str] = {}


def _load_error_templates() -> None:
    """Load error templates from yaml.

    Returns:
        None: Result.
    """
    global _ERROR_TEMPLATES
    if not _ERROR_TEMPLATES:
        yaml_path = os.path.join(os.path.dirname(__file__), "error_messages.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                _ERROR_TEMPLATES = yaml.safe_load(f) or {}


class CompilerError(RuntimeError):
    """Define single authoritative root exception for all ml-switcheroo errors."""

    def __init__(
        self,
        message: str = "",
        node_id: Optional[str] = None,
        op_type: Optional[str] = None,
        backend_target: Optional[str] = None,
        trace_context: Optional[str] = None,
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize the base compiler exception.

        Args:
            message: The explicit message.
            node_id: Optional node ID associated with the error.
            op_type: Optional operation type.
            backend_target: Optional backend target.
            trace_context: Optional trace context.
            **kwargs: Values to inject into predefined error templates.
        """
        _load_error_templates()
        template = _ERROR_TEMPLATES.get(self.__class__.__name__) or _ERROR_TEMPLATES.get("SwitcherooError")
        if template and kwargs and not message:
            try:
                message = template.format(**kwargs)
            except KeyError:
                message = template

        self.node_id = node_id
        self.op_type = op_type
        self.backend_target = backend_target
        self.trace_context = trace_context

        parts = [message]
        if node_id:
            parts.append(f"Node ID: {node_id}")
        if op_type:
            parts.append(f"Op Type: {op_type}")
        if backend_target:
            parts.append(f"Backend: {backend_target}")
        if trace_context:
            parts.append(f"Context: {trace_context}")

        full_message = " | ".join(filter(None, parts))
        super().__init__(full_message)


SwitcherooError = CompilerError


class FrontendBoundaryViolationError(CompilerError):
    """Raised on Rule 1 violation detections."""


class BackendDecouplingError(CompilerError):
    """Raised on Rule 2 violation detections."""


class DependencyViolationError(CompilerError):
    """Raised on Rule 3 violation detections."""


class UnimplementedMathError(CompilerError):
    """Raised when an operation lacks a mathematical equivalent (Rule 4)."""


class AutodiffError(CompilerError):
    """Raised on Rule 5 / backward pass or VJP failure."""


class ShapeInferenceError(CompilerError):
    """Raised on dynamic/symbolic rank mismatch."""


class SnapshotGroundingError(CompilerError):
    """Raised on upstream contract mismatch."""


class TracingError(CompilerError):
    """Raised when control flow breaks proxy tensor tracing."""


class CompilationError(CompilerError):
    """Raised during IR generation or pass failure."""


class ShapeMismatchError(CompilerError):
    """Raised during static shape inference."""


class DTypePromotionError(CompilerError):
    """Raised on invalid automatic type casting."""


class BackendNotSupportedError(CompilerError):
    """Raised when an edge target lacks an op."""


class MissingJVPRuleError(CompilerError):
    """Exception raised when a required JVP/VJP rule is missing for an operation."""


class EvaluationError(CompilerError):
    """Raised during eager node evaluation failure."""


class AutodiffLinkageError(AutodiffError):
    """Raised on VJP node link extraction failure."""


class SymbolicDifferentiationError(AutodiffError):
    """Raised on symbolic AST derivative simplification failure."""
