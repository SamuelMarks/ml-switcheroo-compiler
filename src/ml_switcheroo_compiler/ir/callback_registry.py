"""Serializable handle registry for custom autodiff and rematerialization callbacks.

Allows ``LogicalNode.attributes`` to remain strictly serializable without storing raw
Python ``FunctionType`` / ``MethodType`` objects inside IR attribute dictionaries.
"""

from __future__ import annotations

import dataclasses
import typing
import uuid

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core._shape_protocols import VariadicCallableProtocol
    from ml_switcheroo_compiler.core._shape_types import OpValue
    from ml_switcheroo_compiler.grad.utils import DifferentiableFn

_CALLBACK_TABLE: dict[str, VariadicCallableProtocol] = {}


@dataclasses.dataclass(frozen=True)
class CustomRuleHandle:
    """Serializable handle referencing a registered custom AD or rematerialization callable.

    Conforms structurally to ``ml_switcheroo_ir.CustomRuleHandleProtocol`` via
    ``handle_id`` and ``kind``.

    Attributes:
        rule_id (str): Unique identifier in ``_CALLBACK_TABLE``.
        kind (str): Callback rule category (default ``'custom_rule'``).
    """

    rule_id: str
    kind: str = "custom_rule"

    @property
    def handle_id(self) -> str:
        """Return unique callback handle identifier for ``CustomRuleHandleProtocol``.

        Returns:
            str: Registered rule identifier matching ``rule_id``.
        """
        return self.rule_id

    def resolve(self) -> VariadicCallableProtocol:
        """Return the underlying registered Python callable.

        Returns:
            VariadicCallableProtocol: Registered custom AD or rematerialization function.

        Raises:
            KeyError: If ``rule_id`` is not registered in ``_CALLBACK_TABLE``.
        """
        return _CALLBACK_TABLE[self.rule_id]

    def __call__(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Invoke the underlying registered callable directly.

        Args:
            *args (OpValue): Positional arguments forwarded to the rule.
            **kwargs (OpValue): Keyword arguments forwarded to the rule.

        Returns:
            OpValue: Result returned by the underlying callable.
        """
        fn = self.resolve()
        return fn(*args, **kwargs)


def register_custom_rule_handle(
    fn: DifferentiableFn | None,
    kind: str = "custom_rule",
) -> CustomRuleHandle | None:
    """Register ``fn`` in the process callback table and return a serializable ``CustomRuleHandle``.

    Args:
        fn (DifferentiableFn | None): Python callable to register, or ``None``.
        kind (str): Callback rule category stored on the returned handle.

    Returns:
        CustomRuleHandle | None: Serializable handle, or ``None`` when ``fn`` is ``None``.

    Raises:
        TypeError: If ``fn`` is neither ``None``, a ``CustomRuleHandle``, nor a variadic callable.
    """
    from ml_switcheroo_compiler.core.shape import is_variadic_callable

    if fn is None:
        return None
    if isinstance(fn, CustomRuleHandle):
        return fn
    if not is_variadic_callable(fn):
        raise TypeError(f"Expected callable custom rule, got {type(fn).__name__}")
    rid = f"rule_{uuid.uuid4().hex}"
    _CALLBACK_TABLE[rid] = fn
    return CustomRuleHandle(rule_id=rid, kind=kind)


def resolve_custom_rule_handle(handle: CustomRuleHandle) -> VariadicCallableProtocol:
    """Resolve a ``CustomRuleHandle`` back to its registered Python callable.

    Args:
        handle (CustomRuleHandle): Registered custom rule handle.

    Returns:
        VariadicCallableProtocol: Underlying Python callable registered for ``handle``.

    Raises:
        KeyError: If ``handle.rule_id`` is not present in ``_CALLBACK_TABLE``.
    """
    return handle.resolve()


def clear_custom_rule_handles() -> None:
    """Remove all registered custom rule callbacks from the in-memory table."""
    _CALLBACK_TABLE.clear()


__all__ = [
    "CustomRuleHandle",
    "clear_custom_rule_handles",
    "register_custom_rule_handle",
    "resolve_custom_rule_handle",
]
