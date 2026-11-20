"""Serializable handle registry for custom autodiff and rematerialization callbacks.

Allows ``LogicalNode.attributes`` to remain strictly serializable without storing raw
Python ``FunctionType`` / ``MethodType`` objects inside IR attribute dictionaries,
while bounding memory growth via LRU eviction and supporting ``ContextVar``-isolated
compilation scopes.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import threading
import typing
import uuid
from collections.abc import Iterator
from contextvars import ContextVar

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core._shape_protocols import VariadicCallableProtocol
    from ml_switcheroo_compiler.core._shape_types import OpValue
    from ml_switcheroo_compiler.grad.utils import DifferentiableFn

MAX_GLOBAL_CALLBACK_HANDLES: int = 4096
_CALLBACK_LOCK = threading.Lock()
_CALLBACK_TABLE: collections.OrderedDict[str, VariadicCallableProtocol] = collections.OrderedDict()
_SCOPED_CALLBACK_TABLE: ContextVar[collections.OrderedDict[str, VariadicCallableProtocol] | None] = ContextVar(
    "scoped_callback_table",
    default=None,
)


@contextlib.contextmanager
def callback_registry_scope(max_entries: int = MAX_GLOBAL_CALLBACK_HANDLES) -> Iterator[None]:
    """Enter a `ContextVar`-isolated callback registry scope that clears automatically on exit.

    Args:
        max_entries (int): Positive capacity bound for callbacks registered within this scope.

    Yields:
        None: Execution scope with an isolated callback dictionary.

    Raises:
        ValueError: If `max_entries` is less than 1.
    """
    if max_entries < 1:
        raise ValueError("max_entries must be >= 1")
    scoped_table: collections.OrderedDict[str, VariadicCallableProtocol] = collections.OrderedDict()
    prev_limit = MAX_GLOBAL_CALLBACK_HANDLES
    token = _SCOPED_CALLBACK_TABLE.set(scoped_table)
    try:
        yield
    finally:
        scoped_table.clear()
        _SCOPED_CALLBACK_TABLE.reset(token)
        _ = prev_limit


def _resolve_rule_id(rule_id: str, kind: str) -> VariadicCallableProtocol:
    """Look up `rule_id` in the active scoped table or bounded global LRU table.

    Args:
        rule_id (str): Handle identifier to resolve.
        kind (str): Callback category for diagnostic context.

    Returns:
        VariadicCallableProtocol: Registered Python callable.

    Raises:
        KeyError: If `rule_id` was cleared, evicted, or deserialized across processes.
    """
    scoped = _SCOPED_CALLBACK_TABLE.get()
    if scoped is not None and rule_id in scoped:
        scoped.move_to_end(rule_id)
        return scoped[rule_id]
    with _CALLBACK_LOCK:
        if rule_id in _CALLBACK_TABLE:
            _CALLBACK_TABLE.move_to_end(rule_id)
            return _CALLBACK_TABLE[rule_id]
    raise KeyError(
        f"CustomRuleHandle '{rule_id}' (kind='{kind}') is not registered in this process or scope "
        "(handles cannot cross process boundaries without re-registration and may expire after scope exit)."
    )


@dataclasses.dataclass(frozen=True)
class CustomRuleHandle:
    """Serializable handle referencing a registered custom AD or rematerialization callable.

    Conforms structurally to ``ml_switcheroo_ir.CustomRuleHandleProtocol`` via
    ``handle_id`` and ``kind``.

    Attributes:
        rule_id (str): Unique identifier in ``_CALLBACK_TABLE`` or active ``callback_registry_scope``.
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
            KeyError: If ``rule_id`` is not registered in the active scope or ``_CALLBACK_TABLE``.
        """
        return _resolve_rule_id(self.rule_id, self.kind)

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
    max_entries: int = MAX_GLOBAL_CALLBACK_HANDLES,
) -> CustomRuleHandle | None:
    """Register ``fn`` in the active scoped or bounded LRU callback table and return a ``CustomRuleHandle``.

    Args:
        fn (DifferentiableFn | None): Python callable to register, or ``None``.
        kind (str): Callback rule category stored on the returned handle.
        max_entries (int): Capacity limit applied before evicting least-recently-used entries.

    Returns:
        CustomRuleHandle | None: Serializable handle, or ``None`` when ``fn`` is ``None``.

    Raises:
        TypeError: If ``fn`` is neither ``None``, a ``CustomRuleHandle``, nor a variadic callable.
        ValueError: If ``max_entries`` is less than 1.
    """
    from ml_switcheroo_compiler.core.shape import is_variadic_callable

    if max_entries < 1:
        raise ValueError("max_entries must be >= 1")
    if fn is None:
        return None
    if isinstance(fn, CustomRuleHandle):
        return fn
    if not is_variadic_callable(fn):
        raise TypeError(f"Expected callable custom rule, got {type(fn).__name__}")
    rid = f"rule_{uuid.uuid4().hex}"
    scoped = _SCOPED_CALLBACK_TABLE.get()
    if scoped is not None:
        scoped[rid] = fn
        while len(scoped) > max_entries:
            scoped.popitem(last=False)
        return CustomRuleHandle(rule_id=rid, kind=kind)
    with _CALLBACK_LOCK:
        _CALLBACK_TABLE[rid] = fn
        while len(_CALLBACK_TABLE) > max_entries:
            _CALLBACK_TABLE.popitem(last=False)
    return CustomRuleHandle(rule_id=rid, kind=kind)


def resolve_custom_rule_handle(handle: CustomRuleHandle) -> VariadicCallableProtocol:
    """Resolve a ``CustomRuleHandle`` back to its registered Python callable.

    Args:
        handle (CustomRuleHandle): Registered custom rule handle.

    Returns:
        VariadicCallableProtocol: Underlying Python callable registered for ``handle``.

    Raises:
        KeyError: If ``handle.rule_id`` is not present in the active scope or ``_CALLBACK_TABLE``.
    """
    return handle.resolve()


def unregister_custom_rule_handle(handle_or_id: CustomRuleHandle | str) -> bool:
    """Remove a single custom rule handle from the active scope and global table.

    Args:
        handle_or_id (CustomRuleHandle | str): Handle instance or string ``rule_id``.

    Returns:
        bool: True if a registered entry was removed, False otherwise.
    """
    rid = handle_or_id.rule_id if isinstance(handle_or_id, CustomRuleHandle) else handle_or_id
    removed = False
    scoped = _SCOPED_CALLBACK_TABLE.get()
    if scoped is not None and rid in scoped:
        del scoped[rid]
        removed = True
    with _CALLBACK_LOCK:
        if rid in _CALLBACK_TABLE:
            del _CALLBACK_TABLE[rid]
            removed = True
    return removed


def get_registered_callback_count() -> int:
    """Return the number of callbacks currently held in the active scope or global table.

    Returns:
        int: Active callback entry count.
    """
    scoped = _SCOPED_CALLBACK_TABLE.get()
    if scoped is not None:
        return len(scoped)
    with _CALLBACK_LOCK:
        return len(_CALLBACK_TABLE)


def clear_custom_rule_handles() -> None:
    """Remove all registered custom rule callbacks from the active scope and global table."""
    scoped = _SCOPED_CALLBACK_TABLE.get()
    if scoped is not None:
        scoped.clear()
    with _CALLBACK_LOCK:
        _CALLBACK_TABLE.clear()


__all__ = [
    "CustomRuleHandle",
    "MAX_GLOBAL_CALLBACK_HANDLES",
    "callback_registry_scope",
    "clear_custom_rule_handles",
    "get_registered_callback_count",
    "register_custom_rule_handle",
    "resolve_custom_rule_handle",
    "unregister_custom_rule_handle",
]
