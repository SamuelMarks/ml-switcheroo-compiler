"""Domain operation registration tables for custom autodiff and structural tuple primitives."""

from __future__ import annotations

import typing

from .coercion import coerce_int, coerce_operand_list, coerce_opt_int
from .registry_core import EagerOpRegistry, global_eager_registry
from .types_base import BackendArg


class EagerTableHandlerProtocol(typing.Protocol):
    """Protocol for eager operation table handlers accepting `(backend_module, *args, **kwargs)`."""

    def __call__(self, backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an eager operation handler.

        Args:
            backend_module (BackendArg): Active backend runtime module.
            *args (BackendArg): Positional operands.
            **kwargs (BackendArg): Keyword attributes.

        Returns:
            BackendArg: Computed eager output.
        """
        ...


def _eager_custom_vjp(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Evaluate forward primal output of a `CustomVJP` operation in eager mode.

    Args:
        backend_module (BackendArg): Active backend runtime module.
        *args (BackendArg): Primal input operands.
        **kwargs (BackendArg): Custom VJP metadata attributes.

    Returns:
        BackendArg: Single primal output or tuple of primal outputs.
    """
    del backend_module, kwargs
    return args[0] if len(args) == 1 else tuple(args)


def _eager_custom_jvp(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Evaluate forward primal output of a `CustomJVP` operation in eager mode.

    Args:
        backend_module (BackendArg): Active backend runtime module.
        *args (BackendArg): Primal input operands.
        **kwargs (BackendArg): Custom JVP attributes including optional `'fun'` callback.

    Returns:
        BackendArg: Evaluated forward result or primal tuple.
    """
    del backend_module
    fun = kwargs.get("fun")
    if callable(fun):
        from ml_switcheroo_compiler.core.config import ConfigContext

        with ConfigContext(eager_mode=True):
            return fun(*args)
    return args[0] if len(args) == 1 else tuple(args)


def _eager_process_custom_vjp_call(
    backend_module: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute a custom backward VJP rule callback in eager mode.

    Args:
        backend_module (BackendArg): Active backend runtime module.
        *args (BackendArg): Cotangent operands.
        **kwargs (BackendArg): Attributes containing `'bwd_fn'`.

    Returns:
        BackendArg: Computed backward gradients or `None`.
    """
    del backend_module
    bwd_fn = kwargs["bwd_fn"]
    if callable(bwd_fn):
        return bwd_fn(None, *args)
    return None


def _eager_process_custom_jvp_call(
    backend_module: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute a custom forward JVP rule callback in eager mode.

    Args:
        backend_module (BackendArg): Active backend runtime module.
        *args (BackendArg): Concatenated primal and tangent operands.
        **kwargs (BackendArg): Attributes containing `'jvp_rule'` and `'num_primals'`.

    Returns:
        BackendArg: Pair `(primal_out, tangent_out)`.
    """
    del backend_module
    jvp_rule = kwargs.get("jvp_rule")
    num_primals = coerce_opt_int(kwargs.get("num_primals", len(args) // 2))
    primals = args[:num_primals]
    tangents = args[num_primals:]
    if callable(jvp_rule):
        from ml_switcheroo_compiler.core.config import ConfigContext

        with ConfigContext(eager_mode=True):
            jvp_fn = jvp_rule
            try:
                res: BackendArg = jvp_fn(primals, tangents)
            except TypeError:
                res = jvp_fn(*primals, *tangents)
        if isinstance(res, (tuple, list)) and len(res) == 2:
            return res
        return (None, res)
    return (None, None)


def _eager_tuple_get_item(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Extract an element from a tuple or list operand by integer `'index'` in eager mode.

    Args:
        backend_module (BackendArg): Active backend runtime module.
        *args (BackendArg): Sequence operand tuple.
        **kwargs (BackendArg): Attributes containing `'index'`.

    Returns:
        BackendArg: Indexed element or `None` if operand is not a tuple/list.
    """
    del backend_module
    index: int = coerce_int(kwargs.get("index", 0))
    if isinstance(args[0], (tuple, list)):
        items: list[BackendArg] = coerce_operand_list(args[0])
        return items[index]
    return None


CORE_EAGER_OP_TABLE: dict[str, EagerTableHandlerProtocol] = {
    "CustomVJP": _eager_custom_vjp,
    "CustomJVP": _eager_custom_jvp,
    "ProcessCustomVJPCall": _eager_process_custom_vjp_call,
    "ProcessCustomJVPCall": _eager_process_custom_jvp_call,
    "TupleGetItem": _eager_tuple_get_item,
}


def register_core_eager_ops(registry: EagerOpRegistry) -> None:
    """Bind all `CORE_EAGER_OP_TABLE` handlers onto `registry`.

    Args:
        registry (EagerOpRegistry): Target eager operation registry instance.
    """
    for op_name, handler in CORE_EAGER_OP_TABLE.items():
        registry.register(op_name)(handler)


register_core_eager_ops(global_eager_registry)
