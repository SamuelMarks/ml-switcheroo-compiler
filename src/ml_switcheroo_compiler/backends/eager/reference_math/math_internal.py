"""Generic backend-parametric eager reference operations for `reference_math.math_internal`."""

from __future__ import annotations

import types
from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _parse_scanop_args
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_operand_list,
    global_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("PopulationCount")
def _np_population_count(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Count the number of set bits in the binary representation of each element.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x_arr = arr_mod.asarray(x)
    return arr_mod.array(
        [bin(bool(n) if type(n).__name__ in ("bool", "bool_") else n).count("1") for n in x_arr.flat]
    ).reshape(x_arr.shape)


@global_eager_registry.register("GetPrintoptions")
def _np_get_printoptions_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement GetPrintoptions via get_printoptions.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("get_printoptions")(*args, **kwargs)


@global_eager_registry.register("AffineConfig")
def _np_affineconfig(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement AffineConfig.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return kwargs


@global_eager_registry.register("BlurConfig")
def _np_blurconfig(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement BlurConfig.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.configs import BlurConfig

    return BlurConfig(*args, **kwargs)


@global_eager_registry.register("CustomRoot")
def _np_customroot(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement CustomRoot.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    f = args[0]
    initial_guess = args[1]
    solve = kwargs.get("solve", args[2] if len(args) > 2 else None)
    if solve:
        return require_callable(solve)(f, initial_guess)
    return initial_guess


@global_eager_registry.register("ElasticConfig")
def _np_elasticconfig(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement ElasticConfig.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.configs import ElasticConfig

    return ElasticConfig(*args, **kwargs)


@global_eager_registry.register("LinearOperator")
def _np_linearoperator(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperator.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperator

    return LinearOperator(*args, **kwargs)


@global_eager_registry.register("LinearOperatorAdjoint")
def _np_linearoperatoradjoint(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorAdjoint.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorAdjoint

    return LinearOperatorAdjoint(*args, **kwargs)


@global_eager_registry.register("LinearOperatorCirculant")
def _np_linearoperatorcirculant(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorCirculant.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant

    return LinearOperatorCirculant(*args, **kwargs)


@global_eager_registry.register("LinearOperatorCirculant2D")
def _np_linearoperatorcirculant2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorCirculant2D.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant2D

    return LinearOperatorCirculant2D(*args, **kwargs)


@global_eager_registry.register("LinearOperatorCirculant3D")
def _np_linearoperatorcirculant3d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorCirculant3D.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant3D

    return LinearOperatorCirculant3D(*args, **kwargs)


@global_eager_registry.register("LinearOperatorComposition")
def _np_linearoperatorcomposition(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorComposition.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorComposition

    return LinearOperatorComposition(*args, **kwargs)


@global_eager_registry.register("LinearOperatorHouseholder")
def _np_linearoperatorhouseholder(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorHouseholder.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorHouseholder

    return LinearOperatorHouseholder(*args, **kwargs)


@global_eager_registry.register("LinearOperatorKronecker")
def _np_linearoperatorkronecker(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorKronecker.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorKronecker

    return LinearOperatorKronecker(*args, **kwargs)


@global_eager_registry.register("LinearOperatorLowRankUpdate")
def _np_linearoperatorlowrankupdate(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorLowRankUpdate.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorLowRankUpdate

    return LinearOperatorLowRankUpdate(*args, **kwargs)


@global_eager_registry.register("LinearOperatorToeplitz")
def _np_linearoperatortoeplitz(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement LinearOperatorToeplitz.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorToeplitz

    return LinearOperatorToeplitz(*args, **kwargs)


@global_eager_registry.register("PerspectiveConfig")
def _np_perspectiveconfig(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement PerspectiveConfig.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.configs import PerspectiveConfig

    return PerspectiveConfig(*args, **kwargs)


@global_eager_registry.register("RawMatMul")
def _np_rawmatmul(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement RawMatMul.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
        tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if adapt_backend_module(backend_module).has_symbol("rawmatmul"):
        return adapt_backend_module(backend_module).require_symbol("rawmatmul")(*args, **kwargs)
    return adapt_backend_module(arr_mod).require_symbol("matmul")(args[0], args[1])


@global_eager_registry.register("RawMerge")
def _np_rawmerge(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement RawMerge.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    inputs = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    return (
        (inputs[0], arr_mod.array(0, dtype=arr_mod.int32)) if inputs else (None, arr_mod.array(-1, dtype=arr_mod.int32))
    )


@global_eager_registry.register("RawOp")
def _np_rawop(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement RawOp.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return args[0] if args else None


@global_eager_registry.register("RawSwitch")
def _np_rawswitch(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement RawSwitch.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    data = args[0]
    pred = args[1] if len(args) > 1 else kwargs.get("pred", False)
    if bool(arr_mod.asarray(pred).item()):
        return (None, data)
    return (data, None)


@global_eager_registry.register("ScanOp")
def _np_scanop(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement ScanOp.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    (fn, elems, acc, has_acc) = _parse_scanop_args(args, kwargs)
    if not callable(fn) or elems is None:
        return args[0] if args else None
    elems_arr = arr_mod.asarray(elems)
    if elems_arr.size == 0:
        return elems_arr
    out = arr_mod.empty_like(elems_arr)
    if not has_acc:
        acc = elems_arr[0]
        out[0] = acc
    start_idx = 0 if has_acc else 1
    for i in range(start_idx, elems_arr.shape[0]):
        acc = fn(acc, elems_arr[i])
        out[i] = acc
    return out


@global_eager_registry.register("SwitchOp")
def _np_switchop(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement SwitchOp.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    data = args[0]
    pred = args[1] if len(args) > 1 else kwargs.get("pred", False)
    if bool(arr_mod.asarray(pred).item()):
        return (None, data)
    return (data, None)


@global_eager_registry.register("TensorArrayRead")
def _np_tensorarrayread(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement TensorArrayRead.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    handle = args[0]
    raw_idx = args[1]
    if isinstance(handle, (list, tuple)):
        item_val: BackendArg = handle[int(coerce_int(raw_idx))]
        return item_val
    from ml_switcheroo_compiler.backends.eager_registry import BackendArrayProtocol, coerce_array

    arr = coerce_array(backend_module, handle)
    idx_key: BackendArrayProtocol | int = (
        raw_idx if isinstance(raw_idx, BackendArrayProtocol) else int(coerce_int(raw_idx))
    )
    return arr[idx_key]


@global_eager_registry.register("TensorArrayWrite")
def _np_tensorarraywrite(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement TensorArrayWrite.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    handle = args[0]
    value = args[2]
    idx_int = int(coerce_int(args[1]))
    new_handle = coerce_operand_list(handle)
    if idx_int >= len(new_handle):
        new_handle.extend([None] * (idx_int - len(new_handle) + 1))
    new_handle[idx_int] = value
    return new_handle


@global_eager_registry.register("TensorConfig")
def _np_tensorconfig(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement TensorConfig.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import ShapeProtocol, coerce_shape
    from ml_switcheroo_compiler.core.shape import extract_shape
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    del backend_module
    if len(args) > 0:
        first = args[0]
        shape_arg = (
            extract_shape(first)
            if isinstance(first, ShapeProtocol) and not isinstance(first, (tuple, list, int))
            else coerce_shape(first)
        )
    else:
        shape_arg = coerce_shape(kwargs.get("shape", ()))
    return TensorConfig(shape=shape_arg)
