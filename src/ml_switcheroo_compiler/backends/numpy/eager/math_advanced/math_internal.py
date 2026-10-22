"""Implementation of math internal operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_internal`)."""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: F401

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_int, as_tensor, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

from .math_string_io import _parse_scanop_args


@numpy_eager_registry.register("PopulationCount")
def _np_population_count(backend_module: BackendModuleProtocol, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Count the number of set bits in the binary representation of each element.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    x_arr = np.asarray(x)
    return np.array([bin(bool(n) if isinstance(n, np.bool_) else n).count("1") for n in x_arr.flat]).reshape(x_arr.shape)


@numpy_eager_registry.register("GetPrintoptions")
def _np_get_printoptions_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement GetPrintoptions via get_printoptions.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.get_printoptions(*args, **kwargs)


@numpy_eager_registry.register("AffineConfig")
def _np_affineconfig(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement AffineConfig.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return kwargs


@numpy_eager_registry.register("BlurConfig")
def _np_blurconfig(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement BlurConfig.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.configs import BlurConfig

    return BlurConfig(*args, **kwargs)


@numpy_eager_registry.register("CustomRoot")
def _np_customroot(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement CustomRoot.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    f = args[0]
    initial_guess = args[1]
    solve = kwargs.get("solve", args[2] if len(args) > 2 else None)
    if solve:
        return as_tensor(as_callable(solve)(f, initial_guess))
    return initial_guess


@numpy_eager_registry.register("ElasticConfig")
def _np_elasticconfig(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement ElasticConfig.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.configs import ElasticConfig

    return ElasticConfig(*args, **kwargs)


@numpy_eager_registry.register("LinearOperator")
def _np_linearoperator(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperator.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperator

    return LinearOperator(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorAdjoint")
def _np_linearoperatoradjoint(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorAdjoint.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorAdjoint

    return LinearOperatorAdjoint(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant")
def _np_linearoperatorcirculant(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorCirculant.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant

    return LinearOperatorCirculant(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant2D")
def _np_linearoperatorcirculant2d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorCirculant2D.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant2D

    return LinearOperatorCirculant2D(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant3D")
def _np_linearoperatorcirculant3d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorCirculant3D.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant3D

    return LinearOperatorCirculant3D(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorComposition")
def _np_linearoperatorcomposition(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorComposition.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorComposition

    return LinearOperatorComposition(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorHouseholder")
def _np_linearoperatorhouseholder(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorHouseholder.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorHouseholder

    return LinearOperatorHouseholder(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorKronecker")
def _np_linearoperatorkronecker(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorKronecker.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorKronecker

    return LinearOperatorKronecker(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorLowRankUpdate")
def _np_linearoperatorlowrankupdate(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorLowRankUpdate.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorLowRankUpdate

    return LinearOperatorLowRankUpdate(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorToeplitz")
def _np_linearoperatortoeplitz(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement LinearOperatorToeplitz.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorToeplitz

    return LinearOperatorToeplitz(*args, **kwargs)


@numpy_eager_registry.register("PerspectiveConfig")
def _np_perspectiveconfig(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement PerspectiveConfig.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.configs import PerspectiveConfig

    return PerspectiveConfig(*args, **kwargs)


@numpy_eager_registry.register("RawMatMul")
def _np_rawmatmul(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement RawMatMul.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
        tuple[int, ...]: Result.
    """
    if hasattr(backend_module, "rawmatmul"):
        return as_tensor(backend_module.rawmatmul(*args, **kwargs))
    return as_tensor(getattr(np, "matmul")(args[0], args[1]))


@numpy_eager_registry.register("RawMerge")
def _np_rawmerge(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement RawMerge.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    inputs = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    return (inputs[0], np.array(0, dtype=np.int32)) if inputs else (None, np.array(-1, dtype=np.int32))


@numpy_eager_registry.register("RawOp")
def _np_rawop(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement RawOp.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return args[0] if args else None


@numpy_eager_registry.register("RawSwitch")
def _np_rawswitch(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement RawSwitch.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    data = args[0]
    pred = args[1] if len(args) > 1 else kwargs.get("pred", False)
    if bool(np.asarray(pred).item()):
        return (None, data)
    return (data, None)


@numpy_eager_registry.register("ScanOp")
def _np_scanop(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement ScanOp.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    (fn, elems, acc, has_acc) = _parse_scanop_args(args, kwargs)
    if not callable(fn) or elems is None:
        return args[0] if args else None
    elems_arr = np.asarray(elems)
    if elems_arr.size == 0:
        return elems_arr
    out = np.empty_like(elems_arr)
    if not has_acc:
        acc = elems_arr[0]
        out[0] = acc
    start_idx = 0 if has_acc else 1
    for i in range(start_idx, elems_arr.shape[0]):
        acc = as_callable(fn)(acc, elems_arr[i])
        out[i] = acc
    return out


@numpy_eager_registry.register("SwitchOp")
def _np_switchop(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement SwitchOp.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    data = args[0]
    pred = args[1] if len(args) > 1 else kwargs.get("pred", False)
    if bool(np.asarray(pred).item()):
        return (None, data)
    return (data, None)


@numpy_eager_registry.register("TensorArrayRead")
def _np_tensorarrayread(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement TensorArrayRead.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    handle = args[0]
    index = args[1]
    return as_tensor(getattr(handle, "__getitem__")(index))


@numpy_eager_registry.register("TensorArrayWrite")
def _np_tensorarraywrite(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement TensorArrayWrite.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    handle = args[0]
    index = args[1]
    value = args[2]
    new_handle = list(getattr(handle, "__iter__")())
    if int(as_int(index)) >= len(new_handle):
        new_handle.extend([None] * int(as_tensor(index) - as_tensor(len(new_handle)) + 1))
    as_tensor(new_handle)[index] = value
    return new_handle


@numpy_eager_registry.register("TensorConfig")
def _np_tensorconfig(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement TensorConfig.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    return as_tensor(as_callable(TensorConfig)(*args, **kwargs))
