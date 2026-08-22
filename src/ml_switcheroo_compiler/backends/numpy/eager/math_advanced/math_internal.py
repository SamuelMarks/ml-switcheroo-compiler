# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_internal module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_string_io import _parse_scanop_args


@numpy_eager_registry.register("PopulationCount")
def _np_population_count(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Count the number of set bits in the binary representation of each element.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    x_arr = np.asarray(x)
    return np.array([bin(n).count("1") for n in x_arr.flat]).reshape(x_arr.shape)


@numpy_eager_registry.register("GetPrintoptions")
def _np_get_printoptions_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement GetPrintoptions via get_printoptions.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.get_printoptions(*args, **kwargs)


@numpy_eager_registry.register("AffineConfig")
def _np_affineconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement AffineConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return kwargs


@numpy_eager_registry.register("BlurConfig")
def _np_blurconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement BlurConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.configs import BlurConfig

    return BlurConfig(*args, **kwargs)


@numpy_eager_registry.register("CustomRoot")
def _np_customroot(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement CustomRoot.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    f = args[0]
    initial_guess = args[1]
    solve = kwargs.get("solve", args[2] if len(args) > 2 else None)
    if solve:
        return solve(f, initial_guess)
    return initial_guess


@numpy_eager_registry.register("ElasticConfig")
def _np_elasticconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ElasticConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.configs import ElasticConfig

    return ElasticConfig(*args, **kwargs)


@numpy_eager_registry.register("LinearOperator")
def _np_linearoperator(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperator.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperator

    return LinearOperator(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorAdjoint")
def _np_linearoperatoradjoint(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorAdjoint.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorAdjoint

    return LinearOperatorAdjoint(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant")
def _np_linearoperatorcirculant(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorCirculant.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant

    return LinearOperatorCirculant(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant2D")
def _np_linearoperatorcirculant2d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorCirculant2D.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant2D

    return LinearOperatorCirculant2D(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant3D")
def _np_linearoperatorcirculant3d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorCirculant3D.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant3D

    return LinearOperatorCirculant3D(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorComposition")
def _np_linearoperatorcomposition(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorComposition.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorComposition

    return LinearOperatorComposition(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorHouseholder")
def _np_linearoperatorhouseholder(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorHouseholder.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorHouseholder

    return LinearOperatorHouseholder(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorKronecker")
def _np_linearoperatorkronecker(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorKronecker.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorKronecker

    return LinearOperatorKronecker(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorLowRankUpdate")
def _np_linearoperatorlowrankupdate(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorLowRankUpdate.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorLowRankUpdate

    return LinearOperatorLowRankUpdate(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorToeplitz")
def _np_linearoperatortoeplitz(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorToeplitz.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorToeplitz

    return LinearOperatorToeplitz(*args, **kwargs)


@numpy_eager_registry.register("PerspectiveConfig")
def _np_perspectiveconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement PerspectiveConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.configs import PerspectiveConfig

    return PerspectiveConfig(*args, **kwargs)


@numpy_eager_registry.register("RawMatMul")
def _np_rawmatmul(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawMatMul.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "RawMatMul"):
            cls_or_func = _ops.RawMatMul
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, _ops.OpDef)):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e
    if hasattr(backend_module, "rawmatmul"):
        return backend_module.rawmatmul(*args, **kwargs)
    return np.matmul(args[0], args[1])


@numpy_eager_registry.register("RawMerge")
def _np_rawmerge(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawMerge.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    inputs = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    return (inputs[0], np.array(0, dtype=np.int32)) if inputs else (None, np.array(-1, dtype=np.int32))


@numpy_eager_registry.register("RawOp")
def _np_rawop(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawOp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return args[0] if args else None


@numpy_eager_registry.register("RawSwitch")
def _np_rawswitch(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawSwitch.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    data = args[0]
    pred = args[1] if len(args) > 1 else kwargs.get("pred", False)
    if bool(np.asarray(pred).item()):
        return (None, data)
    return (data, None)


@numpy_eager_registry.register("ScanOp")
def _np_scanop(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ScanOp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

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
        acc = fn(acc, elems_arr[i])
        out[i] = acc
    return out


@numpy_eager_registry.register("SwitchOp")
def _np_switchop(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SwitchOp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    data = args[0]
    pred = args[1] if len(args) > 1 else kwargs.get("pred", False)
    if bool(np.asarray(pred).item()):
        return (None, data)
    return (data, None)


@numpy_eager_registry.register("TensorArrayRead")
def _np_tensorarrayread(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorArrayRead.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    handle = args[0]
    index = args[1]
    return handle[index]


@numpy_eager_registry.register("TensorArrayWrite")
def _np_tensorarraywrite(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorArrayWrite.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    handle = args[0]
    index = args[1]
    value = args[2]
    new_handle = list(handle)
    if index >= len(new_handle):
        new_handle.extend([None] * (index - len(new_handle) + 1))
    new_handle[index] = value
    return new_handle


@numpy_eager_registry.register("TensorConfig")
def _np_tensorconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    return TensorConfig(*args, **kwargs)
