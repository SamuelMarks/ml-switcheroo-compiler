"""Implementation of math signal operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_signal`)."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: F401

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_eager_tensor, ensure_int, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member

from .math_general import _get_np_arg


@numpy_eager_registry.register("Correlate")
def _np_correlate(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.correlate(*args, **kwargs)


@numpy_eager_registry.register("Blackman")
def _np_blackman_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Blackman via blackman.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.blackman(*args, **kwargs)


@numpy_eager_registry.register("Hamming")
def _np_hamming_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Hamming via hamming.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.hamming(*args, **kwargs)


@numpy_eager_registry.register("Hanning")
def _np_hanning_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Hanning via hanning.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.hanning(*args, **kwargs)


@numpy_eager_registry.register("Kaiser")
def _np_kaiser_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Kaiser via kaiser.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.kaiser(*args, **kwargs)


@numpy_eager_registry.register("Rfft")
def _np_rfft(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rfft operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "rfft")(a, **kwargs))


@numpy_eager_registry.register("Ifft")
def _np_ifft(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np ifft operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "ifft")(a, **kwargs))


@numpy_eager_registry.register("Fftn")
def _np_fftn(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fftn operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "fftn")(a, **kwargs))


@numpy_eager_registry.register("Ifftn")
def _np_ifftn(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np ifftn operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "ifftn")(a, **kwargs))


@numpy_eager_registry.register("Rfftn")
def _np_rfftn(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rfftn operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "rfftn")(a, **kwargs))


@numpy_eager_registry.register("Irfftn")
def _np_irfftn(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np irfftn operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "irfftn")(a, **kwargs))


@numpy_eager_registry.register("Ifft2")
def _np_ifft2(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np ifft2 operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "ifft2")(a, **kwargs))


@numpy_eager_registry.register("Rfft2")
def _np_rfft2(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rfft2 operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "rfft2")(a, **kwargs))


@numpy_eager_registry.register("Irfft2")
def _np_irfft2(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np irfft2 operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "irfft2")(a, **kwargs))


@numpy_eager_registry.register("Fftnd")
def _np_fftnd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fftnd operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "fftn")(a, **kwargs))


@numpy_eager_registry.register("Ifftnd")
def _np_ifftnd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np ifftnd operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "ifftn")(a, **kwargs))


@numpy_eager_registry.register("Rfftnd")
def _np_rfftnd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rfftnd operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "rfftn")(a, **kwargs))


@numpy_eager_registry.register("Irfftnd")
def _np_irfftnd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np irfftnd operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "irfftn")(a, **kwargs))


@numpy_eager_registry.register("Fftshift")
def _np_fftshift(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fftshift operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "fftshift")(a, **kwargs))


@numpy_eager_registry.register("Ifftshift")
def _np_ifftshift(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np ifftshift operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "ifftshift")(a, **kwargs))


@numpy_eager_registry.register("Hfft")
def _np_hfft(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np hfft operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return ensure_eager_tensor(get_backend_member(fft, "hfft")(a, **kwargs))


@numpy_eager_registry.register("Rfftfreq")
def _np_rfftfreq(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rfftfreq operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy.fft as fft

    if not args:
        return None
    try:
        n = int(ensure_int(args[0]))
    except (TypeError, ValueError):
        return args[0]
    return ensure_eager_tensor(get_backend_member(fft, "rfftfreq")(n, **kwargs))


@numpy_eager_registry.register("Rrelu")
def _np_rrelu(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rrelu operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    lower = kwargs.get("lower", 0.125)
    upper = kwargs.get("upper", 0.333)
    alpha = get_backend_member(np.random, "uniform")(lower, upper, size=a.shape)
    return np.where(np.greater_equal(a, 0), a, np.multiply(a, alpha))


@numpy_eager_registry.register("Softmax")
def _np_softmax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np softmax operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = ensure_int(kwargs.get("axis", -1))
    e_x = get_backend_member(np, "exp")(a - get_backend_member(np, "max")(a, axis=axis, keepdims=True))
    return ensure_eager_tensor(e_x / e_x.sum(axis=axis, keepdims=True))


@numpy_eager_registry.register("Sigmoid")
def _np_sigmoid(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np sigmoid operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return 1.0 / (1.0 + np.exp(-a))
