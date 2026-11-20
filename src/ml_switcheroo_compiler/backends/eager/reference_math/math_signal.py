"""Generic backend-parametric eager reference operations for `reference_math.math_signal`."""

from __future__ import annotations

import types
from collections.abc import Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _get_np_arg
from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Correlate")
def _np_correlate(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("correlate")(*args, **kwargs)


@global_eager_registry.register("Blackman")
def _np_blackman_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Blackman via blackman.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("blackman")(*args, **kwargs)


@global_eager_registry.register("Hamming")
def _np_hamming_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Hamming via hamming.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("hamming")(*args, **kwargs)


@global_eager_registry.register("Hanning")
def _np_hanning_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Hanning via hanning.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("hanning")(*args, **kwargs)


@global_eager_registry.register("Kaiser")
def _np_kaiser_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Kaiser via kaiser.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("kaiser")(*args, **kwargs)


@global_eager_registry.register("Rfft")
def _np_rfft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("rfft")(a, **kwargs)


@global_eager_registry.register("Ifft")
def _np_ifft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("ifft")(a, **kwargs)


@global_eager_registry.register("Fftn")
def _np_fftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fftn operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("fftn")(a, **kwargs)


@global_eager_registry.register("Ifftn")
def _np_ifftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifftn operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("ifftn")(a, **kwargs)


@global_eager_registry.register("Rfftn")
def _np_rfftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfftn operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("rfftn")(a, **kwargs)


@global_eager_registry.register("Irfftn")
def _np_irfftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np irfftn operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("irfftn")(a, **kwargs)


@global_eager_registry.register("Ifft2")
def _np_ifft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifft2 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("ifft2")(a, **kwargs)


@global_eager_registry.register("Rfft2")
def _np_rfft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfft2 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("rfft2")(a, **kwargs)


@global_eager_registry.register("Irfft2")
def _np_irfft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np irfft2 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("irfft2")(a, **kwargs)


@global_eager_registry.register("Fftnd")
def _np_fftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fftnd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("fftn")(a, **kwargs)


@global_eager_registry.register("Ifftnd")
def _np_ifftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifftnd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("ifftn")(a, **kwargs)


@global_eager_registry.register("Rfftnd")
def _np_rfftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfftnd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("rfftn")(a, **kwargs)


@global_eager_registry.register("Irfftnd")
def _np_irfftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np irfftnd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("irfftn")(a, **kwargs)


@global_eager_registry.register("Fftshift")
def _np_fftshift(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fftshift operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("fftshift")(a, **kwargs)


@global_eager_registry.register("Ifftshift")
def _np_ifftshift(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifftshift operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("ifftshift")(a, **kwargs)


@global_eager_registry.register("Hfft")
def _np_hfft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np hfft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return adapt_backend_module(fft).require_symbol("hfft")(a, **kwargs)


@global_eager_registry.register("Rfftfreq")
def _np_rfftfreq(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfftfreq operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fft = adapt_backend_module(_resolve_eager_mod(backend_module)).require_symbol("fft")

    if not args:
        return None
    try:
        n = int(coerce_int(args[0]))
    except (TypeError, ValueError):
        return args[0]
    return adapt_backend_module(fft).require_symbol("rfftfreq")(n, **kwargs)


@global_eager_registry.register("Rrelu")
def _np_rrelu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rrelu operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    lower = kwargs.get("lower", 0.125)
    upper = kwargs.get("upper", 0.333)
    alpha = adapt_backend_module(arr_mod.random).require_symbol("uniform")(lower, upper, size=a.shape)
    return arr_mod.where(arr_mod.greater_equal(a, 0), a, arr_mod.multiply(a, alpha))


@global_eager_registry.register("Softmax")
def _np_softmax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np softmax operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = coerce_int(kwargs.get("axis", -1))
    e_x = adapt_backend_module(arr_mod).require_symbol("exp")(
        a - adapt_backend_module(arr_mod).require_symbol("max")(a, axis=axis, keepdims=True)
    )
    return e_x / e_x.sum(axis=axis, keepdims=True)


@global_eager_registry.register("Sigmoid")
def _np_sigmoid(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np sigmoid operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return 1.0 / (1.0 + arr_mod.exp(-a))
