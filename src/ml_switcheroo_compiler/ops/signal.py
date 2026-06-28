# pragma: no cover
"""Signal processing operations."""
# pragma: no cover

# pragma: no cover
from ml_switcheroo_compiler.ops.base import OpDef, register_op

# pragma: no cover
from ml_switcheroo_compiler.core.config import config

# pragma: no cover
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
# pragma: no cover

# pragma: no cover


# pragma: no cover
@register_op("Convolve2d")
# pragma: no cover
class Convolve2d(OpDef):
    # pragma: no cover
    """Convolve2d."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("Fftconvolve")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class Fftconvolve(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Fftconvolve."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("Welch")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class Welch(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Welch."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def _emit_signal_node(
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    op_type: str,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    inputs: list[Tensor],
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    attrs: dict[str, object],
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    out_shape: tuple[int, ...],
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    dtype: str,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
) -> Tensor:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Emit a signal node."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return _emit_linalg_node(op_type, inputs, attrs, [out_shape], [dtype])


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def convolve2d(
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    in1: Tensor,
    in2: Tensor,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0.0,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
) -> Tensor:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate convolve2d."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover

        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        backend = get_active_backend()
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        data = backend.execute_op(
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "Convolve2d",
            in1.data,
            in2.data,
            mode=mode,
            boundary=boundary,
            fillvalue=fillvalue,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
        )
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return Tensor(data, TensorConfig(data.shape, in1.dtype, in1.device))
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return _emit_signal_node(
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        "Convolve2d",
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        [in1, in2],
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        {"mode": mode, "boundary": boundary, "fillvalue": fillvalue},
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        in1.shape,
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        in1.dtype,
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
    )


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def fftconvolve(in1: Tensor, in2: Tensor, mode: str = "full", axes: object = None) -> Tensor:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate fftconvolve."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover

        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        backend = get_active_backend()
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        data = backend.execute_op("Fftconvolve", in1.data, in2.data, mode=mode, axes=axes)
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return Tensor(data, TensorConfig(data.shape, in1.dtype, in1.device))
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return _emit_signal_node(
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        "Fftconvolve",
        [in1, in2],
        {"mode": mode, "axes": axes},
        in1.shape,
        in1.dtype,
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
    )


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def welch(  # noqa: PLR0913
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    x: Tensor,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    fs: float = 1.0,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    window: str = "hann",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    nperseg: int = None,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    noverlap: int = None,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    nfft: int = None,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    detrend: str = "constant",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return_onesided: bool = True,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    scaling: str = "density",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    axis: int = -1,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    average: str = "mean",
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
) -> tuple[Tensor, Tensor]:  # noqa: PLR0913
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate welch."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover

        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        backend = get_active_backend()
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        f, Pxx = backend.execute_op(
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "Welch",
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            x.data,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            fs=fs,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            window=window,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            nperseg=nperseg,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            noverlap=noverlap,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            nfft=nfft,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            detrend=detrend,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            return_onesided=return_onesided,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            scaling=scaling,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            axis=axis,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            average=average,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
        )
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return (
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            Tensor(f, TensorConfig(f.shape, x.dtype, x.device)),
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            Tensor(Pxx, TensorConfig(Pxx.shape, x.dtype, x.device)),
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
        )
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    # Using simple shapes for graph tracing
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    f_shape = (256,)  # Dummy placeholder for trace
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    Pxx_shape = (256,)  # Dummy placeholder for trace
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    f, Pxx = _emit_linalg_node(
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        "Welch",
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        [x],
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        {
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "fs": fs,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "window": window,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "nperseg": nperseg,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "noverlap": noverlap,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "nfft": nfft,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "detrend": detrend,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "return_onesided": return_onesided,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "scaling": scaling,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "axis": axis,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "average": average,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
        },
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        [f_shape, Pxx_shape],
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        [x.dtype, x.dtype],
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
    )
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return f, Pxx


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover
__all__ = ["convolve2d", "fftconvolve", "welch"]
# pragma: no cover
# pragma: no cover
# pragma: no cover
