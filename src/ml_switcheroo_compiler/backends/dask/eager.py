# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import Any

try:
    import dask.array as da
except ImportError:
    da = None  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


_OP_MAPPING = None


def _get_op_mapping() -> dict[str, Any]:
    """Retrieve the operation mapping for the Dask backend.

    Returns:
        dict: A dictionary mapping operation types to their implementations.
    """
    global _OP_MAPPING
    if _OP_MAPPING is not None:
        return _OP_MAPPING
    import dask.array as da

    _OP_MAPPING = {
        "Abs": da.abs,  # type: ignore
        "Add": da.add,  # type: ignore
        "All": da.all,  # type: ignore
        "Allclose": da.allclose,  # type: ignore
        "Angle": da.angle,  # type: ignore
        "Any": da.any,  # type: ignore
        "Append": da.append,  # type: ignore
        "ApplyOverAxes": da.apply_over_axes,  # type: ignore
        "Argmax": da.argmax,  # type: ignore
        "Argmin": da.argmin,  # type: ignore
        "Argwhere": da.argwhere,  # type: ignore
        "Average": da.average,  # type: ignore
        "Bincount": da.bincount,  # type: ignore
        "BitwiseAnd": da.bitwise_and,  # type: ignore
        "BitwiseNot": da.bitwise_not,  # type: ignore
        "BitwiseOr": da.bitwise_or,  # type: ignore
        "BitwiseXor": da.bitwise_xor,  # type: ignore
        "Block": da.block,  # type: ignore
        "BroadcastTo": da.broadcast_to,  # type: ignore
        "Cbrt": da.cbrt,  # type: ignore
        "Ceil": da.ceil,  # type: ignore
        "Cholesky": da.linalg.cholesky,
        "Choose": da.choose,  # type: ignore
        "Clip": da.clip,  # type: ignore
        "Compress": da.compress,  # type: ignore
        "Concatenate": da.concatenate,  # type: ignore
        "Conj": da.conj,  # type: ignore
        "Copysign": da.copysign,  # type: ignore
        "Corrcoef": da.corrcoef,  # type: ignore
        "Cos": da.cos,  # type: ignore
        "Cosh": da.cosh,  # type: ignore
        "CountNonzero": da.count_nonzero,  # type: ignore
        "Cov": da.cov,  # type: ignore
        "Cumprod": da.cumprod,  # type: ignore
        "Cumsum": da.cumsum,  # type: ignore
        "Degrees": da.degrees,  # type: ignore
        "Delete": da.delete,  # type: ignore
        "Diag": da.diag,  # type: ignore
        "Diagonal": da.diagonal,  # type: ignore
        "Diff": da.diff,  # type: ignore
        "Digitize": da.digitize,  # type: ignore
        "Divide": da.divide,  # type: ignore
        "Divmod": da.divmod,  # type: ignore
        "Dot": da.dot,  # type: ignore
        "Dstack": da.dstack,  # type: ignore
        "Ediff1d": da.ediff1d,  # type: ignore
        "Einsum": da.einsum,  # type: ignore
        "Equal": da.equal,  # type: ignore
        "Exp": da.exp,  # type: ignore
        "Exp2": da.exp2,  # type: ignore
        "ExpandDims": da.expand_dims,  # type: ignore
        "Expm1": da.expm1,  # type: ignore
        "Extract": da.extract,  # type: ignore
        "Fabs": da.fabs,  # type: ignore
        "Fft": da.fft,
        "Fft2": da.fft.fft2,
        "Fftfreq": da.fft.fftfreq,
        "Fftn": da.fft.fftn,
        "Fftshift": da.fft.fftshift,
        "Fix": da.fix,  # type: ignore
        "Flatnonzero": da.flatnonzero,  # type: ignore
        "Flip": da.flip,  # type: ignore
        "Fliplr": da.fliplr,  # type: ignore
        "Flipud": da.flipud,  # type: ignore
        "FloatPower": da.float_power,  # type: ignore
        "Floor": da.floor,  # type: ignore
        "FloorDivide": da.floor_divide,  # type: ignore
        "Fmax": da.fmax,  # type: ignore
        "Fmin": da.fmin,  # type: ignore
        "Fmod": da.fmod,  # type: ignore
        "Frexp": da.frexp,  # type: ignore
        "Fromfunction": da.fromfunction,  # type: ignore
        "Frompyfunc": da.frompyfunc,  # type: ignore
        "Greater": da.greater,  # type: ignore
        "GreaterEqual": da.greater_equal,  # type: ignore
        "Hfft": da.fft.hfft,
        "Hstack": da.hstack,  # type: ignore
        "Hypot": da.hypot,  # type: ignore
        "Ifft": da.fft.ifft,
        "Ifft2": da.fft.ifft2,
        "Ifftn": da.fft.ifftn,
        "Ifftshift": da.fft.ifftshift,
        "Ihfft": da.fft.ihfft,
        "Imag": da.imag,  # type: ignore
        "Insert": da.insert,  # type: ignore
        "Inv": da.linalg.inv,
        "Invert": da.invert,  # type: ignore
        "Irfft": da.fft.irfft,
        "Irfft2": da.fft.irfft2,
        "Irfftn": da.fft.irfftn,
        "Isclose": da.isclose,  # type: ignore
        "Iscomplex": da.iscomplex,  # type: ignore
        "Isfinite": da.isfinite,  # type: ignore
        "Isin": da.isin,  # type: ignore
        "Isinf": da.isinf,  # type: ignore
        "Isnan": da.isnan,  # type: ignore
        "Isneginf": da.isneginf,  # type: ignore
        "Isposinf": da.isposinf,  # type: ignore
        "Isreal": da.isreal,  # type: ignore
        "Ldexp": da.ldexp,  # type: ignore
        "LeftShift": da.left_shift,  # type: ignore
        "Less": da.less,  # type: ignore
        "LessEqual": da.less_equal,  # type: ignore
        "Log": da.log,  # type: ignore
        "Log10": da.log10,  # type: ignore
        "Log2": da.log2,  # type: ignore
        "Logaddexp": da.logaddexp,  # type: ignore
        "Logaddexp2": da.logaddexp2,  # type: ignore
        "LogicalAnd": da.logical_and,  # type: ignore
        "LogicalNot": da.logical_not,  # type: ignore
        "LogicalOr": da.logical_or,  # type: ignore
        "LogicalXor": da.logical_xor,  # type: ignore
        "Lstsq": da.linalg.lstsq,
        "Lu": da.linalg.lu,
        "Matmul": da.matmul,  # type: ignore
        "Max": da.max,  # type: ignore
        "Maximum": da.maximum,  # type: ignore
        "Mean": da.mean,  # type: ignore
        "Min": da.min,  # type: ignore
        "Minimum": da.minimum,  # type: ignore
        "Mod": da.mod,  # type: ignore
        "Moveaxis": da.moveaxis,  # type: ignore
        "Multiply": da.multiply,  # type: ignore
        "NanToNum": da.nan_to_num,  # type: ignore
        "Nanargmax": da.nanargmax,  # type: ignore
        "Nanargmin": da.nanargmin,  # type: ignore
        "Nancumprod": da.nancumprod,  # type: ignore
        "Nancumsum": da.nancumsum,  # type: ignore
        "Nanmax": da.nanmax,  # type: ignore
        "Nanmean": da.nanmean,  # type: ignore
        "Nanmedian": da.nanmedian,  # type: ignore
        "Nanmin": da.nanmin,  # type: ignore
        "Nanprod": da.nanprod,  # type: ignore
        "Nanstd": da.nanstd,  # type: ignore
        "Nansum": da.nansum,  # type: ignore
        "Nanvar": da.nanvar,  # type: ignore
        "Negative": da.negative,  # type: ignore
        "Nextafter": da.nextafter,  # type: ignore
        "Nonzero": da.nonzero,  # type: ignore
        "Norm": da.linalg.norm,
        "NotEqual": da.not_equal,  # type: ignore
        "Outer": da.outer,  # type: ignore
        "Pad": da.pad,  # type: ignore
        "Percentile": da.percentile,  # type: ignore
        "Positive": da.positive,  # type: ignore
        "Power": da.power,  # type: ignore
        "Prod": da.prod,  # type: ignore
        "Qr": da.linalg.qr,
        "Radians": da.radians,  # type: ignore
        "RavelMultiIndex": da.ravel_multi_index,  # type: ignore
        "Real": da.real,  # type: ignore
        "Reciprocal": da.reciprocal,  # type: ignore
        "Remainder": da.remainder,  # type: ignore
        "Repeat": da.repeat,  # type: ignore
        "Reshape": da.reshape,  # type: ignore
        "Rfft": da.fft.rfft,
        "Rfft2": da.fft.rfft2,
        "Rfftfreq": da.fft.rfftfreq,
        "Rfftn": da.fft.rfftn,
        "RightShift": da.right_shift,  # type: ignore
        "Rint": da.rint,  # type: ignore
        "Roll": da.roll,  # type: ignore
        "Round": da.round,  # type: ignore
        "Searchsorted": da.searchsorted,  # type: ignore
        "Select": da.select,  # type: ignore
        "Sign": da.sign,  # type: ignore
        "Signbit": da.signbit,  # type: ignore
        "Sin": da.sin,  # type: ignore
        "Sinc": da.sinc,  # type: ignore
        "Sinh": da.sinh,  # type: ignore
        "Solve": da.linalg.solve,
        "Sqrt": da.sqrt,  # type: ignore
        "Square": da.square,  # type: ignore
        "Squeeze": da.squeeze,  # type: ignore
        "Stack": da.stack,  # type: ignore
        "Std": da.std,  # type: ignore
        "Subtract": da.subtract,  # type: ignore
        "Sum": da.sum,  # type: ignore
        "Svd": da.linalg.svd,
        "Swapaxes": da.swapaxes,  # type: ignore
        "Take": da.take,  # type: ignore
        "Tan": da.tan,  # type: ignore
        "Tanh": da.tanh,  # type: ignore
        "Tensordot": da.tensordot,  # type: ignore
        "Tile": da.tile,  # type: ignore
        "Trace": da.trace,  # type: ignore
        "Transpose": da.transpose,  # type: ignore
        "TrueDivide": da.true_divide,  # type: ignore
        "Trunc": da.trunc,  # type: ignore
        "Union1d": da.union1d,  # type: ignore
        "Unique": da.unique,  # type: ignore
        "UnravelIndex": da.unravel_index,  # type: ignore
        "Vdot": da.vdot,  # type: ignore
        "Vstack": da.vstack,  # type: ignore
        "Where": da.where,  # type: ignore
    }
    return _OP_MAPPING


def execute_op(cls: type, op_type: str, *args: Any, **kwargs: Any) -> Any:
    """Execute an eager operation using the Dask backend.

    Args:
        cls (type): The tensor class.
        op_type (str): The name of the operation to execute.
        *args (object): Positional arguments for the operation.
        **kwargs (object): Keyword arguments for the operation.

    Returns: Any: The result of the operation execution.

    Raises:
        BackendNotSupportedError: If the operation is not supported by the Dask backend.
    """
    import ml_switcheroo_compiler.backends.eager  # noqa: F401
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(cls, *args, **kwargs)

    op_mapping = _get_op_mapping()
    func = op_mapping.get(op_type)
    if func is not None:
        return func(*args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
