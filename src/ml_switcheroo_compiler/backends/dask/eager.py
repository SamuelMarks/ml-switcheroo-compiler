# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import Any

try:
    import dask.array as da
except ImportError:
    da = None  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


_OP_MAPPING = None


def _get_op_mapping() -> dict:
    """Retrieve the operation mapping for the Dask backend.

    Returns:
        dict: A dictionary mapping operation types to their implementations.
    """
    global _OP_MAPPING
    if _OP_MAPPING is not None:
        return _OP_MAPPING
    import dask.array as da

    _OP_MAPPING = {
        "Abs": da.abs,
        "Add": da.add,
        "All": da.all,
        "Allclose": da.allclose,
        "Angle": da.angle,
        "Any": da.any,
        "Append": da.append,
        "ApplyOverAxes": da.apply_over_axes,
        "Argmax": da.argmax,
        "Argmin": da.argmin,
        "Argwhere": da.argwhere,
        "Average": da.average,
        "Bincount": da.bincount,
        "BitwiseAnd": da.bitwise_and,
        "BitwiseNot": da.bitwise_not,
        "BitwiseOr": da.bitwise_or,
        "BitwiseXor": da.bitwise_xor,
        "Block": da.block,
        "BroadcastTo": da.broadcast_to,
        "Cbrt": da.cbrt,
        "Ceil": da.ceil,
        "Cholesky": da.linalg.cholesky,
        "Choose": da.choose,
        "Clip": da.clip,
        "Compress": da.compress,
        "Concatenate": da.concatenate,
        "Conj": da.conj,
        "Copysign": da.copysign,
        "Corrcoef": da.corrcoef,
        "Cos": da.cos,
        "Cosh": da.cosh,
        "CountNonzero": da.count_nonzero,
        "Cov": da.cov,
        "Cumprod": da.cumprod,
        "Cumsum": da.cumsum,
        "Degrees": da.degrees,
        "Delete": da.delete,
        "Diag": da.diag,
        "Diagonal": da.diagonal,
        "Diff": da.diff,
        "Digitize": da.digitize,
        "Divide": da.divide,
        "Divmod": da.divmod,
        "Dot": da.dot,
        "Dstack": da.dstack,
        "Ediff1d": da.ediff1d,
        "Einsum": da.einsum,
        "Equal": da.equal,
        "Exp": da.exp,
        "Exp2": da.exp2,
        "ExpandDims": da.expand_dims,
        "Expm1": da.expm1,
        "Extract": da.extract,
        "Fabs": da.fabs,
        "Fft": da.fft,
        "Fft2": da.fft.fft2,
        "Fftfreq": da.fft.fftfreq,
        "Fftn": da.fft.fftn,
        "Fftshift": da.fft.fftshift,
        "Fix": da.fix,
        "Flatnonzero": da.flatnonzero,
        "Flip": da.flip,
        "Fliplr": da.fliplr,
        "Flipud": da.flipud,
        "FloatPower": da.float_power,
        "Floor": da.floor,
        "FloorDivide": da.floor_divide,
        "Fmax": da.fmax,
        "Fmin": da.fmin,
        "Fmod": da.fmod,
        "Frexp": da.frexp,
        "Fromfunction": da.fromfunction,
        "Frompyfunc": da.frompyfunc,
        "Greater": da.greater,
        "GreaterEqual": da.greater_equal,
        "Hfft": da.fft.hfft,
        "Hstack": da.hstack,
        "Hypot": da.hypot,
        "Ifft": da.fft.ifft,
        "Ifft2": da.fft.ifft2,
        "Ifftn": da.fft.ifftn,
        "Ifftshift": da.fft.ifftshift,
        "Ihfft": da.fft.ihfft,
        "Imag": da.imag,
        "Insert": da.insert,
        "Inv": da.linalg.inv,
        "Invert": da.invert,
        "Irfft": da.fft.irfft,
        "Irfft2": da.fft.irfft2,
        "Irfftn": da.fft.irfftn,
        "Isclose": da.isclose,
        "Iscomplex": da.iscomplex,
        "Isfinite": da.isfinite,
        "Isin": da.isin,
        "Isinf": da.isinf,
        "Isnan": da.isnan,
        "Isneginf": da.isneginf,
        "Isposinf": da.isposinf,
        "Isreal": da.isreal,
        "Ldexp": da.ldexp,
        "LeftShift": da.left_shift,
        "Less": da.less,
        "LessEqual": da.less_equal,
        "Log": da.log,
        "Log10": da.log10,
        "Log2": da.log2,
        "Logaddexp": da.logaddexp,
        "Logaddexp2": da.logaddexp2,
        "LogicalAnd": da.logical_and,
        "LogicalNot": da.logical_not,
        "LogicalOr": da.logical_or,
        "LogicalXor": da.logical_xor,
        "Lstsq": da.linalg.lstsq,
        "Lu": da.linalg.lu,
        "Matmul": da.matmul,
        "Max": da.max,
        "Maximum": da.maximum,
        "Mean": da.mean,
        "Min": da.min,
        "Minimum": da.minimum,
        "Mod": da.mod,
        "Moveaxis": da.moveaxis,
        "Multiply": da.multiply,
        "NanToNum": da.nan_to_num,
        "Nanargmax": da.nanargmax,
        "Nanargmin": da.nanargmin,
        "Nancumprod": da.nancumprod,
        "Nancumsum": da.nancumsum,
        "Nanmax": da.nanmax,
        "Nanmean": da.nanmean,
        "Nanmedian": da.nanmedian,
        "Nanmin": da.nanmin,
        "Nanprod": da.nanprod,
        "Nanstd": da.nanstd,
        "Nansum": da.nansum,
        "Nanvar": da.nanvar,
        "Negative": da.negative,
        "Nextafter": da.nextafter,
        "Nonzero": da.nonzero,
        "Norm": da.linalg.norm,
        "NotEqual": da.not_equal,
        "Outer": da.outer,
        "Pad": da.pad,
        "Percentile": da.percentile,
        "Positive": da.positive,
        "Power": da.power,
        "Prod": da.prod,
        "Qr": da.linalg.qr,
        "Radians": da.radians,
        "RavelMultiIndex": da.ravel_multi_index,
        "Real": da.real,
        "Reciprocal": da.reciprocal,
        "Remainder": da.remainder,
        "Repeat": da.repeat,
        "Reshape": da.reshape,
        "Rfft": da.fft.rfft,
        "Rfft2": da.fft.rfft2,
        "Rfftfreq": da.fft.rfftfreq,
        "Rfftn": da.fft.rfftn,
        "RightShift": da.right_shift,
        "Rint": da.rint,
        "Roll": da.roll,
        "Round": da.round,
        "Searchsorted": da.searchsorted,
        "Select": da.select,
        "Sign": da.sign,
        "Signbit": da.signbit,
        "Sin": da.sin,
        "Sinc": da.sinc,
        "Sinh": da.sinh,
        "Solve": da.linalg.solve,
        "Sqrt": da.sqrt,
        "Square": da.square,
        "Squeeze": da.squeeze,
        "Stack": da.stack,
        "Std": da.std,
        "Subtract": da.subtract,
        "Sum": da.sum,
        "Svd": da.linalg.svd,
        "Swapaxes": da.swapaxes,
        "Take": da.take,
        "Tan": da.tan,
        "Tanh": da.tanh,
        "Tensordot": da.tensordot,
        "Tile": da.tile,
        "Trace": da.trace,
        "Transpose": da.transpose,
        "TrueDivide": da.true_divide,
        "Trunc": da.trunc,
        "Union1d": da.union1d,
        "Unique": da.unique,
        "UnravelIndex": da.unravel_index,
        "Vdot": da.vdot,
        "Vstack": da.vstack,
        "Where": da.where,
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
