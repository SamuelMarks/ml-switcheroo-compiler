# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import Any

try:
    import cupy as cp
except ImportError:
    cp = None


_OP_MAPPING = None


def _get_op_mapping() -> dict[str, Any]:
    """Retrieve the operation mapping for the CuPy backend.

    Returns:
        dict: A dictionary mapping operation types to their implementations.
    """
    global _OP_MAPPING
    if _OP_MAPPING is not None:
        return _OP_MAPPING

    if cp is None:
        _OP_MAPPING = {}  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        return _OP_MAPPING

    _OP_MAPPING = {
        "Abs": cp.abs,
        "Add": cp.add,
        "All": cp.all,
        "Allclose": cp.allclose,
        "Angle": cp.angle,
        "Any": cp.any,
        "Append": cp.append,
        "ApplyOverAxes": cp.apply_over_axes,
        "Argmax": cp.argmax,
        "Argmin": cp.argmin,
        "Argwhere": cp.argwhere,
        "Average": cp.average,
        "Bincount": cp.bincount,
        "BitwiseAnd": cp.bitwise_and,
        "BitwiseNot": cp.bitwise_not,
        "BitwiseOr": cp.bitwise_or,
        "BitwiseXor": cp.bitwise_xor,
        "Block": cp.block,
        "BroadcastTo": cp.broadcast_to,
        "Cbrt": cp.cbrt,
        "Ceil": cp.ceil,
        "Cholesky": cp.linalg.cholesky,
        "Choose": cp.choose,
        "Clip": cp.clip,
        "Compress": cp.compress,
        "Concatenate": cp.concatenate,
        "Conj": cp.conj,
        "Copysign": cp.copysign,
        "Corrcoef": cp.corrcoef,
        "Cos": cp.cos,
        "Cosh": cp.cosh,
        "CountNonzero": cp.count_nonzero,
        "Cov": cp.cov,
        "Cumprod": cp.cumprod,
        "Cumsum": cp.cumsum,
        "Degrees": cp.degrees,
        "Delete": cp.delete,
        "Diag": cp.diag,
        "Diagonal": cp.diagonal,
        "Diff": cp.diff,
        "Digitize": cp.digitize,
        "Divide": cp.divide,
        "Divmod": cp.divmod,
        "Dot": cp.dot,
        "Dstack": cp.dstack,
        "Ediff1d": cp.ediff1d,
        "Einsum": cp.einsum,
        "Equal": cp.equal,
        "Exp": cp.exp,
        "Exp2": cp.exp2,
        "ExpandDims": cp.expand_dims,
        "Expm1": cp.expm1,
        "Extract": cp.extract,
        "Fabs": cp.fabs,
        "Fft": cp.fft.fft,
        "Fft2": cp.fft.fft2,
        "Fftfreq": cp.fft.fftfreq,
        "Fftn": cp.fft.fftn,
        "Fftshift": cp.fft.fftshift,
        "Fix": cp.fix,
        "Flatnonzero": cp.flatnonzero,
        "Flip": cp.flip,
        "Fliplr": cp.fliplr,
        "Flipud": cp.flipud,
        "FloatPower": cp.float_power,
        "Floor": cp.floor,
        "FloorDivide": cp.floor_divide,
        "Fmax": cp.fmax,
        "Fmin": cp.fmin,
        "Fmod": cp.fmod,
        "Frexp": cp.frexp,
        "Greater": cp.greater,
        "GreaterEqual": cp.greater_equal,
        "Hfft": cp.fft.hfft,
        "Hstack": cp.hstack,
        "Hypot": cp.hypot,
        "Ifft": cp.fft.ifft,
        "Ifft2": cp.fft.ifft2,
        "Ifftn": cp.fft.ifftn,
        "Ifftshift": cp.fft.ifftshift,
        "Ihfft": cp.fft.ihfft,
        "Imag": cp.imag,
        "Insert": cp.insert,
        "Inv": cp.linalg.inv,
        "Invert": cp.invert,
        "Irfft": cp.fft.irfft,
        "Irfft2": cp.fft.irfft2,
        "Irfftn": cp.fft.irfftn,
        "Isclose": cp.isclose,
        "Iscomplex": cp.iscomplex,
        "Isfinite": cp.isfinite,
        "Isin": cp.isin,
        "Isinf": cp.isinf,
        "Isnan": cp.isnan,
        "Isneginf": cp.isneginf,
        "Isposinf": cp.isposinf,
        "Isreal": cp.isreal,
        "Ldexp": cp.ldexp,
        "LeftShift": cp.left_shift,
        "Less": cp.less,
        "LessEqual": cp.less_equal,
        "Log": cp.log,
        "Log10": cp.log10,
        "Log2": cp.log2,
        "Logaddexp": cp.logaddexp,
        "Logaddexp2": cp.logaddexp2,
        "LogicalAnd": cp.logical_and,
        "LogicalNot": cp.logical_not,
        "LogicalOr": cp.logical_or,
        "LogicalXor": cp.logical_xor,
        "Lstsq": cp.linalg.lstsq,
        "Matmul": cp.matmul,
        "Max": cp.max,
        "Maximum": cp.maximum,
        "Mean": cp.mean,
        "Min": cp.min,
        "Minimum": cp.minimum,
        "Mod": cp.mod,
        "Moveaxis": cp.moveaxis,
        "Multiply": cp.multiply,
        "NanToNum": cp.nan_to_num,
        "Nanargmax": cp.nanargmax,
        "Nanargmin": cp.nanargmin,
        "Nancumprod": cp.nancumprod,
        "Nancumsum": cp.nancumsum,
        "Nanmax": cp.nanmax,
        "Nanmean": cp.nanmean,
        "Nanmedian": cp.nanmedian,
        "Nanmin": cp.nanmin,
        "Nanprod": cp.nanprod,
        "Nanstd": cp.nanstd,
        "Nansum": cp.nansum,
        "Nanvar": cp.nanvar,
        "Negative": cp.negative,
        "Nextafter": cp.nextafter,
        "Nonzero": cp.nonzero,
        "Norm": cp.linalg.norm,
        "NotEqual": cp.not_equal,
        "Outer": cp.outer,
        "Pad": cp.pad,
        "Percentile": cp.percentile,
        "Positive": cp.positive,
        "Power": cp.power,
        "Prod": cp.prod,
        "Qr": cp.linalg.qr,
        "Radians": cp.radians,
        "RavelMultiIndex": cp.ravel_multi_index,
        "Real": cp.real,
        "Reciprocal": cp.reciprocal,
        "Remainder": cp.remainder,
        "Repeat": cp.repeat,
        "Reshape": cp.reshape,
        "Rfft": cp.fft.rfft,
        "Rfft2": cp.fft.rfft2,
        "Rfftfreq": cp.fft.rfftfreq,
        "Rfftn": cp.fft.rfftn,
        "RightShift": cp.right_shift,
        "Rint": cp.rint,
        "Roll": cp.roll,
        "Round": cp.round,
        "Searchsorted": cp.searchsorted,
        "Select": cp.select,
        "Sign": cp.sign,
        "Signbit": cp.signbit,
        "Sin": cp.sin,
        "Sinc": cp.sinc,
        "Sinh": cp.sinh,
        "Solve": cp.linalg.solve,
        "Sqrt": cp.sqrt,
        "Square": cp.square,
        "Squeeze": cp.squeeze,
        "Stack": cp.stack,
        "Std": cp.std,
        "Subtract": cp.subtract,
        "Sum": cp.sum,
        "Svd": cp.linalg.svd,
        "Swapaxes": cp.swapaxes,
        "Take": cp.take,
        "Tan": cp.tan,
        "Tanh": cp.tanh,
        "Tensordot": cp.tensordot,
        "Tile": cp.tile,
        "Trace": cp.trace,
        "Transpose": cp.transpose,
        "TrueDivide": cp.true_divide,
        "Trunc": cp.trunc,
        "Union1d": cp.union1d,
        "Unique": cp.unique,
        "UnravelIndex": cp.unravel_index,
        "Vdot": cp.vdot,
        "Vstack": cp.vstack,
        "Where": cp.where,
    }
    return _OP_MAPPING


def execute_op(cls: type, op_type: str, *args: Any, **kwargs: Any) -> Any:
    """Execute an eager operation using the CuPy backend.

    Args:
        cls (type): The tensor class.
        op_type (str): The name of the operation to execute.
        *args (object): Positional arguments for the operation.
        **kwargs (object): Keyword arguments for the operation.

    Returns: Any: The result of the operation execution.

    Raises:
        BackendNotSupportedError: If the operation is not supported by the CuPy backend.
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
