# ruff: noqa: E501
"""Backend utilities."""

_OP_MAPPING = None


def _get_op_mapping() -> dict:
    """Retrieve the operation mapping for the Keras backend.

    Returns:
        dict: A dictionary mapping operation types to their implementations.
    """
    global _OP_MAPPING
    if _OP_MAPPING is not None:
        return _OP_MAPPING
    import keras.ops as keras_ops

    _OP_MAPPING = {
        "Abs": keras_ops.abs,
        "Add": keras_ops.add,
        "All": keras_ops.all,
        "Angle": keras_ops.angle,
        "Any": keras_ops.any,
        "Append": keras_ops.append,
        "Argmax": keras_ops.argmax,
        "Argmin": keras_ops.argmin,
        "Argpartition": keras_ops.argpartition,
        "Argsort": keras_ops.argsort,
        "AssociativeScan": keras_ops.associative_scan,
        "Average": keras_ops.average,
        "Bincount": keras_ops.bincount,
        "BitwiseAnd": keras_ops.bitwise_and,
        "BitwiseNot": keras_ops.bitwise_not,
        "BitwiseOr": keras_ops.bitwise_or,
        "BitwiseXor": keras_ops.bitwise_xor,
        "BroadcastTo": keras_ops.broadcast_to,
        "CTCLoss": keras_ops.ctc_loss,
        "Cast": keras_ops.cast,
        "Ceil": keras_ops.ceil,
        "Cholesky": keras_ops.cholesky,
        "Clip": keras_ops.clip,
        "Concatenate": keras_ops.concatenate,
        "Cond": keras_ops.cond,
        "Conj": keras_ops.conj,
        "ConvTranspose": keras_ops.conv_transpose,
        "Correlate": keras_ops.correlate,
        "Cos": keras_ops.cos,
        "Cosh": keras_ops.cosh,
        "CountNonzero": keras_ops.count_nonzero,
        "Cross": keras_ops.cross,
        "CtcLoss": keras_ops.ctc_loss,
        "Cumprod": keras_ops.cumprod,
        "Cumsum": keras_ops.cumsum,
        "Det": keras_ops.det,
        "Diag": keras_ops.diag,
        "Diagflat": keras_ops.diagflat,
        "Diagonal": keras_ops.diagonal,
        "Diff": keras_ops.diff,
        "Digitize": keras_ops.digitize,
        "Divide": keras_ops.divide,
        "DivideNoNan": keras_ops.divide_no_nan,
        "Dot": keras_ops.dot,
        "Eig": keras_ops.eig,
        "Eigh": keras_ops.eigh,
        "Einsum": keras_ops.einsum,
        "Equal": keras_ops.equal,
        "Erf": keras_ops.erf,
        "Erfinv": keras_ops.erfinv,
        "Exp": keras_ops.exp,
        "Exp2": keras_ops.exp2,
        "ExpandDims": keras_ops.expand_dims,
        "Expm1": keras_ops.expm1,
        "Fft": keras_ops.fft,
        "Fft2": keras_ops.fft2,
        "Flip": keras_ops.flip,
        "Floor": keras_ops.floor,
        "FloorDivide": keras_ops.floor_divide,
        "GetItem": keras_ops.get_item,
        "Greater": keras_ops.greater,
        "GreaterEqual": keras_ops.greater_equal,
        "HardSilu": keras_ops.hard_silu,
        "HardSwish": keras_ops.hard_swish,
        "Hstack": keras_ops.hstack,
        "Ifft2": keras_ops.ifft2,
        "Imag": keras_ops.imag,
        "InTopK": keras_ops.in_top_k,
        "Inner": keras_ops.inner,
        "Inv": keras_ops.inv,
        "Irfft": keras_ops.irfft,
        "Isclose": keras_ops.isclose,
        "Isfinite": keras_ops.isfinite,
        "Isinf": keras_ops.isinf,
        "Isnan": keras_ops.isnan,
        "Istft": keras_ops.istft,
        "LeftShift": keras_ops.left_shift,
        "Less": keras_ops.less,
        "LessEqual": keras_ops.less_equal,
        "Log": keras_ops.log,
        "Log10": keras_ops.log10,
        "Log2": keras_ops.log2,
        "LogSigmoid": keras_ops.log_sigmoid,
        "LogSoftmax": keras_ops.log_softmax,
        "Logaddexp": keras_ops.logaddexp,
        "LogicalAnd": keras_ops.logical_and,
        "LogicalNot": keras_ops.logical_not,
        "LogicalOr": keras_ops.logical_or,
        "LogicalXor": keras_ops.logical_xor,
        "Logsumexp": keras_ops.logsumexp,
        "Lstsq": keras_ops.lstsq,
        "LuFactor": keras_ops.lu_factor,
        "Matmul": keras_ops.matmul,
        "Max": keras_ops.max,
        "Maximum": keras_ops.maximum,
        "Mean": keras_ops.mean,
        "Min": keras_ops.min,
        "Minimum": keras_ops.minimum,
        "Mod": keras_ops.mod,
        "Moveaxis": keras_ops.moveaxis,
        "Multiply": keras_ops.multiply,
        "NanToNum": keras_ops.nan_to_num,
        "Negative": keras_ops.negative,
        "Nonzero": keras_ops.nonzero,
        "Norm": keras_ops.norm,
        "NotEqual": keras_ops.not_equal,
        "OneHot": keras_ops.one_hot,
        "Outer": keras_ops.outer,
        "Pad": keras_ops.pad,
        "Polar": keras_ops.polar,
        "Power": keras_ops.power,
        "Prod": keras_ops.prod,
        "Qr": keras_ops.qr,
        "Quantile": keras_ops.quantile,
        "Real": keras_ops.real,
        "Reciprocal": keras_ops.reciprocal,
        "Repeat": keras_ops.repeat,
        "Reshape": keras_ops.reshape,
        "Rfft": keras_ops.rfft,
        "RightShift": keras_ops.right_shift,
        "Roll": keras_ops.roll,
        "Round": keras_ops.round,
        "Rsqrt": keras_ops.rsqrt,
        "Scan": keras_ops.scan,
        "Scatter": keras_ops.scatter,
        "Searchsorted": keras_ops.searchsorted,
        "Select": keras_ops.select,
        "Sigmoid": keras_ops.sigmoid,
        "Sign": keras_ops.sign,
        "Signbit": keras_ops.signbit,
        "Sin": keras_ops.sin,
        "Sinh": keras_ops.sinh,
        "Size": keras_ops.size,
        "Slice": keras_ops.slice,
        "Slogdet": keras_ops.slogdet,
        "Softmax": keras_ops.softmax,
        "Softsign": keras_ops.softsign,
        "Solve": keras_ops.solve,
        "Sort": keras_ops.sort,
        "SparsePlus": keras_ops.sparse_plus,
        "SparseSigmoid": keras_ops.sparse_sigmoid,
        "Split": keras_ops.split,
        "Sqrt": keras_ops.sqrt,
        "Square": keras_ops.square,
        "Squareplus": keras_ops.squareplus,
        "Squeeze": keras_ops.squeeze,
        "Stack": keras_ops.stack,
        "Std": keras_ops.std,
        "Stft": keras_ops.stft,
        "Subtract": keras_ops.subtract,
        "Sum": keras_ops.sum,
        "Svd": keras_ops.svd,
        "Swapaxes": keras_ops.swapaxes,
        "Switch": keras_ops.switch,
        "Take": keras_ops.take,
        "TakeAlongAxis": keras_ops.take_along_axis,
        "Tan": keras_ops.tan,
        "Tanh": keras_ops.tanh,
        "Tensordot": keras_ops.tensordot,
        "Tile": keras_ops.tile,
        "TopK": keras_ops.top_k,
        "Trace": keras_ops.trace,
        "Transpose": keras_ops.transpose,
        "TrueDivide": keras_ops.true_divide,
        "Trunc": keras_ops.trunc,
        "UnravelIndex": keras_ops.unravel_index,
        "Unstack": keras_ops.unstack,
        "Vdot": keras_ops.vdot,
        "Vstack": keras_ops.vstack,
        "Where": keras_ops.where,
    }
    return _OP_MAPPING


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute an eager operation using the Keras backend.

    Args:
        cls (type): The tensor class.
        op_type (str): The name of the operation to execute.
        *args (object): Positional arguments for the operation.
        **kwargs (object): Keyword arguments for the operation.

    Returns:
        object: The result of the operation execution.

    Raises:
        BackendNotSupportedError: If the operation is not supported by the Keras backend.
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
