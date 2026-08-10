# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import Any

_OP_MAPPING = None


def _get_op_mapping() -> dict:
    """Retrieve the operation mapping for the TensorFlow backend.

    Returns:
        dict: A dictionary mapping operation types to their TensorFlow implementations.
    """
    global _OP_MAPPING
    if _OP_MAPPING is not None:
        return _OP_MAPPING
    import tensorflow as tf

    _OP_MAPPING = {
        "Abs": tf.abs,
        "AccumulateN": tf.math.accumulate_n,
        "Acos": tf.acos,
        "Acosh": tf.acosh,
        "Add": tf.add,
        "AddN": tf.add_n,
        "Adjoint": tf.linalg.adjoint,
        "Angle": tf.math.angle,
        "Argmax": tf.argmax,
        "Argmin": tf.argmin,
        "Argsort": tf.argsort,
        "AsString": tf.as_string,
        "Asin": tf.asin,
        "Asinh": tf.asinh,
        "Atan": tf.atan,
        "Atan2": tf.atan2,
        "Atanh": tf.atanh,
        "BandPart": tf.linalg.band_part,
        "BandedTriangularSolve": tf.linalg.banded_triangular_solve,
        "BesselI0": tf.math.bessel_i0,
        "BesselI0e": tf.math.bessel_i0e,
        "BesselI1": tf.math.bessel_i1,
        "BesselI1e": tf.math.bessel_i1e,
        "Betainc": tf.math.betainc,
        "Bincount": tf.math.bincount,
        "Bitcast": tf.bitcast,
        "BooleanMask": tf.boolean_mask,
        "BroadcastTo": tf.broadcast_to,
        "CTCLoss": tf.nn.ctc_loss,
        "Cast": tf.cast,
        "Ceil": tf.math.ceil,
        "Cholesky": tf.linalg.cholesky,
        "CholeskySolve": tf.linalg.cholesky_solve,
        "Cond": tf.cond,
        "ConfusionMatrix": tf.math.confusion_matrix,
        "Conj": tf.math.conj,
        "ConvTranspose": tf.nn.conv_transpose,
        "Cos": tf.cos,
        "Cosh": tf.cosh,
        "CountNonzero": tf.math.count_nonzero,
        "Cross": tf.linalg.cross,
        "CtcLoss": tf.nn.ctc_loss,
        "Cumprod": tf.math.cumprod,
        "Cumsum": tf.cumsum,
        "CumulativeLogsumexp": tf.math.cumulative_logsumexp,
        "Det": tf.linalg.det,
        "Diag": tf.linalg.diag,
        "Digamma": tf.math.digamma,
        "Divide": tf.divide,
        "DivideNoNan": tf.math.divide_no_nan,
        "Dropout": tf.nn.dropout,
        "DynamicPartition": tf.dynamic_partition,
        "DynamicStitch": tf.dynamic_stitch,
        "EditDistance": tf.edit_distance,
        "Eig": tf.eig,
        "Eigh": tf.linalg.eigh,
        "EighTridiagonal": tf.linalg.eigh_tridiagonal,
        "Eigvals": tf.eigvals,
        "Eigvalsh": tf.linalg.eigvalsh,
        "Einsum": tf.einsum,
        "Equal": tf.equal,
        "Erf": tf.math.erf,
        "Erfc": tf.math.erfc,
        "Erfcinv": tf.math.erfcinv,
        "Erfinv": tf.math.erfinv,
        "Exp": tf.exp,
        "ExpandDims": tf.expand_dims,
        "Expm1": tf.math.expm1,
        "ExtractVolumePatches": tf.extract_volume_patches,
        "Fftnd": tf.fftnd,
        "Floor": tf.floor,
        "FractionalAvgPool": tf.nn.fractional_avg_pool,
        "FractionalMaxPool": tf.nn.fractional_max_pool,
        "Gather": tf.gather,
        "GatherNd": tf.gather_nd,
        "Greater": tf.greater,
        "GreaterEqual": tf.greater_equal,
        "Ifftnd": tf.ifftnd,
        "Igamma": tf.math.igamma,
        "Igammac": tf.math.igammac,
        "Imag": tf.math.imag,
        "InTopK": tf.math.in_top_k,
        "Inv": tf.linalg.inv,
        "InvertPermutation": tf.math.invert_permutation,
        "Irfftnd": tf.irfftnd,
        "IsNonDecreasing": tf.math.is_non_decreasing,
        "IsStrictlyIncreasing": tf.math.is_strictly_increasing,
        "L2Normalize": tf.linalg.l2_normalize,
        "Lbeta": tf.math.lbeta,
        "Less": tf.less,
        "LessEqual": tf.less_equal,
        "Lgamma": tf.math.lgamma,
        "Log": tf.math.log,
        "LogSigmoid": tf.math.log_sigmoid,
        "LogSoftmax": tf.math.log_softmax,
        "LogicalAnd": tf.logical_and,
        "LogicalNot": tf.logical_not,
        "LogicalOr": tf.logical_or,
        "LogicalXor": tf.math.logical_xor,
        "Lookup": tf.lookup,
        "Lstsq": tf.linalg.lstsq,
        "Lu": tf.linalg.lu,
        "LuMatrixInverse": tf.linalg.lu_matrix_inverse,
        "LuReconstruct": tf.linalg.lu_reconstruct,
        "LuSolve": tf.linalg.lu_solve,
        "Matmul": tf.matmul,
        "MatrixRank": tf.linalg.matrix_rank,
        "MatrixTranspose": tf.linalg.matrix_transpose,
        "Maximum": tf.maximum,
        "Minimum": tf.minimum,
        "Mod": tf.math.mod,
        "Multiply": tf.multiply,
        "MultiplyNoNan": tf.math.multiply_no_nan,
        "Ndtri": tf.math.ndtri,
        "Negative": tf.negative,
        "Nextafter": tf.math.nextafter,
        "Norm": tf.norm,
        "NotEqual": tf.not_equal,
        "OneHot": tf.one_hot,
        "Pad": tf.pad,
        "Pinv": tf.linalg.pinv,
        "Polygamma": tf.math.polygamma,
        "Polyval": tf.math.polyval,
        "Qr": tf.linalg.qr,
        "Rank": tf.rank,
        "Real": tf.math.real,
        "Reciprocal": tf.math.reciprocal,
        "ReciprocalNoNan": tf.math.reciprocal_no_nan,
        "ReduceEuclideanNorm": tf.math.reduce_euclidean_norm,
        "Repeat": tf.repeat,
        "Reshape": tf.reshape,
        "Rfftnd": tf.rfftnd,
        "Rint": tf.math.rint,
        "Roll": tf.roll,
        "Round": tf.round,
        "Rsqrt": tf.math.rsqrt,
        "Scan": tf.scan,
        "ScatterNd": tf.scatter_nd,
        "Searchsorted": tf.searchsorted,
        "Sigmoid": tf.sigmoid,
        "Sign": tf.sign,
        "Sin": tf.sin,
        "Sinh": tf.sinh,
        "Size": tf.size,
        "Slice": tf.slice,
        "Slogdet": tf.linalg.slogdet,
        "SobolSample": tf.math.sobol_sample,
        "Softmax": tf.math.softmax,
        "Softsign": tf.math.softsign,
        "Solve": tf.linalg.solve,
        "Sort": tf.sort,
        "SpaceToBatch": tf.space_to_batch,
        "SpaceToBatchND": tf.space_to_batch_nd,
        "Split": tf.split,
        "Sqrt": tf.sqrt,
        "Sqrtm": tf.linalg.sqrtm,
        "Square": tf.square,
        "SquaredDifference": tf.math.squared_difference,
        "Squeeze": tf.squeeze,
        "Stack": tf.stack,
        "StridedSlice": tf.strided_slice,
        "Subtract": tf.subtract,
        "Svd": tf.linalg.svd,
        "Tan": tf.tan,
        "Tanh": tf.tanh,
        "Tensordot": tf.tensordot,
        "Tile": tf.tile,
        "TopK": tf.math.top_k,
        "Trace": tf.linalg.trace,
        "Transpose": tf.transpose,
        "TriangularSolve": tf.linalg.triangular_solve,
        "TridiagonalMatmul": tf.linalg.tridiagonal_matmul,
        "TridiagonalSolve": tf.linalg.tridiagonal_solve,
        "Unique": tf.unique,
        "UnravelIndex": tf.unravel_index,
        "Unstack": tf.unstack,
        "Where": tf.where,
        "Xdivy": tf.math.xdivy,
        "Xlog1py": tf.math.xlog1py,
        "Xlogy": tf.math.xlogy,
        "ZeroFraction": tf.math.zero_fraction,
        "Zeta": tf.math.zeta,
    }
    return _OP_MAPPING


def execute_op(cls: type, op_type: str, *args: Any, **kwargs: Any) -> Any:
    """Execute an eager operation using the TensorFlow backend.

    Args:
        cls (type): The tensor class.
        op_type (str): The name of the operation to execute.
        *args (object): Positional arguments for the operation.
        **kwargs (object): Keyword arguments for the operation.

    Returns: Any: The result of the operation execution.

    Raises:
        BackendNotSupportedError: If the operation is not supported by the TensorFlow backend.
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
