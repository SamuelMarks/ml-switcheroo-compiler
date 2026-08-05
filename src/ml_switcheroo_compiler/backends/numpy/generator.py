# ruff: noqa: E501
"""Core abstractions and logic definitions for generator.py."""

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode

from .numpy_mixins import NumpyAudioVisitor, NumpyScatterVisitor, NumpyVisionVisitor


class NumpyTypeTranslator:
    """Utility for Numpy type mappings."""

    @staticmethod
    def get_fallback_prefix() -> str:
        """Retrieve the default fallback prefix string for NumPy operations.

        Returns:
            str: The fallback prefix string 'np'.
        """
        return "np"

    @staticmethod
    def get_ops_map() -> dict:
        """Evaluate get_ops_map operation.

        Returns:
        dict: Result.
        """
        return {
            "Infeed": "{0}",
            "Outfeed": "{0}",
            "AxisIndex": "0",
            "AllToAll": "{0}",
            "Pmax": "{0}",
            "Pmin": "{0}",
            "PsumScatter": "{0}",
            "Pswapaxes": "{0}",
            "Ppermute": "{0}",
            "Pshuffle": "{0}",
            "CreateToken": "0",
            "WithShardingConstraint": "{0}",
            "RandomCategorical": "np.argmax({1} + np.random.gumbel(size={1}.shape), axis={axis})",
            "MultivariateNormal": 'np.random.multivariate_normal({1}, {2}, size={shape}, method="{method}")',
            "Beta": "np.random.beta({1}, {2}, size={shape})",
            "Dirichlet": "np.random.dirichlet({1}, size={shape})",
            "Gamma": "np.random.gamma({1}, size={shape})",
            "RngBitGenerator": "np.random.randint(0, 255, size={shape})",
            "RngUniform": "np.random.uniform({0}, {1}, size={shape})",
        }


class NumpyASTVisitor:
    """Visitor methods for Numpy AST traversal."""

    _OP_MAP = {
        "BroadcastToRank": "np.expand_dims({0}, axis=tuple(range({0}.ndim, {1}))) if hasattr({0}, 'ndim') else {0}",
        "Collapse": "np.reshape({0}, (-1,) + {0}.shape[{1}:])",
        "ConvTransposeShapeTuple": "()",
        "IndexInDim": "np.take({0}, {1}, axis={2})",
        "RNNCellDeviceWrapper": "tf.nn.rnn_cell.DeviceWrapper",
        "RNNCellDropoutWrapper": "tf.nn.rnn_cell.DropoutWrapper",
        "RNNCellResidualWrapper": "tf.nn.rnn_cell.ResidualWrapper",
        "SparseBincount": "tf.sparse.bincount",
        "SparseCrossHashed": "tf.sparse.cross_hashed",
        "SparseExpandDims": "tf.sparse.expand_dims",
        "SparseEye": "tf.sparse.eye",
        "SparseFillEmptyRows": "tf.sparse.fill_empty_rows",
        "SparseMapValues": "tf.sparse.map_values",
        "SparseMask": "tf.sparse.mask",
        "SparseMaximum": "tf.sparse.maximum",
        "SparseMinimum": "tf.sparse.minimum",
        "SparseReduceMax": "tf.sparse.reduce_max",
        "SparseReduceSum": "tf.sparse.reduce_sum",
        "SparseReorder": "tf.sparse.reorder",
        "SparseResetShape": "tf.sparse.reset_shape",
        "SparseReshape": "tf.sparse.reshape",
        "SparseRetain": "tf.sparse.retain",
        "SparseSegmentMean": "tf.sparse.segment_mean",
        "SparseSegmentSqrtN": "tf.sparse.segment_sqrt_n",
        "SparseSegmentSum": "tf.sparse.segment_sum",
        "SparseSlice": "tf.sparse.slice",
        "SparseSoftmax": "tf.sparse.softmax",
        "SparseToIndicator": "tf.sparse.to_indicator",
        "SparseTranspose": "tf.sparse.transpose",
        "RaggedConstant": "tf.ragged.constant",
        "RaggedCrossHashed": "tf.ragged.cross_hashed",
        "RaggedRange": "tf.ragged.range",
        "RaggedRowSplitsToSegmentIds": "tf.ragged.row_splits_to_segment_ids",
        "RaggedSegmentIdsToRowSplits": "tf.ragged.segment_ids_to_row_splits",
        "RaggedStack": "tf.ragged.stack",
        "RaggedStackDynamicPartitions": "tf.ragged.stack_dynamic_partitions",
        "Trace": "np.trace({0}, offset={offset}, axis1={axis1}, axis2={axis2})",
        "Outer": "np.outer({0}, {1})",
        "Svdvals": "np.linalg.svd({0}, compute_uv=False)",
        "Tensordot": "np.tensordot({0}, {1}, axes={axes})",
        "Tensorinv": "np.linalg.tensorinv({0}, ind={ind})",
        "Tensorsolve": "np.linalg.tensorsolve({0}, {1}, axes={axes})",
        "Vecdot": "np.sum({0} * {1}, axis={axis})",
        "Adjoint": "tf.linalg.adjoint",
        "LuMatrixInverse": "np.linalg.inv(np.take_along_axis(np.matmul(np.tril({0}, -1) + np.eye({0}.shape[-1], dtype={0}.dtype), np.triu({0})), np.broadcast_to(np.expand_dims(np.argsort({1}, axis=-1), -1), {0}.shape), axis=-2))",
        "LuReconstruct": "np.take_along_axis(np.matmul(np.tril({0}, -1) + np.eye({0}.shape[-1], dtype={0}.dtype), np.triu({0})), np.broadcast_to(np.expand_dims(np.argsort({1}, axis=-1), -1), {0}.shape), axis=-2)",
        "BandPart": "tf.linalg.band_part",
        "CholeskySolve": "np.linalg.solve(np.matmul({0}, np.swapaxes({0}, -1, -2)), {1})",
        "Add": "np.add",
        "Zeros": "np.zeros",
        "Ones": "np.ones",
        "Full": "np.full",
        "Arange": "np.arange",
        "Sort": "np.sort",
        "ArgSort": "np.argsort",
        "Allclose": "np.allclose",
        "Fftnd": "np.fft.fftn({0})",
        "Ifftnd": "np.fft.ifftn({0})",
        "Rfftnd": "np.fft.rfftn({0})",
        "Irfftnd": "np.fft.irfftn({0})",
        "Fftshift": "np.fft.fftshift({0})",
        "Ifftshift": "np.fft.ifftshift({0})",
        "Fft": "np.fft.fft",
        "Rfft": "np.fft.rfft",
        "Fftn": "np.fft.fftn",
        "Ifft": "np.fft.ifft",
        "Ifftn": "np.fft.ifftn",
        "Rfftn": "np.fft.rfftn",
        "Irfftn": "np.fft.irfftn",
        "Ifft2": "np.fft.ifft2",
        "Rfft2": "np.fft.rfft2",
        "Irfft2": "np.fft.irfft2",
        "Hfft": "np.fft.hfft",
        "Rfftfreq": "np.fft.rfftfreq({0}, d={d})",
        "Sigmoid": "scipy.special.expit({0})",
        "Softmax": "scipy.special.softmax({0}, axis={axis})",
        "LogSoftmax": "scipy.special.log_softmax({0}, axis={axis})",
        "OneHot": "np.eye({depth})[{0}]",
        "Clip": "np.clip({0}, a_min={a_min}, a_max={a_max})",
        "Erfinv": "scipy.special.erfinv",
        "NanToNum": "np.nan_to_num",
        "Subtract": "np.subtract",
        "Multiply": "np.multiply",
        "TrueDivide": "np.divide",
        "Exp": "np.exp",
        "Log": "np.log",
        "Matmul": "np.matmul",
        "Sin": "np.sin",
        "Acos": "np.arccos",
        "Acosh": "np.arccosh",
        "Asin": "np.arcsin",
        "Asinh": "np.arcsinh",
        "IgammaGradA": "lambda a, x: a",
        "RandomGammaGrad": "lambda a, x: a",
        "Igamma": "scipy.special.gammainc",
        "Igammac": "scipy.special.gammaincc",
        "Polygamma": "scipy.special.polygamma",
        "Zeta": "scipy.special.zeta",
        "BesselI0e": "scipy.special.i0e",
        "BesselI1e": "scipy.special.i1e",
        "Betainc": "scipy.special.betainc",
        "Atan": "np.arctan",
        "Atan2": "np.arctan2",
        "Atanh": "np.arctanh",
        "Cos": "np.cos",
        "Sum": "np.sum",
        "Cummax": "np.maximum.accumulate",
        "Cummin": "np.minimum.accumulate",
        "Logcumsumexp": "scipy.special.logsumexp",
        "Cumprod": "np.cumprod",
        "Cumsum": "np.cumsum",
        "Cumlogsumexp": "np.logaddexp.accumulate",
        "Mean": "np.mean",
        "Max": "np.max",
        "Min": "np.min",
        "BroadcastTo": "np.broadcast_to",
        "Reshape": "np.reshape",
        "Reverse": "np.flip",
        "Transpose": "np.transpose",
        "Equal": "np.equal",
        "NotEqual": "np.not_equal",
        "Greater": "np.greater",
        "Less": "np.less",
        "Negative": "np.negative",
        "Cholesky": "np.linalg.cholesky",
        "Svd": "np.linalg.svd",
        "Qr": "np.linalg.qr",
        "Inv": "np.linalg.inv",
        "Pinv": "np.linalg.pinv",
        "Det": "np.linalg.det",
        "Slogdet": "np.linalg.slogdet",
        "Eigh": "np.linalg.eigh",
        "Eig": "np.linalg.eig",
        "Eigvalsh": "np.linalg.eigvalsh",
        "Cond": "np.linalg.cond",
        "Lstsq": "np.linalg.lstsq({0}, {1}, rcond={rcond})[0]",
        "MatrixNorm": "np.linalg.norm({0}, keepdims={keepdims})",
        "VectorNorm": "np.linalg.norm({0}, axis={axis}, keepdims={keepdims}, ord={ord})",
        "MatrixRank": "np.linalg.matrix_rank({0}, tol={tol}, hermitian={hermitian})",
        "MatrixTranspose": "np.swapaxes({0}, -1, -2)",
        "MultiDot": "np.linalg.multi_dot({0})",
        "Diagonal": "np.diagonal({0}, offset={offset}, axis1={axis1}, axis2={axis2})",
        "MatrixPower": "np.linalg.matrix_power",
        "Solve": "np.linalg.solve",
        "TridiagonalSolve": "scipy.linalg.solve_banded((1, 1), np.stack([{2}, {1}, {0}], axis=-2), {3})",
        "TridiagonalMatmul": (
            "np.expand_dims({1}, -1) * {3}"
            " + np.expand_dims(np.concatenate([np.zeros_like({0}[..., :1]), {0}[..., 1:]], axis=-1), -1) * np.concatenate([np.zeros_like({3}[..., :1, :]), {3}[..., :-1, :]], axis=-2)"
            " + np.expand_dims(np.concatenate([{2}[..., :-1], np.zeros_like({2}[..., -1:])], axis=-1), -1) * np.concatenate([{3}[..., 1:, :], np.zeros_like({3}[..., -1:, :])], axis=-2)"
        ),
        "TriangularSolve": "scipy.linalg.solve_triangular({0}, {1}, lower={lower}, unit_diagonal={unit_diagonal}, trans=1 if {adjoint} else 0, check_finite=False)",
        "Lu": "scipy.linalg.lu",
        "LuFactor": "scipy.linalg.lu_factor",
        "LuSolve": "scipy.linalg.lu_solve",
        "Poly": "np.poly",
        "Polyadd": "np.polyadd",
        "Polyder": "np.polyder",
        "Polydiv": "np.polydiv",
        "Polyfit": "np.polyfit",
        "Polyint": "np.polyint",
        "Polymul": "np.polymul",
        "Polysub": "np.polysub",
        "Polyval": "np.polyval",
        "Roots": "np.roots",
        "BroadcastedIota": "np.broadcasted_iota",
        "Bincount": "np.bincount",
        "Histogram": "np.histogram",
        "Histogram2d": "np.histogram2d",
        "HistogramBinEdges": "np.histogram_bin_edges",
        "Histogramdd": "np.histogramdd",
        "Geomspace": "np.geomspace",
        "Gradient": "np.gradient",
        "I0": "np.i0",
        "Mgrid": "np.mgrid",
        "Ogrid": "np.ogrid",
        "R_": "np.r_",
        "C_": "np.c_",
        "Fromfile": "np.fromfile",
        "Fromfunction": "np.fromfunction",
        "Fromiter": "np.fromiter",
        "Frompyfunc": "np.frompyfunc",
        "Fromstring": "np.fromstring",
        "Norm": "np.linalg.norm",
        "MatrixExponential": "scipy.linalg.expm",
        "Cross": "np.cross",
    }

    @classmethod
    def _format_kwargs(cls, kwargs: dict[str, object]) -> str:
        """Evaluate _format_kwargs operation.

        Args:
        kwargs (object): The kwargs parameter.

        Returns:
        str: Result.
        """
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["equation", "dimension"]}
        if "dimension" in kwargs:
            filtered_kwargs["axis"] = kwargs["dimension"]
        return ", ".join(f"{k}={v}" for k, v in filtered_kwargs.items())

    @classmethod
    def visit_TriInv(cls, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate Python code for a triangular matrix inverse operation.

        Args:
            node (object): The IR node representing the TriInv operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for triangular matrix inverse.
        """
        return f"np.linalg.inv({input_vars[0]})"

    @classmethod
    def visit_TruncateDiv(cls, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate Python code for a truncated division operation.

        Args:
            node (IRNode): The IR node representing the TruncateDiv operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for truncated division.
        """
        x, y = input_vars
        return f"np.trunc(np.divide({x}, {y}))"

    @classmethod
    def visit_TruncateMod(cls, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate Python code for a truncated modulo operation.

        Args:
            node (IRNode): The IR node representing the TruncateMod operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for truncated modulo.
        """
        x, y = input_vars
        return f"np.fmod({x}, {y})"

    @classmethod
    def generic_visit(cls, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate default NumPy code.

        Args:
            node (object): The IR node.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated default NumPy code string for the node.
        """
        if "dimension" in kwargs:
            return f"np.{node.op_type.lower()}({input_vars[0]}, axis={kwargs['dimension']})"
        op_type = getattr(node, "op_type", "")
        np_func = cls._OP_MAP.get(op_type, f"np.{op_type.lower()}")
        args_str = ", ".join(input_vars)
        kwargs_str = cls._format_kwargs(kwargs)
        if kwargs_str:
            args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str
        return f"{np_func}({args_str})"


@register_backend("numpy")
class NumpyGenerator(
    PythonStringGenerator,
):
    """Generate NumPy python code from IR."""

    def __init__(self, graph: object) -> None:
        """Initialize the NumPy generator with an IR graph.

        Args:
            graph (object): The IR graph to generate code from.
        """
        super().__init__(graph)
        self.visitors.extend(
            [
                *get_shared_ast_visitors(generator=self),
                NumpyVisionVisitor(),
                NumpyAudioVisitor(),
                NumpyScatterVisitor(),
            ]
        )

    @classmethod
    def get_numpy_rng(cls, *args: object, **kwargs: object) -> object:
        """Get a numpy random generator.

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: The rng.
        """
        import numpy as np

        return np.random.default_rng(*args, **kwargs)

    @classmethod
    def load(cls, *args: object, **kwargs: object) -> object:
        """Load data using NumPy's load function.

        Args:
            *args (object): Positional arguments for np.load.
            **kwargs (object): Keyword arguments for np.load.

        Returns:
            object: The loaded NumPy data.
        """
        import numpy as np

        return np.load(*args, **kwargs)

    @classmethod
    def save(cls, *args: object, **kwargs: object) -> None:
        """Save data using NumPy's save function.

        Args:
            *args (object): Positional arguments for np.save.
            **kwargs (object): Keyword arguments for np.save.: This function does not return a value.
        """
        import numpy as np

        np.save(*args, **kwargs)

    @classmethod
    def savez(cls, *args: object, **kwargs: object) -> None:
        """Save multiple arrays into a single file in uncompressed .npz format.

        Args:
            *args (object): Positional arguments for np.savez.
            **kwargs (object): Keyword arguments for np.savez.: This function does not return a value.
        """
        import numpy as np

        np.savez(*args, **kwargs)

    @classmethod
    def savez_compressed(cls, *args: object, **kwargs: object) -> None:
        """Save multiple arrays into a single file in compressed .npz format.

        Args:
            *args (object): Positional arguments for np.savez_compressed.
            **kwargs (object): Keyword arguments for np.savez_compressed.: This function does not return a value.
        """
        import numpy as np

        np.savez_compressed(*args, **kwargs)

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "np"

    def get_helper_functions(self) -> list[str]:
        """Evaluate get_helper_functions operation.

        Returns:
        object: Result.
        """
        res = []
        return res

    _import_header = (
        "import numpy as np",
        "import scipy.special",
        "import scipy.linalg",
        "import dataclasses",
        "",
        "@dataclasses.dataclass",
        "class PerspectiveConfig:",
        "    interpolation: str",
        "    fill_value: float",
        "    data_format: object",
        "",
        "def np_perspective_transform(images, start_points, end_points, config):",
        "    interpolation = config.interpolation",
        "    fill_value = config.fill_value",
        "    data_format = config.data_format",
        "    def get_h(src, dst):",
        "        A = np.zeros((*dst.shape[:-2], 8, 8), dtype=np.float32)",
        "        B = np.zeros((*dst.shape[:-2], 8), dtype=np.float32)",
        "        for i in range(4):",
        "            u, v = dst[..., i, 0], dst[..., i, 1]",
        "            x, y = src[..., i, 0], src[..., i, 1]",
        "            A[..., i*2, 0] = u",
        "            A[..., i*2, 1] = v",
        "            A[..., i*2, 2] = 1.0",
        "            A[..., i*2, 6] = -x * u",
        "            A[..., i*2, 7] = -x * v",
        "            A[..., i*2+1, 3] = u",
        "            A[..., i*2+1, 4] = v",
        "            A[..., i*2+1, 5] = 1.0",
        "            A[..., i*2+1, 6] = -y * u",
        "            A[..., i*2+1, 7] = -y * v",
        "            B[..., i*2] = x",
        "            B[..., i*2+1] = y",
        "        h = np.linalg.solve(A, B)",
        "        return np.reshape(np.concatenate([h, np.ones((*dst.shape[:-2], 1), dtype=np.float32)], axis=-1), (*dst.shape[:-2], 3, 3))",
        "    has_batch = images.ndim == MAGIC_VAL_4",
        "    if not has_batch:",
        "        images = np.expand_dims(images, 0)",
        "        start_points = np.expand_dims(start_points, 0)",
        "        end_points = np.expand_dims(end_points, 0)",
        '    if data_format == "channels_first":',
        "        images = np.transpose(images, (0, 2, 3, 1))",
        "    H_mat = get_h(start_points, end_points)",
        "    B_sz, H_dim, W_dim, C_dim = images.shape",
        "    y_grid, x_grid = np.meshgrid(np.arange(H_dim), np.arange(W_dim), indexing='ij')",
        "    y_grid = y_grid.astype(np.float32)",
        "    x_grid = x_grid.astype(np.float32)",
        "    coords = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)",
        "    ",
        "    out_list = []",
        "    for b in range(B_sz):",
        "        t_coords = np.matmul(coords, np.transpose(H_mat[b]))",
        "        t_coords = t_coords / t_coords[..., 2:3]",
        "        src_x = t_coords[..., 0]",
        "        src_y = t_coords[..., 1]",
        "        c_list = []",
        "        for c in range(C_dim):",
        '            order = 1 if interpolation == "bilinear" else 0',
        "            from scipy.ndimage import map_coordinates",
        "            c_res = map_coordinates(images[b, ..., c], [src_y, src_x], order=order, mode='constant', cval=fill_value)",
        "            c_list.append(c_res)",
        "        out_list.append(np.stack(c_list, axis=-1))",
        "    out = np.stack(out_list, axis=0)",
        "    ",
        '    if data_format == "channels_first":',
        "        out = np.transpose(out, (0, 3, 1, 2))",
        "    if not has_batch:",
        "        out = out[0]",
        "    return out",
        "",
        "def np_power_iteration(w, num_iters, u=None):",
        "    if u is None:",
        "        u = np.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype)",
        "    for _ in range(num_iters):",
        "        w_t = np.swapaxes(w, -1, -2)",
        "        v = np.matmul(w_t, u)",
        "        v = v / (np.linalg.norm(v, axis=-2, keepdims=True) + 1e-12)",
        "        u = np.matmul(w, v)",
        "        u = u / (np.linalg.norm(u, axis=-2, keepdims=True) + 1e-12)",
        "    sigma = np.matmul(np.swapaxes(u, -1, -2), np.matmul(w, v))",
        "    return np.squeeze(v, -1), np.squeeze(u, -1), np.squeeze(np.squeeze(sigma, -1), -1)",
    )

    def visit_PowerIteration(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate Python code for executing power iteration on a matrix.

        Args:
            node (IRNode): The IR node representing the PowerIteration operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for power iteration.
        """
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"np_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generate Python code for Einstein summation convention.

        Args:
            node (IRNode): The IR node representing the Einsum operation.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional keyword arguments.

        Returns:
            str: The generated NumPy code string for the einsum operation.
        """
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"np.einsum('{eq}', {args_str})"

    def get_fallback_prefix(self) -> str:
        """Retrieve the default fallback prefix string for NumPy operations from the generator.

        Returns:
            str: The fallback prefix string 'np'.
        """
        return NumpyTypeTranslator.get_fallback_prefix()

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Evaluate get_ops_map operation.

        Args:
            kwargs (dict): The kwargs parameter.

        Returns:
            dict: Result.
        """
        res = super().get_ops_map(kwargs)
        res.update(NumpyTypeTranslator.get_ops_map())
        return res
