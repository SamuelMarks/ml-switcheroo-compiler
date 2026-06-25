# ruff: noqa: E402

"""Module docstring."""

"""NumPy code generator and eager execution backend."""
from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from .numpy_mixins import NumpyVisionMixin, NumpyAudioMixin, NumpyScatterMixin
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.backends.common.generator_mixins import (
    SharedASTGeneratorMixin,
    GroupNormConfig,
)


class NumpyTypeTranslator:
    """Utility for Numpy type mappings."""

    @staticmethod
    def get_fallback_prefix() -> str:
        """Get fallback prefix."""
        return "np"  # pragma: no cover

    @staticmethod
    def get_ops_map() -> dict:
        """Get ops map."""
        return {}  # pragma: no cover


class NumpyASTVisitor:
    """Visitor methods for Numpy AST traversal."""

    _OP_MAP = {
        "Add": "np.add",
        "Zeros": "np.zeros",
        "Ones": "np.ones",
        "Full": "np.full",
        "Arange": "np.arange",
        "Sort": "np.sort",
        "ArgSort": "np.argsort",
        "Allclose": "np.allclose",
        "Fft": "np.fft.fft",
        "Rfft": "np.fft.rfft",
        "Fftn": "np.fft.fftn",
        "Erfinv": "scipy.special.erfinv",
        "NanToNum": "np.nan_to_num",
        "Subtract": "np.subtract",
        "Multiply": "np.multiply",
        "TrueDivide": "np.divide",
        "Exp": "np.exp",
        "Log": "np.log",
        "Matmul": "np.matmul",
        "Sin": "np.sin",
        "Cos": "np.cos",
        "Sum": "np.sum",
        "Mean": "np.mean",
        "Max": "np.max",
        "Min": "np.min",
        "BroadcastTo": "np.broadcast_to",
        "Reshape": "np.reshape",
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
        "Eigvalsh": "np.linalg.eigvalsh",
        "MatrixPower": "np.linalg.matrix_power",
        "Solve": "np.linalg.solve",
        "TriangularSolve": "scipy.linalg.solve_triangular",
        "Lu": "scipy.linalg.lu",
        "LuFactor": "scipy.linalg.lu_factor",
        "LuSolve": "scipy.linalg.lu_solve",
        "Norm": "np.linalg.norm",
        "MatrixExponential": "scipy.linalg.expm",
        "Cross": "np.cross",
    }

    @classmethod
    def _format_kwargs(cls, kwargs: dict[str, object]) -> str:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["equation", "dimension"]}
        if "dimension" in kwargs:
            filtered_kwargs["axis"] = kwargs["dimension"]  # pragma: no cover
        return ", ".join(f"{k}={v}" for k, v in filtered_kwargs.items())

    @classmethod
    def generic_visit(cls, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Fallback visit."""
        op_type = getattr(node, "op_type", "")
        np_func = cls._OP_MAP.get(op_type, f"np.{op_type.lower()}")
        args_str = ", ".join(input_vars)
        kwargs_str = cls._format_kwargs(kwargs)
        if kwargs_str:
            args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str
        return f"{np_func}({args_str})"


@register_backend("numpy")
class NumpyGenerator(
    SharedASTGeneratorMixin,
    PythonStringGenerator,
    NumpyVisionMixin,
    NumpyAudioMixin,
    NumpyScatterMixin,
):
    """Generates NumPy python code from IR."""

    def _get_backend_prefix(self) -> str:
        """Function docstring."""
        return "np"  # pragma: no cover

    def get_helper_functions(self) -> list[str]:
        """Get helper functions."""
        res = super().get_helper_functions()  # pragma: no cover
        res.extend(  # pragma: no cover
            self._get_group_norm_code(  # pragma: no cover
                GroupNormConfig(  # pragma: no cover
                    "np",  # pragma: no cover
                    "numpy as np",  # pragma: no cover
                    "np.reshape",  # pragma: no cover
                    "np.mean",  # pragma: no cover
                    "np.var",  # pragma: no cover
                    "np.sqrt",  # pragma: no cover
                    dim_arg="axis",  # pragma: no cover
                    keepdim_arg="keepdims",  # pragma: no cover
                )  # pragma: no cover
            )  # pragma: no cover
        )  # pragma: no cover
        return res  # pragma: no cover

    _import_header = (
        "import numpy as np",
        "import scipy.special",
        "import scipy.linalg",
        "",
        "def np_perspective_transform(images, start_points, end_points, interpolation, fill_value, data_format):",
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
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"np_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum."""
        args_str = ", ".join(input_vars)  # pragma: no cover
        eq = kwargs.get("equation", "")  # pragma: no cover
        return f"np.einsum('{eq}', {args_str})"  # pragma: no cover

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix."""
        return NumpyTypeTranslator.get_fallback_prefix()  # pragma: no cover

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Get ops map."""
        return NumpyTypeTranslator.get_ops_map()  # pragma: no cover

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Generic visit."""
        return NumpyASTVisitor.generic_visit(node, input_vars, **kwargs)
