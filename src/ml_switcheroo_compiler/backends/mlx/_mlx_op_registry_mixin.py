"""Operator registry mapping mixin for the MLX code generator."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)


class MLXOpRegistryMixin:
    """Op Registry Mixin."""

    _SIMPLE_OPS_MAP = {
        "BroadcastInDim": "{0}.broadcast_in_dim({1}, {2})",
        "ConvGeneralDilated": "{0}.conv_general_dilated({1}, {2})",
        "DotGeneral": "{0}.dot_general({1}, {2})",
        "DynamicSlice": "{0}.dynamic_slice({1}, {2})",
        "DynamicUpdateSlice": "{0}.dynamic_update_slice({1}, {2})",
        "Pmean": "{0}.pmean({1})",
        "Psum": "{0}.psum({1})",
        "Matmul": "mx.matmul({0}, {1})",
        "Trace": "mx.trace({0}, offset={offset}, axis1={axis1}, axis2={axis2})",
        "Outer": "mx.outer({0}, {1})",
        "Svdvals": "mx.linalg.svd({0}, compute_uv=False)",
        "Tensordot": "mx.tensordot({0}, {1}, axes={axes})",
        "Tensorinv": "mx.linalg.tensorinv({0}, ind={ind})",
        "Tensorsolve": "mx.linalg.tensorsolve({0}, {1}, axes={axes})",
        "Vecdot": "mx.sum({0} * {1}, axis={axis})",
        "Adjoint": "tf.linalg.adjoint",
        "LuMatrixInverse": (
            "mx.linalg.inv(mx.take_along_axis(mx.matmul(mx.tril({0}, -1) "
            "+ mx.eye({0}.shape[-1], dtype={0}.dtype), mx.triu({0})), "
            "mx.broadcast_to(mx.expand_dims(mx.argsort({1}, axis=-1), -1), {0}.shape), axis=-2))"
        ),
        "LuReconstruct": (
            "mx.take_along_axis(mx.matmul(mx.tril({0}, -1) "
            "+ mx.eye({0}.shape[-1], dtype={0}.dtype), mx.triu({0})), "
            "mx.broadcast_to(mx.expand_dims(mx.argsort({1}, axis=-1), -1), {0}.shape), axis=-2)"
        ),
        "BandPart": "tf.linalg.band_part",
        "CholeskySolve": "mx.linalg.solve(mx.matmul({0}, mx.swapaxes({0}, -1, -2)), {1})",
        "Dot": "mx.dot({0}, {1})",
        "BroadcastTo": "mx.broadcast_to({0}, {shape})",
        "Reshape": "mx.reshape({0}, {shape})",
        "TruncateDiv": "mx.trunc(mx.divide({0}, {1}))",
        "TruncateMod": "mx.remainder({0}, {1})",
        "TrueDivide": "mx.divide({0}, {1})",
        "Relu": "nn.relu({0})",
        "Sigmoid": "mx.sigmoid({0})",
        "Softmax": "mx.softmax({0}, axis={axis})",
        "LogSoftmax": "mx.log_softmax({0}, axis={axis})",
        "OneHot": "mx.eye({depth})[{0}]",
        "Clip": "mx.clip({0}, a_min={a_min}, a_max={a_max})",
        "Arange": "mx.arange({0})",
        "Frombuffer": "mx.frombuffer({0}, dtype={dtype}, count={count}, offset={offset})",
        "Sort": "mx.sort({0}, axis={dimension})",
        "ArgSort": "mx.argsort({0}, axis={dimension})",
        "Allclose": "mx.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
        "Fftnd": "mx.fft.fftn({0})",
        "Ifftnd": "mx.fft.ifftn({0})",
        "Rfftnd": "mx.fft.rfftn({0})",
        "Irfftnd": "mx.fft.irfftn({0})",
        "Fftshift": "mx.fft.fftshift({0})",
        "Ifftshift": "mx.fft.ifftshift({0})",
        "Fft": "mx.fft.fft({0})",
        "Rfft": "mx.fft.rfft({0})",
        "Fftn": "mx.fft.fftn({0})",
        "Ifft": "mx.fft.ifft({0})",
        "Ifftn": "mx.fft.ifftn({0})",
        "Rfftn": "mx.fft.rfftn({0})",
        "Irfftn": "mx.fft.irfftn({0})",
        "Ifft2": "mx.fft.ifft2({0})",
        "Rfft2": "mx.fft.rfft2({0})",
        "Irfft2": "mx.fft.irfft2({0})",
        "Hfft": "mx.fft.hfft({0})",
        "Rfftfreq": "mx.fft.rfftfreq({0}, d={d})",
        "Erfinv": "mx.erfinv({0})",
        "NanToNum": (
            "mx.where(mx.isnan({0}), {nan}, mx.where(mx.isposinf({0}), {posinf}, "
            "mx.where(mx.isneginf({0}), {neginf}, {0})))"
        ),
        "AssignVariable": "{0}",
        "StopGradient": "mx.stop_gradient({0})",
        "Resize": "mx.image.resize({0}, {size})",
        "AffineGrid": "mx.image.affine_grid({0}, {size})",
        "GridSample": "mx.image.grid_sample({0}, {1})",
        "DrawBoundingBoxes": "{0}",
        "RgbToYiq": "mx.image.rgb_to_yiq({0})",
        "YiqToRgb": "mx.image.yiq_to_rgb({0})",
        "RgbToYuv": "mx.image.rgb_to_yuv({0})",
        "YuvToRgb": "mx.image.yuv_to_rgb({0})",
        "Fft2d": "mx.fft.fft2({0}, s={s}, axes={axes})",
        "Ifft2d": "mx.fft.ifft2({0}, s={s}, axes={axes})",
        "Fft3d": "mx.fft.fftn({0}, s={s}, axes={axes})",
        "Ifft3d": "mx.fft.ifftn({0}, s={s}, axes={axes})",
        "Rfft2d": "mx.fft.rfft2({0}, s={s}, axes={axes})",
        "Rfft3d": "mx.fft.rfftn({0}, s={s}, axes={axes})",
        "Irfft": "mx.fft.irfft({0}, n={n}, axis={axis})",
        "Irfft2d": "mx.fft.irfft2({0}, s={s}, axes={axes})",
        "Irfft3d": "mx.fft.irfftn({0}, s={s}, axes={axes})",
        "HannWindow": "mx.linalg.hann_window({window_length})",
        "ReadVariable": "{0}",
        "Cholesky": "mx.linalg.cholesky({0})",
        "Svd": "mx.linalg.svd({0})",
        "Qr": "mx.linalg.qr({0})",
        "Inv": "mx.linalg.inv({0})",
        "Solve": "mx.linalg.solve({0}, {1})",
        "Eigvalsh": "mx.linalg.eigvalsh({0})",
        "Cond": "mx.linalg.cond({0}, p={p})",
        "Lstsq": "mx.linalg.lstsq({0}, {1}, rcond={rcond})[0]",
        "MatrixNorm": "mx.linalg.norm({0}, keepdims={keepdims})",
        "VectorNorm": "mx.linalg.norm({0}, axis={axis}, keepdims={keepdims}, ord={ord})",
        "MatrixRank": "mx.linalg.matrix_rank({0}, tol={tol}, hermitian={hermitian})",
        "MatrixTranspose": "mx.swapaxes({0}, -1, -2)",
        "MultiDot": "mx.linalg.multi_dot({0})",
        "Diagonal": "mx.diagonal({0}, offset={offset}, axis1={axis1}, axis2={axis2})",
        "Eigh": "mx.linalg.eigh({0})",
        "Eig": "mx.linalg.eig({0})",
        "MatrixPower": "mx.linalg.matrix_power({0}, {n})",
        "Norm": "mx.linalg.norm({0}, ord={ord}, axis={axis}, keepdims={keepdims})",
        "Det": "mx.linalg.det({0})",
        "Slogdet": "mx.linalg.slogdet({0})",
        "Poly": "mx.poly({0})",
        "Polyadd": "mx.polyadd({0})",
        "Polyder": "mx.polyder({0})",
        "Polydiv": "mx.polydiv({0})",
        "Polyfit": "mx.polyfit({0})",
        "Polyint": "mx.polyint({0})",
        "Polymul": "mx.polymul({0})",
        "Polysub": "mx.polysub({0})",
        "Polyval": "mx.polyval({0})",
        "Roots": "mx.roots({0})",
        "BroadcastedIota": "mx.broadcasted_iota({0})",
        "Bincount": "mx.bincount({0})",
        "Histogram": "mx.histogram({0})",
        "Histogram2d": "mx.histogram2d({0})",
        "HistogramBinEdges": "mx.histogram_bin_edges({0})",
        "Histogramdd": "mx.histogramdd({0})",
        "Geomspace": "mx.geomspace({0})",
        "Gradient": "mx.gradient({0})",
        "I0": "mx.i0({0})",
        "Mgrid": "mx.mgrid({0})",
        "Ogrid": "mx.ogrid({0})",
        "R_": "mx.r_({0})",
        "C_": "mx.c_({0})",
        "Fromfile": "mx.fromfile({0})",
        "Fromfunction": "mx.fromfunction({0})",
        "Fromiter": "mx.fromiter({0})",
        "Frompyfunc": "mx.frompyfunc({0})",
        "Fromstring": "mx.fromstring({0})",
        "Pinv": "mx.linalg.pinv({0})",
        "TriInv": "mx.linalg.inv({0})",
        "TridiagonalSolve": (
            "mx.linalg.solve(mx.diag({1}) + mx.diag({0}[..., 1:], k=-1) + mx.diag({2}[..., :-1], k=1), {3})"
        ),
        "TridiagonalMatmul": (
            "mx.expand_dims({1}, -1) * {3} "
            "+ mx.expand_dims(mx.concatenate([mx.zeros_like({0}[..., :1]), {0}[..., 1:]], axis=-1), -1) "
            "* mx.concatenate([mx.zeros_like({3}[..., :1, :]), {3}[..., :-1, :]], axis=-2) "
            "+ mx.expand_dims(mx.concatenate([{2}[..., :-1], mx.zeros_like({2}[..., -1:])], axis=-1), -1) "
            "* mx.concatenate([{3}[..., 1:, :], mx.zeros_like({3}[..., -1:, :])], axis=-2)"
        ),
        "TriangularSolve": (
            "mx.linalg.solve_triangular({0}.swapaxes(-1, -2).conjugate() if {adjoint} else {0}, {1}, "
            "upper=not {lower} if not {adjoint} else {lower})"
        ),
        "Lu": "mx.linalg.lu({0})",
        "LuFactor": "mx.linalg.lu_factor({0})",
        "LuSolve": "mx.linalg.lu_solve({0}, {1})",
        "MatrixExponential": "mx.linalg.matrix_exp({0})",
        "Cross": "mx.linalg.cross({0}, {1}, axis={axis})",
        "GatherNd": "mx.gather_nd({0}, {1})",
        "ScatterNd": "mx.scatter_nd({0}, {1}, {shape})",
        "TensorScatterUpdate": (
            "(lambda c, i, u: c.at[tuple(i[..., d] for d in range(i.shape[-1]))]"
            ".add(u - c[tuple(i[..., d] for d in range(i.shape[-1]))]))(mx.asarray({0}), {1}, {2})"
        ),
        "TensorScatterAdd": (
            "(lambda c, i, u: c.at[tuple(i[..., d] for d in range(i.shape[-1]))].add(u))(mx.asarray({0}), {1}, {2})"
        ),
        "TensorScatterMax": (
            "(lambda c, i, u: c.at[tuple(i[..., d] for d in range(i.shape[-1]))].maximum(u))(mx.asarray({0}), {1}, {2})"
        ),
        "TensorScatterMin": (
            "(lambda c, i, u: c.at[tuple(i[..., d] for d in range(i.shape[-1]))].minimum(u))(mx.asarray({0}), {1}, {2})"
        ),
        "Scatter": "mx.scatter({0}, {1}, {2}, {dim})",
        "ScatterAdd": "mx.scatter_add({0}, {1}, {2}, {dim})",
        "Gather": "mx.take({0}, {1}, {dim})",
        "PRNGKey": "mx.random.key({seed})",
        "RandomSplit": "mx.random.split({0}, {num})",
        "RandomUniform": "mx.random.uniform(low={minval}, high={maxval}, shape={shape}, key={0})",
        "RandomNormal": "mx.random.normal(shape={shape}, key={0})",
        "RandomRandint": "mx.random.randint(low={minval}, high={maxval}, shape={shape}, key={0})",
        "RandomBernoulli": "mx.random.bernoulli(p={p}, shape={shape}, key={0})",
    }

    def visit_AllReduce(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for AllReduce.

        Args:
            node: The node.
            input_vars: The input vars.
            **kwargs: Extra args.

        Returns:
            str: Generated code.
        """
        tensor: str = input_vars[0]
        return f"mx.distributed.all_sum({tensor})"

    def visit_AllGather(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for AllGather.

        Args:
            node: The node.
            input_vars: The input vars.
            **kwargs: Extra args.

        Returns:
            str: Generated code.
        """
        tensor: str = input_vars[0]
        return f"mx.distributed.all_gather({tensor})"

    def visit_AllToAll(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for AllToAll.

        Args:
            node: The node.
            input_vars: The input vars.
            **kwargs: Extra args.

        Returns:
            str: Generated code.
        """
        tensor: str = input_vars[0]
        return f"mx.distributed.all_to_all({tensor}) if hasattr(mx.distributed, 'all_to_all') else {tensor}"

    def visit_ReduceScatter(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for ReduceScatter.

        Args:
            node: The node.
            input_vars: The input vars.
            **kwargs: Extra args.

        Returns:
            str: Generated code.
        """
        tensor: str = input_vars[0]
        return f"mx.distributed.recv({tensor}) if hasattr(mx.distributed, 'recv') else {tensor}"
