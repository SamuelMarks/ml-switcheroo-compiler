# ruff: noqa: E501
"""Mixins."""


class PyTorchScatterVisitor:
    """Mixin."""

    def visit_TensorScatterUpdate(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterUpdate nodes."""
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]})"

    def visit_TensorScatterAdd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterAdd nodes."""
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]}, accumulate=True)"

    def visit_TensorScatterMax(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate tensor scatter max."""
        return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amax', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMin(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate tensor scatter min."""
        return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amin', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"


class PyTorchDistributedVisitor:
    """Mixin."""

    def visit_all_gather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_gather."""
        tensor = input_vars[0]
        return f"torch.distributed.all_gather_into_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_reduce_scatter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for reduce_scatter."""
        tensor = input_vars[0]
        return f"torch.distributed.reduce_scatter_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_all_reduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_reduce."""
        tensor = input_vars[0]
        return f"torch.distributed.all_reduce({tensor})"


class PyTorchLinalgMixin:
    """Linalg."""

    def _get_linalg_ops(self, kwargs: dict) -> dict[str, str]:
        return _PYTORCH_OP_REGISTRY

    """Linalg Mixin."""


_PYTORCH_OP_REGISTRY = {
    "BroadcastInDim": "{0}.broadcast_in_dim({1}, {2})",
    "ConvGeneralDilated": "{0}.conv_general_dilated({1}, {2})",
    "DotGeneral": "{0}.dot_general({1}, {2})",
    "DynamicSlice": "{0}.dynamic_slice({1}, {2})",
    "DynamicUpdateSlice": "{0}.dynamic_update_slice({1}, {2})",
    "Pmean": "{0}.pmean({1})",
    "Psum": "{0}.psum({1})",
    "Matmul": "torch.matmul({0}, {1})",
    "Trace": "torch.diagonal({0}, offset={offset}, dim1={axis1}, dim2={axis2}).sum(-1)",
    "Outer": "torch.outer({0}, {1})",
    "Svdvals": "torch.linalg.svdvals({0})",
    "Tensordot": "torch.tensordot({0}, {1}, dims={axes})",
    "Tensorinv": "torch.linalg.tensorinv({0}, ind={ind})",
    "Tensorsolve": "torch.linalg.tensorsolve({0}, {1}, dims={axes})",
    "Vecdot": "torch.linalg.vecdot({0}, {1}, dim={axis})",
    "Adjoint": "tf.linalg.adjoint",
    "LuMatrixInverse": "torch.linalg.inv(torch.take_along_dim(torch.matmul(torch.tril({0}, -1) + torch.eye({0}.shape[-1], dtype={0}.dtype, device={0}.device), torch.triu({0})), torch.broadcast_to(torch.unsqueeze(torch.argsort({1}, dim=-1), -1), {0}.shape), dim=-2))",
    "LuReconstruct": "torch.take_along_dim(torch.matmul(torch.tril({0}, -1) + torch.eye({0}.shape[-1], dtype={0}.dtype, device={0}.device), torch.triu({0})), torch.broadcast_to(torch.unsqueeze(torch.argsort({1}, dim=-1), -1), {0}.shape), dim=-2)",
    "BandPart": "tf.linalg.band_part",
    "CholeskySolve": "torch.cholesky_solve({1}, {0})",
    "Dot": "torch.dot({0}, {1})",
    "Fftnd": "torch.fft.fftn({0})",
    "Ifftnd": "torch.fft.ifftn({0})",
    "Rfftnd": "torch.fft.rfftn({0})",
    "Irfftnd": "torch.fft.irfftn({0})",
    "Fftshift": "torch.fft.fftshift({0})",
    "Ifftshift": "torch.fft.ifftshift({0})",
    "Fft": "torch.fft.fft({0})",
    "Rfft": "torch.fft.rfft({0})",
    "Fftn": "torch.fft.fftn({0})",
    "Ifft": "torch.fft.ifft({0})",
    "Ifftn": "torch.fft.ifftn({0})",
    "Rfftn": "torch.fft.rfftn({0})",
    "Irfftn": "torch.fft.irfftn({0})",
    "Ifft2": "torch.fft.ifft2({0})",
    "Rfft2": "torch.fft.rfft2({0})",
    "Irfft2": "torch.fft.irfft2({0})",
    "Hfft": "torch.fft.hfft({0})",
    "Rfftfreq": "torch.fft.rfftfreq({0}, d={d})",
    "Cholesky": "torch.linalg.cholesky({0})",
    "Svd": "torch.linalg.svd({0})",
    "Qr": "torch.linalg.qr({0})",
    "Inv": "torch.linalg.inv({0})",
    "Pinv": "torch.linalg.pinv({0})",
    "Det": "torch.linalg.det({0})",
    "Slogdet": "torch.linalg.slogdet({0})",
    "Poly": "torch.tensor(np.poly({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.poly({0}))",
    "Polyadd": "torch.tensor(np.polyadd({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.polyadd({0}))",
    "Polyder": "torch.tensor(np.polyder({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.polyder({0}))",
    "Polydiv": "torch.tensor(np.polydiv({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.polydiv({0}))",
    "Polyfit": "torch.tensor(np.polyfit({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.polyfit({0}))",
    "Polyint": "torch.tensor(np.polyint({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.polyint({0}))",
    "Polymul": "torch.tensor(np.polymul({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.polymul({0}))",
    "Polysub": "torch.tensor(np.polysub({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.polysub({0}))",
    "Polyval": "torch.tensor(np.polyval({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.polyval({0}))",
    "Roots": "torch.tensor(np.roots({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.roots({0}))",
    "BroadcastedIota": "torch.tensor(np.broadcasted_iota({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.broadcasted_iota({0}))",
    "Bincount": "torch.tensor(np.bincount({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.bincount({0}))",
    "Histogram": "torch.tensor(np.histogram({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.histogram({0}))",
    "Histogram2d": "torch.tensor(np.histogram2d({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.histogram2d({0}))",
    "HistogramBinEdges": "torch.tensor(np.histogram_bin_edges({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.histogram_bin_edges({0}))",
    "Histogramdd": "torch.tensor(np.histogramdd({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.histogramdd({0}))",
    "Geomspace": "torch.tensor(np.geomspace({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.geomspace({0}))",
    "Gradient": "torch.tensor(np.gradient({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.gradient({0}))",
    "I0": "torch.tensor(np.i0({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.i0({0}))",
    "Mgrid": "torch.tensor(np.mgrid({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.mgrid({0}))",
    "Ogrid": "torch.tensor(np.ogrid({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.ogrid({0}))",
    "R_": "torch.tensor(np.r_({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.r_({0}))",
    "C_": "torch.tensor(np.c_({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.c_({0}))",
    "Fromfile": "torch.tensor(np.fromfile({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.fromfile({0}))",
    "Fromfunction": "torch.tensor(np.fromfunction({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.fromfunction({0}))",
    "Fromiter": "torch.tensor(np.fromiter({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.fromiter({0}))",
    "Frompyfunc": "torch.tensor(np.frompyfunc({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.frompyfunc({0}))",
    "Fromstring": "torch.tensor(np.fromstring({0}.cpu().numpy())).to({0}.device) if hasattr({0}, 'cpu') else torch.tensor(np.fromstring({0}))",
    "Eigh": "torch.linalg.eigh({0})",
    "Eig": "torch.linalg.eig({0})",
    "Eigvalsh": "torch.linalg.eigvalsh({0})",
    "Cond": "torch.linalg.cond({0}, p={p})",
    "Lstsq": "torch.linalg.lstsq({0}, {1}, rcond={rcond}).solution",
    "MatrixNorm": "torch.linalg.matrix_norm({0}, keepdim={keepdims})",
    "VectorNorm": "torch.linalg.vector_norm({0}, dim={axis}, keepdim={keepdims}, ord={ord})",
    "MatrixRank": "torch.linalg.matrix_rank({0}, tol={tol}, hermitian={hermitian})",
    "MatrixTranspose": "{0}.mT",
    "MultiDot": "torch.linalg.multi_dot({0})",
    "Diagonal": "torch.diagonal({0}, offset={offset}, dim1={axis1}, dim2={axis2})",
    "MatrixPower": "torch.linalg.matrix_power({0}, {n})",
    "Solve": "torch.linalg.solve({0}, {1})",
    "TriInv": "torch.linalg.inv({0})",
    "TriangularSolve": "torch.linalg.solve_triangular({0}.mH if {adjoint} else {0}, {1}, upper=not {lower} if not {adjoint} else {lower}, unitriangular={unit_diagonal})",
    "Lu": "torch.linalg.lu({0})",
    "LuFactor": "torch.linalg.lu_factor({0})",
    "LuSolve": "torch.linalg.lu_solve({0}, {1}, {2})",
    "Norm": "torch.linalg.norm({0}, ord={ord}, dim={axis}, keepdim={keepdims})",
    "MatrixExponential": "torch.linalg.matrix_exp({0})",
    "Cross": "torch.cross({0}, {1}, dim={axis})",
    "Relu": "torch.nn.functional.relu({0})",
    "Relu6": "torch.nn.functional.relu6({0})",
    "AdaptiveLogSoftmaxWithLoss": "torch.nn.functional.adaptive_log_softmax_with_loss({0}, {1}, cutoffs={cutoffs}, add_cluster_prob={add_cluster_prob})",
    "LeakyRelu": "torch.nn.functional.leaky_relu({0}, negative_slope={alpha})",
    "Elu": "torch.nn.functional.elu({0})",
    "Selu": "torch.nn.functional.selu({0})",
    "Gelu": "torch.nn.functional.gelu({0}, approximate='tanh' if {approximate} else 'none')",
    "Sigmoid": "torch.sigmoid({0})",
    "Softmax": "torch.nn.functional.softmax({0}, dim={axis})",
    "LogSoftmax": "torch.nn.functional.log_softmax({0}, dim={axis})",
    "OneHot": "torch.nn.functional.one_hot({0}, num_classes={depth})",
    "Clip": "torch.clamp({0}, min={a_min}, max={a_max})",
    "Softplus": "torch.nn.functional.softplus({0})",
    "Softsign": "torch.nn.functional.softsign({0})",
    "Conv1D": "torch.nn.functional.conv1d({0}, {1}, stride={stride}, padding='{padding}'.lower())",
    "Conv2D": "torch.nn.functional.conv2d({0}, {1}, stride={strides}, padding='{padding}'.lower())",
    "Conv3D": "torch.nn.functional.conv3d({0}, {1}, stride={strides}, padding='{padding}'.lower())",
    "MaxPool1D": "torch.nn.functional.max_pool1d({0}, kernel_size={ksize}, stride={strides}, padding='{padding}'.lower())",
    "MaxPool2D": "torch.nn.functional.max_pool2d({0}, kernel_size={ksize}, stride={strides}, padding='{padding}'.lower())",
    "MaxPool3D": "torch.nn.functional.max_pool3d({0}, kernel_size={ksize}, stride={strides}, padding='{padding}'.lower())",
    "AvgPool1D": "torch.nn.functional.avg_pool1d({0}, kernel_size={ksize}, stride={strides}, padding='{padding}'.lower())",
    "AvgPool2D": "torch.nn.functional.avg_pool2d({0}, kernel_size={ksize}, stride={strides}, padding='{padding}'.lower())",
    "AvgPool3D": "torch.nn.functional.avg_pool3d({0}, kernel_size={ksize}, stride={strides}, padding='{padding}'.lower())",
}


class PyTorchNNMixin:
    """NN Mixin."""

    def _get_nn_ops(self, kwargs: dict) -> dict[str, str]:
        return {}
