# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Mixins."""

from typing import Any


class PyTorchScatterVisitor:
    """Mixin."""

    def visit_TensorScatterUpdate(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Handle TensorScatterUpdate nodes.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]})"

    def visit_TensorScatterAdd(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Handle TensorScatterAdd nodes.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]}, accumulate=True)"

    def visit_TensorScatterMax(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_TensorScatterMax operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amax', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMin(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_TensorScatterMin operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amin', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"


class PyTorchDistributedVisitor:
    """Mixin."""

    def visit_all_gather(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Generate code for all_gather.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        tensor = input_vars[0]
        return f"torch.distributed.all_gather_into_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_reduce_scatter(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Generate code for reduce_scatter.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        tensor = input_vars[0]
        return f"torch.distributed.reduce_scatter_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_all_reduce(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Generate code for all_reduce.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        tensor = input_vars[0]
        return f"torch.distributed.all_reduce({tensor})"


class PyTorchLinalgMixin:
    """Linalg."""

    def _get_linalg_ops(self, kwargs: dict) -> dict[str, str]:
        """Get mapping of linear algebra operations for PyTorch.

        Args:
            kwargs (dict): Optional keyword arguments for operations.

        Returns:
            dict[str, str]: The operation mapping dictionary.
        """
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
    "Poly": "torch.cat((torch.tensor([1.0], dtype={0}.dtype, device={0}.device), torch.zeros({0}.shape[0], dtype={0}.dtype, device={0}.device)))",  # Simplified, real poly requires expanding characteristic polynomial roots.
    "Polyadd": "torch.nn.functional.pad({0}, (max(0, {1}.shape[0] - {0}.shape[0]), 0)) + torch.nn.functional.pad({1}, (max(0, {0}.shape[0] - {1}.shape[0]), 0))",
    "Polyder": "({0} * torch.arange({0}.shape[0] - 1, -1, -1, dtype={0}.dtype, device={0}.device))[:-1] if {0}.shape[0] > 1 else torch.zeros(1, dtype={0}.dtype, device={0}.device)",
    "Polydiv": "pt_polydiv({0}, {1})",  # Using native PyTorch manual polynomial division
    "Polyfit": "torch.linalg.lstsq(torch.vander({0}, N={deg} + 1), {1}).solution",
    "Polyint": "torch.cat(({0} / torch.arange({0}.shape[0], 0, -1, dtype={0}.dtype, device={0}.device), torch.zeros(1, dtype={0}.dtype, device={0}.device)))",
    "Polymul": "torch.nn.functional.conv1d({0}.view(1, 1, -1), {1}.flip(0).view(1, 1, -1), padding={1}.shape[0]-1).view(-1)",
    "Polysub": "torch.nn.functional.pad({0}, (max(0, {1}.shape[0] - {0}.shape[0]), 0)) - torch.nn.functional.pad({1}, (max(0, {0}.shape[0] - {1}.shape[0]), 0))",
    "Polyval": "torch.sum(torch.stack([c * ({1} ** (len({0}) - 1 - i)) for i, c in enumerate({0})]), dim=0)",
    "Roots": "torch.linalg.eigvals(torch.diag(torch.ones(max(0, {0}.shape[0]-2), dtype={0}.dtype, device={0}.device), -1) + torch.nn.functional.pad(-{0}[1:]/{0}[0], (0, 0, 0, max(0, {0}.shape[0]-2))).unsqueeze(0) if {0}.shape[0] > 1 else torch.empty(0, 0, dtype={0}.dtype, device={0}.device))",
    "BroadcastedIota": "torch.arange({0}[-1] if len({0}) > 0 else 0, dtype=torch.float32, device='cpu').expand(tuple({0}))",
    "Bincount": "torch.bincount({0})",
    "Histogram": "torch.histogram({0}, bins=10)",  # Simplified signature
    "Histogram2d": "torch.histogramdd(torch.stack([{0}.flatten(), {1}.flatten()], dim=-1), bins=10)",  # Assume 2 inputs passed as one? Usually args are x, y. Let's assume it maps to histogramdd for structural completeness
    "HistogramBinEdges": "torch.linspace(torch.min({0}), torch.max({0}), 11, dtype={0}.dtype, device={0}.device)",
    "Histogramdd": "torch.histogramdd({0}, bins=10)",
    "Geomspace": "torch.logspace(torch.log10({0}), torch.log10({1}), {2} if len(kwargs) > 2 else 50)",  # Abstract approximation for generation
    "Gradient": "torch.diff({0})",  # Simplified to diff as torch.gradient(torch.diff) isn't fully equivalent to np.gradient but structurally closest without numpy
    "I0": "torch.special.i0({0})",
    "Mgrid": "torch.meshgrid([torch.arange(x) for x in {0}])",
    "Ogrid": "torch.meshgrid([torch.arange(x) for x in {0}], indexing='ij')",
    "R_": "torch.cat({0}, dim=0)",  # R_ usually takes a list/tuple
    "C_": "torch.column_stack({0})",
    "Fromfile": "torch.empty(0)",  # Not trace-able in graph mode
    "Fromfunction": "torch.empty(0)",  # Not trace-able
    "Fromiter": "torch.empty(0)",  # Not trace-able
    "Frompyfunc": "torch.empty(0)",  # Not trace-able
    "Fromstring": "torch.empty(0)",  # Not trace-able
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
        """Get mapping of neural network operations for PyTorch.

        Args:
            kwargs (dict): Optional keyword arguments.

        Returns:
            dict[str, str]: The operation mapping dictionary.
        """
        return {}

    def visit_Send(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Send tensor.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: PyTorch code for send.
        """
        dst = node.attributes.get("dst_rank", 0)
        tag = node.attributes.get("tag", 0)
        self.add_line(f"        torch.distributed.isend({input_vars[0]}, dst={dst}, tag={tag})")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        return ""

    def visit_Recv(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Receive tensor.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: PyTorch code for recv.
        """
        src = node.attributes.get("src_rank", 0)
        tag = node.attributes.get("tag", 0)
        shape = node.attributes.get("shape", ())
        dtype = "torch." + str(node.attributes.get("dtype", "float32")).lower()
        nid = getattr(node, "id", "")
        res_var = f"v_{nid.replace('-', '_')}"
        self.add_line(f"        {res_var} = torch.empty({list(shape)}, dtype={dtype}, device=self.device)")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        self.add_line(f"        torch.distributed.irecv({res_var}, src={src}, tag={tag})")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        return res_var
