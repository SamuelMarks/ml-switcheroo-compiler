# ruff: noqa: E402

"""Module docstring."""

from ml_switcheroo_compiler.backends.common.generator_mixins import GroupNormConfig

"""PyTorch Target Emission."""
from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorMixin
from .pytorch_mixins import PyTorchScatterMixin, PyTorchDistributedMixin
from ml_switcheroo_compiler.backends.registry import register_backend


class PyTorchVisionVisitor:
    """Handles vision ops for PyTorch."""

    handled_ops = {
        "ElasticTransform",
        "PerspectiveTransform",
        "ExtractBoundingBoxes",
        "IoU",
        "NonMaxSuppression",
        "ResizeBicubic",
        "ResizeLanczos3",
        "GaussianBlur",
        "MedianFilter",
    }
    _handlers = {
        "ElasticTransform": lambda vars: (
            f"torchvision.transforms.functional.elastic_transform({vars[0]}, {vars[1]})"
        ),
        "PerspectiveTransform": lambda vars: (
            f"torchvision.transforms.functional.perspective({vars[0]}, {vars[1]}, {vars[2]})"
        ),
        "ExtractBoundingBoxes": lambda vars: (
            f"torchvision.ops.roi_align({vars[0]}, {vars[1]}, 1.0)"
        ),
        "IoU": lambda vars: f"torchvision.ops.box_iou({vars[0]}, {vars[1]})",
        "NonMaxSuppression": lambda vars: f"torchvision.ops.nms({vars[0]}, {vars[1]}, 0.5)",
        "ResizeBicubic": lambda vars: f"torch.nn.functional.interpolate({vars[0]}, mode='bicubic')",
        "ResizeLanczos3": lambda vars: f"torch.nn.functional.interpolate({vars[0]}, mode='linear')",
        "GaussianBlur": lambda vars: f"torchvision.transforms.functional.gaussian_blur({vars[0]})",
        "MedianFilter": lambda vars: f"torchaudio.functional.median_filter({vars[0]})",
    }

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit."""
        op_type = getattr(node, "op_type", "")  # pragma: no cover
        handler = self._handlers.get(op_type)  # pragma: no cover
        return handler(input_vars) if handler else ""  # pragma: no cover


class PyTorchAudioVisitor:
    """Handles audio ops for PyTorch."""

    handled_ops = {"Istft", "MelFilterbank", "Mfcc"}
    _handlers = {
        "Istft": lambda vars: f"torch.istft({vars[0]})",
        "MelFilterbank": lambda vars: f"torchaudio.functional.melscale_fbanks({vars[0]})",
        "Mfcc": lambda vars: f"torchaudio.transforms.MFCC()({vars[0]})",
    }

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit."""
        op_type = getattr(node, "op_type", "")  # pragma: no cover
        handler = self._handlers.get(op_type)  # pragma: no cover
        return handler(input_vars) if handler else ""  # pragma: no cover


@register_backend("pytorch")
class PyTorchCodeGenerator(
    SharedASTGeneratorMixin, ClassBasedGenerator, PyTorchScatterMixin, PyTorchDistributedMixin
):
    """PyTorch code generator."""

    def __init__(self, graph: object) -> None:
        """Init."""
        super().__init__(graph)
        self.vision_visitor = PyTorchVisionVisitor()
        self.audio_visitor = PyTorchAudioVisitor()

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit."""
        op_type = getattr(node, "op_type", "")
        if op_type in self.vision_visitor.handled_ops:
            return self.vision_visitor.visit(node, input_vars, **kwargs)  # pragma: no cover
        if op_type in self.audio_visitor.handled_ops:
            return self.audio_visitor.visit(node, input_vars, **kwargs)  # pragma: no cover
        return super().visit(node, input_vars, **kwargs)

    def _get_backend_prefix(self) -> str:
        """Function docstring."""
        return "pt"  # pragma: no cover

    """Emit PyTorch-compatible code from IR."""

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate power iteration."""
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"pt_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle Einsum nodes."""
        args_str = ", ".join(input_vars)  # pragma: no cover
        eq = kwargs.get("equation", "")  # pragma: no cover
        return f"torch.einsum('{eq}', {args_str})"  # pragma: no cover

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "torch"

    def get_fallback_axis_kwarg(self) -> str:
        """Get the fallback axis keyword argument name."""
        return "dim"

    def get_fallback_keepdims_kwarg(self) -> str:
        """Get the fallback keepdims keyword argument name."""
        return "keepdim"

    def _get_math_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "TrueDivide": "torch.true_divide({0}, {1})",
            "Sum": "torch.sum({0}, dim={axis}, keepdim={keepdims})",
            "Mean": "torch.mean({0}, dim={axis}, keepdim={keepdims})",
            "Max": "torch.max({0}, dim={axis}, keepdim={keepdims})",
            "Min": "torch.min({0}, dim={axis}, keepdim={keepdims})",
            "Prod": "torch.prod({0}, dim={axis}, keepdim={keepdims})",
            "All": "torch.all({0}, dim={axis}, keepdim={keepdims})",
            "AnyOp": "torch.any({0}, dim={axis}, keepdim={keepdims})",
            "Erfinv": "torch.erfinv({0})",
            "NanToNum": "torch.nan_to_num({0}, nan={nan}, posinf={posinf}, neginf={neginf})",
        }

    def _get_linalg_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "Matmul": "torch.matmul({0}, {1})",
            "Dot": "torch.dot({0}, {1})",
            "Fft": "torch.fft.fft({0})",
            "Rfft": "torch.fft.rfft({0})",
            "Fftn": "torch.fft.fftn({0})",
            "Cholesky": "torch.linalg.cholesky({0})",
            "Svd": "torch.linalg.svd({0})",
            "Qr": "torch.linalg.qr({0})",
            "Inv": "torch.linalg.inv({0})",
            "Pinv": "torch.linalg.pinv({0})",
            "Det": "torch.linalg.det({0})",
            "Slogdet": "torch.linalg.slogdet({0})",
            "Eigh": "torch.linalg.eigh({0})",
            "Eigvalsh": "torch.linalg.eigvalsh({0})",
            "MatrixPower": "torch.linalg.matrix_power({0}, {n})",
            "Solve": "torch.linalg.solve({0}, {1})",
            "TriangularSolve": "torch.linalg.solve_triangular({0}, {1}, upper=not {lower}, unitriangular={unit_diagonal})",
            "Lu": "torch.linalg.lu({0})",
            "LuFactor": "torch.linalg.lu_factor({0})",
            "LuSolve": "torch.linalg.lu_solve({0}, {1}, {2})",
            "Norm": "torch.linalg.norm({0}, ord={ord}, dim={axis}, keepdim={keepdims})",
            "MatrixExponential": "torch.linalg.matrix_exp({0})",
            "Cross": "torch.cross({0}, {1}, dim={axis})",
        }

    def _get_nn_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "Relu": "torch.nn.functional.relu({0})",
            "Relu6": "torch.nn.functional.relu6({0})",
            "LeakyRelu": "torch.nn.functional.leaky_relu({0}, negative_slope={alpha})",
            "Elu": "torch.nn.functional.elu({0})",
            "Selu": "torch.nn.functional.selu({0})",
            "Gelu": "torch.nn.functional.gelu({0}, approximate='tanh' if {approximate} else 'none')",
            "Sigmoid": "torch.sigmoid({0})",
            "Softmax": "torch.nn.functional.softmax({0}, dim={axis})",
            "LogSoftmax": "torch.nn.functional.log_softmax({0}, dim={axis})",
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

    def _get_creation_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "Arange": "torch.arange({0})",
            "Zeros": "torch.zeros({shape})"
            + (
                ", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)"
                if "dtype" in kwargs
                else ""
            ),
            "Ones": "torch.ones({shape})"
            + (
                ", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)"
                if "dtype" in kwargs
                else ""
            ),
            "Full": "torch.full({shape}, {fill_value})"
            + (
                ", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)"
                if "dtype" in kwargs
                else ""
            ),
        }

    def _get_array_ops(self, kwargs: dict) -> dict[str, str]:
        return {
            "BroadcastTo": "{0}.expand({shape})",
            "Reshape": "torch.reshape({0}, {shape})",
            "Sort": "torch.sort({0}, dim={dimension})",
            "ArgSort": "torch.argsort({0}, dim={dimension})",
            "Allclose": "torch.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "AssignVariable": "{0}",
            "StopGradient": "{0}.detach()",
            "Resize": "torch.nn.functional.interpolate({0}, size={size}, mode={method}, antialias={antialias})",
            "AffineGrid": "torch.nn.functional.affine_grid({0}, {size}, align_corners={align_corners})",
            "GridSample": "torch.nn.functional.grid_sample({0}, {1}, mode={mode}, padding_mode={padding_mode}, align_corners={align_corners})",
            "DrawBoundingBoxes": "torchvision.utils.draw_bounding_boxes({0}, {1}, colors={colors}, labels={texts})",
            "RgbToYiq": "{0}",
            "YiqToRgb": "{0}",
            "RgbToYuv": "{0}",
            "YuvToRgb": "{0}",
            "Ifft": "torch.fft.ifft({0}, n={n}, dim={axis})",
            "Fft2d": "torch.fft.fft2({0}, s={s}, dim={axes})",
            "Ifft2d": "torch.fft.ifft2({0}, s={s}, dim={axes})",
            "Fft3d": "torch.fft.fftn({0}, s={s}, dim={axes})",
            "Ifft3d": "torch.fft.ifftn({0}, s={s}, dim={axes})",
            "Rfft2d": "torch.fft.rfft2({0}, s={s}, dim={axes})",
            "Rfft3d": "torch.fft.rfftn({0}, s={s}, dim={axes})",
            "Irfft": "torch.fft.irfft({0}, n={n}, dim={axis})",
            "Irfft2d": "torch.fft.irfft2({0}, s={s}, dim={axes})",
            "Irfft3d": "torch.fft.irfftn({0}, s={s}, dim={axes})",
            "Stft": "torch.stft({0}, n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}, window={window}, center={center}, normalized={normalized}, onesided={onesided}, return_complex={return_complex})",
            "Istft": "torch.istft({0}, n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}, window={window}, center={center}, normalized={normalized}, onesided={onesided}, length={length}, return_complex={return_complex})",
            "HannWindow": "torch.hann_window({window_length}, periodic={periodic})",
            "HammingWindow": "torch.hamming_window({window_length}, periodic={periodic}, alpha={alpha}, beta={beta})",
            "KaiserWindow": "torch.kaiser_window({window_length}, periodic={periodic}, beta={beta})",
            "ReadVariable": "{0}",
            "TensorScatterUpdate": "{0}.clone().index_put_(tuple({1}.unbind(-1)), {2})",
            "TensorScatterAdd": "{0}.clone().index_put_(tuple({1}.unbind(-1)), {2}, accumulate=True)",
            "Transpose": "torch.permute({0}, {axes})" if "axes" in kwargs else "{0}.t()",
            "Argmax": "torch.argmax({0}, dim={axis}, keepdim={keepdims})",
            "Argmin": "torch.argmin({0}, dim={axis}, keepdim={keepdims})",
            "Cast": "getattr({0}, str('{dtype}'))()",
            "Bitcast": "{0}.view(getattr(torch, str('{dtype}')))",
        }

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Execute get_ops_map.

        Args:
            kwargs (Any): Argument kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        ops = {}
        ops.update(self._get_math_ops(kwargs))
        ops.update(self._get_linalg_ops(kwargs))
        ops.update(self._get_nn_ops(kwargs))
        ops.update(self._get_creation_ops(kwargs))
        ops.update(self._get_array_ops(kwargs))
        return ops

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = self.{var_name}")

    def _get_prefix_code(self) -> list[str]:
        """Return prefix code."""
        import os

        tmpl_path = os.path.join(os.path.dirname(__file__), "pytorch_prefix.py.tmpl")
        with open(tmpl_path) as f:
            pt_prefix_template = f.read()
        return [
            "import torch",
            "import torch.nn as nn",
            *self._get_group_norm_code(
                GroupNormConfig(
                    "pt",
                    "torch",
                    "torch.reshape",
                    "torch.mean",
                    "torch.var",
                    "torch.sqrt",
                    "dim",
                    "keepdim",
                    ", unbiased=False",
                )
            ),
            *pt_prefix_template.split("\n"),
        ]

    def _emit_init_body(self) -> bool:
        """Emit initialization code. Return True if params were emitted, False otherwise."""
        has_params = False
        for node in self.sorted_nodes:
            if node.op_type == "Constant":
                val_repr = self.emit_constant(node)
                var_name = self.assign_var_name(node.id, "const")
                self.add_line(
                    f"self.register_parameter('{var_name}', nn.Parameter(torch.tensor({val_repr})))"
                )
                has_params = True
        return has_params
