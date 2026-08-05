# ruff: noqa: E501
"""Core abstractions and logic definitions for generator.py."""

import os

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend

from .pytorch_mixins import PyTorchDistributedVisitor, PyTorchLinalgMixin, PyTorchNNMixin, PyTorchScatterVisitor


class PyTorchVisionVisitor:
    """Handle vision ops for PyTorch."""

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
        "ElasticTransform": lambda vars: f"torchvision.transforms.functional.elastic_transform({vars[0]}, {vars[1]})",
        "PerspectiveTransform": lambda vars: f"torchvision.transforms.functional.perspective({vars[0]}, {vars[1]}, {vars[2]})",
        "ExtractBoundingBoxes": lambda vars: f"torchvision.ops.roi_align({vars[0]}, {vars[1]}, 1.0)",
        "IoU": lambda vars: f"torchvision.ops.box_iou({vars[0]}, {vars[1]})",
        "NonMaxSuppression": lambda vars: f"torchvision.ops.nms({vars[0]}, {vars[1]}, 0.5)",
        "ResizeBicubic": lambda vars: f"torch.nn.functional.interpolate({vars[0]}, mode='bicubic')",
        "ResizeLanczos3": lambda vars: f"torch.nn.functional.interpolate({vars[0]}, mode='linear')",
        "GaussianBlur": lambda vars: f"torchvision.transforms.functional.gaussian_blur({vars[0]})",
        "MedianFilter": lambda vars: f"torchaudio.functional.median_filter({vars[0]})",
    }

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Process a vision operation node and produce corresponding PyTorch code.

        Args:
            node (object): The IR node representing the vision operation to process.
            input_vars (list[str]): The names of the variables used as inputs for the operation.
            **kwargs (object): Additional keyword arguments representing operation attributes.

        Returns:
            str: The generated PyTorch code for the given node.
        """
        op_type = getattr(node, "op_type", "")
        handler = self._handlers.get(op_type)
        return handler(input_vars) if handler else ""


class PyTorchAudioVisitor:
    """Handle audio ops for PyTorch."""

    handled_ops = {"Istft", "MelFilterbank", "Mfcc"}
    _handlers = {
        "Istft": lambda vars: f"torch.istft({vars[0]})",
        "MelFilterbank": lambda vars: f"torchaudio.functional.melscale_fbanks({vars[0]})",
        "Mfcc": lambda vars: f"torchaudio.transforms.MFCC()({vars[0]})",
    }

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Process an audio operation node and produce corresponding PyTorch code.

        Args:
            node (object): The IR node representing the audio operation to process.
            input_vars (list[str]): The names of the variables used as inputs for the operation.
            **kwargs (object): Additional keyword arguments representing operation attributes.

        Returns:
            str: The generated PyTorch code for the given node.
        """
        op_type = getattr(node, "op_type", "")
        handler = self._handlers.get(op_type)
        return handler(input_vars) if handler else ""


@register_backend("pytorch")
class PyTorchCodeGenerator(PyTorchLinalgMixin, PyTorchNNMixin, ClassBasedGenerator):
    """PyTorch code generator."""

    def __init__(self, graph: object) -> None:
        """Initialize the PyTorch code generator with the given computation graph.

        Args:
            graph (object): The computation graph to compile.
        """
        super().__init__(graph)
        self.vision_visitor = PyTorchVisionVisitor()
        self.audio_visitor = PyTorchAudioVisitor()
        self.visitors.extend([*get_shared_ast_visitors(generator=self), PyTorchScatterVisitor(), PyTorchDistributedVisitor()])

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Process an IR node and produce the corresponding PyTorch code string.

        Args:
            node (object): The IR node representing the operation.
            input_vars (list[str]): The names of the input variables for the operation.
            **kwargs (object): Additional attributes and parameters for the operation.

        Returns:
            str: The generated PyTorch code string.
        """
        op_type = getattr(node, "op_type", "")
        if op_type in self.vision_visitor.handled_ops:
            return self.vision_visitor.visit(node, input_vars, **kwargs)
        if op_type in self.audio_visitor.handled_ops:
            return self.audio_visitor.visit(node, input_vars, **kwargs)
        return super().visit(node, input_vars, **kwargs)

    def _get_backend_prefix(self) -> str:
        """Retrieve the short prefix used for backend-specific variables and namespaces.

        Returns:
            str: The backend prefix string ("pt" for PyTorch).
        """
        return "pt"

    def visit_PowerIteration(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for a power iteration operation.

        Args:
            node (object): The IR node representing the power iteration.
            input_vars (list[str]): The names of the input variables.
            **kwargs (object): Additional arguments such as iteration counts.

        Returns:
            str: The generated PyTorch code for power iteration.
        """
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"pt_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_RaggedDot(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for a ragged dot product operation.

        Args:
            node (object): The IR node representing the ragged dot.
            input_vars (list[str]): The names of the input matrices.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: The generated PyTorch code for the ragged dot operation.
        """
        return f"pt_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for an Einstein summation operation.

        Args:
            node (object): The IR node representing the einsum operation.
            input_vars (list[str]): The input variable names.
            **kwargs (object): Additional parameters, including the equation string.

        Returns:
            str: The generated PyTorch code using torch.einsum.
        """
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"torch.einsum('{eq}', {args_str})"

    def get_fallback_prefix(self) -> str:
        """Retrieve the fallback library prefix used for missing generic operations.

        Returns:
            str: The fallback prefix string ("torch").
        """
        return "torch"

    def get_fallback_axis_kwarg(self) -> str:
        """Retrieve the keyword argument name used for specifying dimensions.

        Returns:
            str: The axis keyword argument name ("dim").
        """
        return "dim"

    def get_fallback_keepdims_kwarg(self) -> str:
        """Retrieve the keyword argument name used for keeping dimensions.

        Returns:
            str: The keepdims keyword argument name ("keepdim").
        """
        return "keepdim"

    def _get_math_ops(self, kwargs: dict) -> dict[str, str]:
        """Provide a dictionary mapping math operations to their PyTorch code templates.

        Args:
            kwargs (dict): Parameters dictionary to evaluate operation templates.

        Returns:
            dict[str, str]: The dictionary of math operation templates.
        """
        return {
            "TruncateDiv": "torch.trunc({0} / {1})",
            "TruncateMod": "torch.fmod({0}, {1})",
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

    def _get_creation_ops(self, kwargs: dict) -> dict[str, str]:
        """Provide a dictionary mapping tensor creation operations to PyTorch templates.

        Args:
            kwargs (dict): Parameters dictionary used for evaluating creation logic.

        Returns:
            dict[str, str]: The dictionary of creation operation templates.
        """
        return {
            "Arange": "torch.arange({0})",
            "Zeros": "torch.zeros({shape})" + (", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)" if "dtype" in kwargs else ""),
            "Ones": "torch.ones({shape})" + (", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)" if "dtype" in kwargs else ""),
            "Full": "torch.full({shape}, {fill_value})" + (", dtype=getattr(torch, '" + str(kwargs.get("dtype")) + "', torch.float32)" if "dtype" in kwargs else ""),
        }

    def _get_array_ops(self, kwargs: dict) -> dict[str, str]:
        """Provide a dictionary mapping array operations to PyTorch code templates.

        Args:
            kwargs (dict): The parameters dictionary required for shaping the templates.

        Returns:
            dict[str, str]: The dictionary mapping operations to PyTorch strings.
        """
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
        """Retrieve the complete dictionary mapping all supported operations to PyTorch code templates.

        Args:
            kwargs (dict): The keyword arguments dict for resolving dynamic parameters.

        Returns:
            dict[str, str]: A dictionary mapping operation type names to format strings.
        """
        ops = super().get_ops_map(kwargs)
        ops["Beta"] = "torch.distributions.beta.Beta({1}, {2}).sample({shape})"
        ops["Dirichlet"] = "torch.distributions.dirichlet.Dirichlet({1}).sample({shape})"
        ops["Gamma"] = "torch.distributions.gamma.Gamma({1}, 1.0).sample({shape})"
        ops["RngBitGenerator"] = "torch.randint(0, 255, {shape})"
        ops["RngUniform"] = "({1} - {0}) * torch.rand({shape}) + {0}"
        ops["Infeed"] = "{0}"
        ops["Outfeed"] = "{0}"
        ops["AxisIndex"] = "0"
        ops["AllToAll"] = "{0}"
        ops["Pmax"] = "{0}"
        ops["Pmin"] = "{0}"
        ops["PsumScatter"] = "{0}"
        ops["Pswapaxes"] = "{0}"
        ops["Ppermute"] = "{0}"
        ops["Pshuffle"] = "{0}"
        ops["CreateToken"] = "0"
        ops["WithShardingConstraint"] = "{0}"
        return ops

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Emit the code to assign a constant parameter to a local variable.

        Args:
            var_name (str): The var_name parameter.
            val_repr (str): The val_repr parameter.
        """
        self.add_line(f"{var_name} = self.{var_name}")

    def _get_prefix_code(self) -> list[str]:
        """Generate the prefix code required for the PyTorch module, including imports and utilities.

        Returns:
            list[str]: A list of string code lines to be emitted before the main class.
        """
        tmpl_path = os.path.join(os.path.dirname(__file__), "pytorch_prefix.py.tmpl")
        with open(tmpl_path) as f:
            pt_prefix_template = f.read()
        return ["import torch", "import torch.nn as nn", *pt_prefix_template.splitlines()]

    def _emit_init_body(self) -> bool:
        """Emit the initialization code for the PyTorch module's parameters.

        Returns:
            bool: True if any parameters were registered in the module, False otherwise.
        """
        has_params = False
        for node in self.sorted_nodes:
            if node.op_type == "Constant":
                val_repr = self.emit_constant(node)
                var_name = self.assign_var_name(node.id, "const")
                self.add_line(f"self.register_parameter('{var_name}', nn.Parameter(torch.tensor({val_repr})))")
                has_params = True
        return has_params

    @classmethod
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> object:
        """Load.

        Args:
        filepath (str): The filepath parameter.
        allow_pickle (bool): The allow_pickle parameter.
        fix_imports (bool): The fix_imports parameter.
        encoding (str): The encoding parameter.

        Returns:
        object: Result.
        """
        import torch

        return torch.load(filepath, weights_only=not allow_pickle)

    @classmethod
    def save(cls: type, file: str, arr: object, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): The file parameter.
            arr (object): The arr parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
        """
        import torch

        torch.save(arr, file)

    @classmethod
    def savez(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Savez.

        Args:
            file (str): The file parameter.
            *args (object): Positional args.
            **kwds (object): Keyword args.
        """
        import torch

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        torch.save(data, file)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Savez compressed.

        Args:
            file (str): The file parameter.
            *args (object): Positional args.
            **kwds (object): Keyword args.
        """
        import torch

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        torch.save(data, file)
