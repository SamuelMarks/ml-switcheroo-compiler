# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for generator.py."""

import os

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph

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

    def visit(self, node, input_vars: list[str], **kwargs) -> str:
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

    def visit(self, node, input_vars: list[str], **kwargs) -> str:
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

    _base_class_name: str = "nn.Module"

    def __init__(self, graph: IRGraph) -> None:
        """Initialize the PyTorch code generator with the given computation graph.

        Args:
            graph (object): The computation graph to compile.
        """
        super().__init__(graph)
        self.vision_visitor = PyTorchVisionVisitor()
        self.audio_visitor = PyTorchAudioVisitor()
        self.visitors.extend([*get_shared_ast_visitors(generator=self), PyTorchScatterVisitor()])

    def visit(self, node, input_vars: list[str], **kwargs) -> str:
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

    def visit_PowerIteration(self, node, input_vars: list[str], **kwargs) -> str:
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

    def visit_RaggedDot(self, node, input_vars: list[str], **kwargs) -> str:
        """Generate PyTorch code for a ragged dot product operation.

        Args:
            node (object): The IR node representing the ragged dot.
            input_vars (list[str]): The names of the input matrices.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: The generated PyTorch code for the ragged dot operation.
        """
        return f"pt_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node, input_vars: list[str], **kwargs) -> str:
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

    def generate(self) -> str:
        """Generate code using strict AST construction (CST) from a base NumPy string."""
        from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source
        from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator

        gen = NumpyGenerator(self.graph)
        base_code = gen.generate()
        return transpile_source(base_code, target_framework="pytorch")

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

    def _get_math_ops(self, kwargs) -> dict[str, str]:
        """Get math ops."""
        return {k: v.ast_template for k, v in __import__("ml_switcheroo_compiler.backends.mapping_loader", fromlist=["load_backend_mappings"]).load_backend_mappings("pytorch").operations.items() if v.ast_template}

    def _get_creation_ops(self, kwargs) -> dict[str, str]:
        """Get creation ops."""
        return {k: v.ast_template for k, v in __import__("ml_switcheroo_compiler.backends.mapping_loader", fromlist=["load_backend_mappings"]).load_backend_mappings("pytorch").operations.items() if v.ast_template}

    def _get_array_ops(self, kwargs) -> dict[str, str]:
        """Get array ops."""
        return {k: v.ast_template for k, v in __import__("ml_switcheroo_compiler.backends.mapping_loader", fromlist=["load_backend_mappings"]).load_backend_mappings("pytorch").operations.items() if v.ast_template}

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
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII"):
        """Load.

        Args:
        filepath (str): The filepath parameter.
        allow_pickle (bool): The allow_pickle parameter.
        fix_imports (bool): The fix_imports parameter.
        encoding (str): The encoding parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        import torch

        return torch.load(filepath, weights_only=not allow_pickle)

    @classmethod
    def save(cls: type, file: str, arr, allow_pickle: bool = True, fix_imports: bool = True) -> None:
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
    def savez(cls: type, file: str, *args, **kwds) -> None:
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
    def savez_compressed(cls: type, file: str, *args, **kwds) -> None:
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
