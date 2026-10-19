"""Implementation of generator operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.pytorch.generator`)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import cast

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue
from ml_switcheroo_compiler.ir.core import IRGraph

from .pytorch_mixins import PyTorchDistributedVisitor, PyTorchLinalgMixin, PyTorchNNMixin, PyTorchScatterVisitor  # noqa: F401


@register_backend("pytorch")
class PyTorchCodeGenerator(PyTorchLinalgMixin, PyTorchNNMixin, ClassBasedGenerator):
    """PyTorch code generator."""

    @classmethod
    def get_memory_info(cls, device: EagerValue = None) -> dict[str, int]:
        """Retrieve current and peak memory telemetry from torch.cuda or stdlib resource.

        Args:
            device (EagerValue): Optional device identifier.

        Returns:
            dict[str, int]: Mapping with 'current' and 'peak' memory usage in bytes.
        """
        import resource

        import torch

        if torch.cuda.is_available() and (device is None or "cuda" in str(device)):
            dev = device if device is not None else "cuda"
            return {
                "current": int(getattr(torch.cuda, "memory_allocated")(dev)),
                "peak": int(getattr(torch.cuda, "max_memory_allocated")(dev)),
            }
        rss: int = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {"current": rss, "peak": rss}

    @classmethod
    def initialize_distributed(cls, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Initialize distributed context.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.
        """
        import torch.distributed

        backend_config = kwargs.get("backend_config")
        if not torch.distributed.is_initialized() and backend_config:
            torch.distributed.init_process_group(backend=getattr(backend_config, "backend"), init_method=getattr(backend_config, "init_method"), world_size=getattr(backend_config, "world_size"), rank=getattr(backend_config, "rank"))

    _base_class_name: str = "nn.Module"

    def __init__(self, graph: IRGraph) -> None:
        """Initialize the PyTorch code generator with the given computation graph.

        Args:
            graph (IRGraph): The computation graph to compile.
        """
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self), PyTorchScatterVisitor()])

    def visit(self, node: EagerValue, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Process an IR node and produce the corresponding PyTorch code string.

        Args:
            node (EagerValue): The IR node representing the operation.
            input_vars (list[str]): The names of the input variables for the operation.
            **kwargs (EagerValue): Additional attributes and parameters for the operation.

        Returns:
            str: The generated PyTorch code string.
        """
        return super().visit(cast("LogicalNode", node), input_vars, **kwargs)

    def visit_PowerIteration(self, node: LogicalNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Generate PyTorch code for a power iteration operation.

        Args:
            node (LogicalNode): The IR node representing the power iteration.
            input_vars (list[str]): The names of the input variables.
            **kwargs (EagerValue): Additional arguments such as iteration counts.

        Returns:
            str: The generated PyTorch code for power iteration.
        """
        num_iters = node.attributes.get("num_iters", 1)
        u_var = input_vars[1] if len(input_vars) > 1 else "None"
        return f"pt_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_RaggedDot(self, node: LogicalNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Generate PyTorch code for a ragged dot product operation.

        Args:
            node (LogicalNode): The IR node representing the ragged dot.
            input_vars (list[str]): The names of the input matrices.
            **kwargs (EagerValue): Additional operation attributes.

        Returns:
            str: The generated PyTorch code for the ragged dot operation.
        """
        return f"pt_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: LogicalNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Generate PyTorch code for an Einstein summation operation.

        Args:
            node (LogicalNode): The IR node representing the einsum operation.
            input_vars (list[str]): The input variable names.
            **kwargs (EagerValue): Additional parameters, including the equation string.

        Returns:
            str: The generated PyTorch code using torch.einsum.
        """
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"torch.einsum('{eq}', {args_str})"

    def generate(self) -> str:
        """Generate PyTorch code directly from the IR graph as an nn.Module class.

        Returns:
            str: The generated PyTorch nn.Module source code.
        """
        return super().generate()

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

    def _get_math_ops(self, kwargs: EagerValue) -> dict[str, str]:
        """Get PyTorch mathematical operation templates.

        Args:
            kwargs (EagerValue): Keyword arguments dictionary.

        Returns:
            dict[str, str]: Mapping from operation name to AST template string.
        """
        del kwargs
        from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

        return {k: v.ast_template for k, v in load_backend_mappings("pytorch").operations.items() if v.ast_template}

    def _get_creation_ops(self, kwargs: EagerValue) -> dict[str, str]:
        """Get PyTorch tensor creation operation templates.

        Args:
            kwargs (EagerValue): Keyword arguments dictionary.

        Returns:
            dict[str, str]: Mapping from operation name to AST template string.
        """
        return self._get_math_ops(kwargs)

    def _get_array_ops(self, kwargs: EagerValue) -> dict[str, str]:
        """Get PyTorch array manipulation operation templates.

        Args:
            kwargs (EagerValue): Keyword arguments dictionary.

        Returns:
            dict[str, str]: Mapping from operation name to AST template string.
        """
        return self._get_math_ops(kwargs)

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
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> EagerValue:
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

        return cast(EagerValue, torch.load(filepath, weights_only=not allow_pickle))

    @classmethod
    def save(cls: type, file: str, arr: EagerValue, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): The file parameter.
            arr (EagerValue): The arr parameter.
            allow_pickle (bool): The allow_pickle parameter.
            fix_imports (bool): The fix_imports parameter.
        """
        import torch

        torch.save(arr, file)

    @classmethod
    def savez(cls: type, file: str, *args: EagerValue, **kwds: EagerValue) -> None:
        """Savez.

        Args:
            file (str): The file parameter.
            *args (EagerValue): Positional args.
            **kwds (EagerValue): Keyword args.
        """
        import torch

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        torch.save(data, file)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: EagerValue, **kwds: EagerValue) -> None:
        """Savez compressed.

        Args:
            file (str): The file parameter.
            *args (EagerValue): Positional args.
            **kwds (EagerValue): Keyword args.
        """
        import torch

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        torch.save(data, file)

    def _create_pytorch_graph_evaluator(self, graph: IRGraph) -> EagerValue:
        """Create an interpreter-backed callable evaluating the IR graph for PyTorch.

        Args:
            graph (IRGraph): Target computational graph.

        Returns:
            EagerValue: Graph evaluation callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        class GraphModule:
            """Callable module wrapper evaluating the IR graph."""

            def __call__(self, *fn_args: EagerValue) -> EagerValue:
                """Forward execution call.

                Args:
                    *fn_args (EagerValue): Input tensors.

                Returns:
                    EagerValue: Computed outputs.
                """
                input_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"]
                inputs: dict[str, EagerValue] = {}
                for i, inp_node in enumerate(input_nodes):
                    if i < len(fn_args):
                        arg_val = fn_args[i]
                        inputs[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
                evaluated = evaluate_graph(graph, inputs=cast("dict[str, EvalValue]", inputs))
                if hasattr(graph, "outputs") and graph.outputs:
                    if len(graph.outputs) == 1:
                        return cast(EagerValue, evaluated.get(graph.outputs[0]))
                    return tuple(evaluated.get(out_id) for out_id in graph.outputs)
                return cast(EagerValue, evaluated)

        return GraphModule()

    def _wrap_pytorch_runner(self, compiled_fn: EagerValue, torch_mod: EagerValue) -> EagerValue:
        """Wrap a compiled or traced PyTorch callable to convert inputs via torch.as_tensor.

        Args:
            compiled_fn (EagerValue): Compiled or traced PyTorch function.
            torch_mod (EagerValue): Loaded torch module.

        Returns:
            EagerValue: Wrapped execution callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor

        def runner(*w_args: EagerValue, **w_kwargs: EagerValue) -> EagerValue:
            """Execute compiled PyTorch callable.

            Args:
                *w_args (EagerValue): Input tensors.
                **w_kwargs (EagerValue): Keyword arguments.

            Returns:
                EagerValue: Output tensor data.
            """
            del w_kwargs
            pt_args = [getattr(torch_mod, "as_tensor")(a.data if isinstance(a, Tensor) else a) for a in w_args]
            return cast(Callable[..., EagerValue], compiled_fn)(*pt_args)

        return runner

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> EagerValue:
        """Compile IRGraph into an optimized executable callable using TorchScript or torch.compile.

        Args:
            graph (IRGraph): Target computational graph to compile.
            **kwargs (EagerValue): Optional compiler options ('mode', 'backend', 'sample_inputs').

        Returns:
            EagerValue: Optimized executable callable wrapping the compiled PyTorch module.
        """
        mod = self._create_pytorch_graph_evaluator(graph)

        try:
            import torch

            backend = str(kwargs.get("backend", "inductor"))
            mode = str(kwargs.get("mode", "default"))
            sample_inputs = kwargs.get("sample_inputs")

            if hasattr(torch, "compile"):
                try:
                    compiled_fn = getattr(torch, "compile")(mod, backend=backend, mode=mode)
                    return self._wrap_pytorch_runner(compiled_fn, torch)
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

            if sample_inputs is not None and isinstance(sample_inputs, (list, tuple)):
                pt_samples = [torch.as_tensor(x) for x in sample_inputs]
                traced = cast(Callable[..., EagerValue], getattr(torch.jit, "trace"))(mod, pt_samples)
                return self._wrap_pytorch_runner(traced, torch)

            return mod
        except Exception:
            return mod


PyTorchGenerator = PyTorchCodeGenerator
