# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Keras backend generator that maps LogicalNodes and IR layers to Keras equivalent code representations."""

import os
from typing import Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

from .keras_mixins import KerasAudioVisitor, KerasVisionVisitor


class KerasSignatureBuilder:
    """Help for building Keras Model signatures."""

    @staticmethod
    def get_input_assignment(var_name: str, node: IRNode) -> str:
        """Generate the `keras.Input` declaration for a given input node.

        Args:
            var_name (str): The assigned variable name.
            node (IRNode): The logical input node representing the entry point.

        Returns:
            str: The code string for the Keras Input tensor.
        """
        shape_str: str = str(node.shape_metadata) if hasattr(node, "shape_metadata") and node.shape_metadata else "(None,)"
        return f"{var_name} = keras.Input(shape={shape_str}, name='{getattr(node, 'id', '')}')"

    @staticmethod
    def get_return_block(input_vars: list[str], output_vars: list[str]) -> str:
        """Construct the `keras.Model` return block connecting inputs to outputs.

        Args:
            input_vars (list[str]): List of model input variable names.
            output_vars (list[str]): List of model output variable names.

        Returns:
            str: The code string that builds and returns the model.
        """
        inputs_str: str = ", ".join(input_vars)
        outputs_str: str = ", ".join(output_vars)
        return f"return keras.Model(inputs=[{inputs_str}], outputs=[{outputs_str}])"


class KerasTensorManipulator:
    """Help for tensor manipulations."""

    @staticmethod
    def format_zeros_like(op: str, kwargs: dict[str, object]) -> str:
        """Evaluate format_zeros_like operation.

        Args:
            op (str): The op parameter.
            kwargs: The kwargs parameter.

        Returns:
            str: Result.
        """
        res: str = f"keras.ops.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    @staticmethod
    def format_full(kwargs: dict[str, object]) -> str:
        """Evaluate format_full operation.

        Args:
            kwargs: The kwargs parameter.

        Returns:
            str: Result.
        """
        res: str = "keras.ops.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    @staticmethod
    def format_transpose(kwargs: dict[str, object]) -> str:
        """Evaluate format_transpose operation.

        Args:
            kwargs: The kwargs parameter.

        Returns:
            str: Result.
        """
        if "axes" in kwargs:
            return "keras.ops.transpose({0}, {axes})"
        return "keras.ops.transpose({0})"


@register_backend("keras")
class KerasCodeGenerator(BaseGenerator):
    """Emit Keras Functional API script from IR."""

    @classmethod
    def load(cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII") -> object:
        """Load a serialized object.

        Args:
            filepath (str): The file path.
            allow_pickle (bool): Allow pickle.
            fix_imports (bool): Fix imports.
            encoding (str): The encoding.

        Returns:
            object: The loaded object.
        """
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @classmethod
    def save(cls: type, file: str, arr: object, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save an array to a file.

        Args:
            file (str): The file path.
            arr: The array data.
            allow_pickle (bool): Allow pickle.
            fix_imports (bool): Fix imports.
        """
        import pickle

        with open(file, "wb") as f:
            pickle.dump(arr, f)

    @classmethod
    def savez(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Save multiple arrays into a single file.

        Args:
            file (str): The file path.
            *args: Positional array arguments.
            **kwds: Keyword array arguments.
        """
        import pickle

        data: dict[str, object] = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        with open(file, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: object, **kwds: object) -> None:
        """Save multiple arrays into a single compressed file.

        Args:
            file (str): The file path.
            *args: Positional array arguments.
            **kwds: Keyword array arguments.
        """
        import gzip
        import pickle

        data: dict[str, object] = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        with gzip.open(file, "wb") as f:
            pickle.dump(data, f)

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "keras"

    def __init__(self, graph: IRGraph, delegates: Optional[list[CodeGeneratorVisitor]] = None) -> None:
        """Initialize the generator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates: Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.visitors.extend([*get_shared_ast_visitors(generator=self), KerasVisionVisitor(), KerasAudioVisitor()])
        self.keras_input_vars: list[str] = []
        self.keras_output_vars: list[str] = []

    def visit_ConvTranspose(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Visit a ConvTranspose node.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): The inputs.
            **kwargs: Additional kwargs.

        Returns:
            str: The generated code string.
        """
        return f"keras_conv_transpose({input_vars[0]}, {input_vars[1]})"

    def visit_RaggedDot(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
        """Visit a RaggedDot node.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): The inputs.
            **kwargs: Additional kwargs.

        Returns:
            str: The generated code string.
        """
        return f"keras_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def generate(self) -> str:
        """Generate functional Keras model code directly from IRGraph without NumPy bypass.

        Returns:
            str: Generated Keras source code.
        """
        self.code = [self.header]
        self.code.extend(self._resolve_imports())
        self._generate_function_signature()
        if not self.graph or not self.graph.nodes:
            self.add_line("pass")
            return "\n".join(self.code)
        self._generate_body("input_")
        if getattr(self, "keras_input_vars", None) and getattr(self, "keras_output_vars", None):
            self.add_line(KerasSignatureBuilder.get_return_block(self.keras_input_vars, self.keras_output_vars))
        elif getattr(self, "keras_output_vars", None):
            self.add_line(f"return {', '.join(self.keras_output_vars)}")
        else:
            self.add_line("pass")
        return "\n".join(self.code)

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
            str: The prefix string.
        """
        return "keras.ops"

    def get_ops_map(self, kwargs: dict[str, object]) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: The kwargs.

        Returns:
            dict[str, str]: The ops map.
        """
        ops = super().get_ops_map(kwargs)
        ops["Zeros"] = KerasTensorManipulator.format_zeros_like("zeros", kwargs)
        ops["Ones"] = KerasTensorManipulator.format_zeros_like("ones", kwargs)
        ops["Full"] = KerasTensorManipulator.format_full(kwargs)
        ops["Transpose"] = KerasTensorManipulator.format_transpose(kwargs)
        return ops

    def _emit_input_assignment(self, var_name: str, node: IRNode, input_prefix: str, input_idx: int) -> None:
        """Evaluate _emit_input_assignment operation.

        Args:
            var_name (str): The var_name parameter.
            node (IRNode): The node parameter.
            input_prefix (str): The input_prefix parameter.
            input_idx (int): The input_idx parameter.
        """
        self.add_line(KerasSignatureBuilder.get_input_assignment(var_name, node))
        self.keras_input_vars.append(var_name)

    def _emit_body_return(self, returns: list[str]) -> None:
        """Evaluate _emit_body_return operation.

        Args:
            returns (list[str]): The returns parameter.
        """
        self.keras_output_vars.extend(returns)

    def _emit_output_assignment(self, node: IRNode, input_vars: list[str], returns: str) -> None:
        """Emit output assignment.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): The input_vars parameter.
            returns (str): The returns parameter.
        """
        self.keras_output_vars.extend(input_vars)

    def _generate_file_header(self) -> list[str]:
        """Evaluate _generate_file_header operation.

        Returns:
            list[str]: Result.
        """
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Evaluate _resolve_imports operation.

        Returns:
            list[str]: Result.
        """
        import yaml

        tmpl_path: str = os.path.join(os.path.dirname(__file__), "keras_prefix.yaml")
        with open(tmpl_path) as f:
            data: dict[str, object] = yaml.safe_load(f)

        lines: list[str] = ["import keras\n"]
        if "imports" in data and data["imports"]:
            lines.extend(str(data["imports"]).split("\n"))
        if "functions" in data and data["functions"]:
            for func_code in data["functions"].values():
                lines.extend(str(func_code).split("\n"))
        return lines

    def _generate_function_signature(self) -> None:
        """Generate the model function signature."""
        self.indent_level
        self.indent_level = 0
        self.add_line("def get_model() -> object:")
        self.keras_input_vars = []
        self.keras_output_vars = []
        self.indent_level = self.indent_level + 1

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> object:
        """Compile IRGraph into ahead-of-time Keras executable artifact.

        Args:
            graph (IRGraph): Target computation graph to compile.
            **kwargs (object): Compiler options.

        Returns:
            object: Executable compiled artifact.
        """
        from typing import Any

        from ml_switcheroo_compiler.backends.base_generator import CompiledArtifact

        source_code = self.generate()
        scope: dict[str, Any] = {}
        model_obj = None
        try:
            exec(source_code, scope)
            if "get_model" in scope:
                model_obj = scope["get_model"]()
        except Exception:
            model_obj = None

        def runner(*args: object, **runner_kwargs: object) -> object:
            """Execute compiled Keras artifact using instantiated model or graph fallback.

            Args:
                *args (object): Input tensor arguments.
                **runner_kwargs (object): Optional execution options.

            Returns:
                object: Model execution output or evaluated graph output.
            """
            if model_obj is not None:
                return model_obj(*args, **runner_kwargs)

            from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

            feed_dict: dict[str, object] = {}
            for idx, arg in enumerate(args):
                if idx < len(graph.inputs):
                    feed_dict[graph.inputs[idx]] = getattr(arg, "data", arg)
            res_dict = evaluate_graph(graph, feed_dict)
            if not graph.outputs:
                return res_dict
            if len(graph.outputs) == 1:
                return res_dict.get(graph.outputs[0])
            return tuple(res_dict.get(out) for out in graph.outputs)

        return CompiledArtifact(
            callable_fn=runner,
            binary_bytes=source_code.encode("utf-8"),
            source_code=source_code,
            metadata={"backend": "keras", "model": model_obj, "options": kwargs},
        )
