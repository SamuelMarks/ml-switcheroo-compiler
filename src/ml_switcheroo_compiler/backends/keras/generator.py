# ruff: noqa: E402
"""Keras backend generator that maps LogicalNodes and IR layers to Keras equivalent code representations."""

from __future__ import annotations

import os

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, require_callable
from ml_switcheroo_compiler.backends.keras.eager import execute_op as _keras_execute_op
from ml_switcheroo_compiler.backends.protocols import EagerBackendAdapter, StaticEagerBackendAdapter
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.core.shape import ShapeMetadataProtocol, extract_data

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
        raw_node: BackendArg = node
        shape_str: str = (
            str(raw_node.shape_metadata)
            if isinstance(raw_node, ShapeMetadataProtocol) and raw_node.shape_metadata
            else "(None,)"
        )
        return f"{var_name} = keras.Input(shape={shape_str}, name='{node.id or ''}')"

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
    def format_zeros_like(op: str, kwargs: dict[str, BackendArg]) -> str:
        """Execute format zeros like for compiler processing.

        Args:
            op (str): Input `op` argument (op).
            kwargs: Input `kwargs` argument (kwargs).

        Returns:
            str: Computed `str` output.
        """
        res: str = f"keras.ops.{op}({{shape}})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    @staticmethod
    def format_full(kwargs: dict[str, BackendArg]) -> str:
        """Execute format full for compiler processing.

        Args:
            kwargs: Input `kwargs` argument (kwargs).

        Returns:
            str: Computed `str` output.
        """
        res: str = "keras.ops.full({shape}, {fill_value})"
        if "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    @staticmethod
    def format_transpose(kwargs: dict[str, BackendArg]) -> str:
        """Execute format transpose for compiler processing.

        Args:
            kwargs: Input `kwargs` argument (kwargs).

        Returns:
            str: Computed `str` output.
        """
        if "axes" in kwargs:
            return "keras.ops.transpose({0}, {axes})"
        return "keras.ops.transpose({0})"


@register_backend("keras")
class KerasCodeGenerator(BaseGenerator):
    """Emit Keras Functional API script from IR."""

    keras_input_vars: list[str] = []
    keras_output_vars: list[str] = []

    @classmethod
    def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an operation eagerly on the Keras backend.

        Args:
            op_type (str): Canonical operation name.
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            BackendArg: Evaluated operation result.
        """
        return _keras_execute_op(cls, op_type, *args, **kwargs)

    @classmethod
    def load(
        cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII"
    ) -> BackendArg:
        """Load a serialized object.

        Args:
            filepath (str): The file path.
            allow_pickle (bool): Allow pickle.
            fix_imports (bool): Fix imports.
            encoding (str): The encoding.

        Returns:
            BackendArg: The loaded object.
        """
        import pickle

        with open(filepath, "rb") as f:
            loaded: BackendArg = pickle.load(f)
            return loaded

    @classmethod
    def save(cls: type, file: str, arr: BackendArg, allow_pickle: bool = True, fix_imports: bool = True) -> None:
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
    def savez(cls: type, file: str, *args: BackendArg, **kwds: BackendArg) -> None:
        """Save multiple arrays into a single file.

        Args:
            file (str): The file path.
            *args: Positional array arguments.
            **kwds: Keyword array arguments.
        """
        import pickle

        data: dict[str, BackendArg] = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        with open(file, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: BackendArg, **kwds: BackendArg) -> None:
        """Save multiple arrays into a single compressed file.

        Args:
            file (str): The file path.
            *args: Positional array arguments.
            **kwds: Keyword array arguments.
        """
        import gzip
        import pickle

        data: dict[str, BackendArg] = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        with gzip.open(file, "wb") as f:
            pickle.dump(data, f)

    def _get_backend_prefix(self) -> str:
        """Retrieve the backend prefix property or mapping.

        Returns:
            str: The evaluated or processed output.
        """
        return "keras"

    def __init__(self, graph: IRGraph, delegates: list[CodeGeneratorVisitor] | None = None) -> None:
        """Initialize the generator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates: Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.visitors.extend([*get_shared_ast_visitors(generator=self), KerasVisionVisitor(), KerasAudioVisitor()])
        self.keras_input_vars: list[str] = []
        self.keras_output_vars: list[str] = []

    def visit_ConvTranspose(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Visit a ConvTranspose node.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): The inputs.
            **kwargs: Additional kwargs.

        Returns:
            str: The generated code string.
        """
        return f"keras_conv_transpose({input_vars[0]}, {input_vars[1]})"

    def visit_RaggedDot(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
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
        if not self.graph.nodes:
            self.add_line("pass")
            return "\n".join(self.code)
        self._generate_body("input_")
        if self.keras_input_vars and self.keras_output_vars:
            self.add_line(KerasSignatureBuilder.get_return_block(self.keras_input_vars, self.keras_output_vars))
        elif self.keras_output_vars:
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

    def get_ops_map(self, kwargs: dict[str, BackendArg] | None = None) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: The kwargs.

        Returns:
            dict[str, str]: The ops map.
        """
        kw = kwargs if kwargs is not None else {}
        ops = super().get_ops_map(kw)
        ops["Zeros"] = KerasTensorManipulator.format_zeros_like("zeros", kw)
        ops["Ones"] = KerasTensorManipulator.format_zeros_like("ones", kw)
        ops["Full"] = KerasTensorManipulator.format_full(kw)
        ops["Transpose"] = KerasTensorManipulator.format_transpose(kw)
        return ops

    def _emit_input_assignment(self, var_name: str, node: IRNode, input_prefix: str, input_idx: int) -> None:
        """Execute emit input assignment for compiler processing.

        Args:
            var_name (str): Input `var_name` argument (var name).
            node (IRNode): Input `node` argument (node).
            input_prefix (str): Input `input_prefix` argument (input prefix).
            input_idx (int): Input `input_idx` argument (input idx).
        """
        self.add_line(KerasSignatureBuilder.get_input_assignment(var_name, node))
        self.keras_input_vars.append(var_name)

    def _emit_body_return(self, returns: list[str]) -> None:
        """Execute emit body return for compiler processing.

        Args:
            returns (list[str]): Input `returns` argument (returns).
        """
        self.keras_output_vars.extend(returns)

    def _emit_output_assignment(self, node: IRNode, input_vars: list[str], returns: str) -> None:
        """Emit output assignment.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): Input `input_vars` argument (input vars).
            returns (str): Input `returns` argument (returns).
        """
        self.keras_output_vars.extend(input_vars)

    def _generate_file_header(self) -> list[str]:
        """Execute generate file header for compiler processing.

        Returns:
            list[str]: Computed `list[str]` output.
        """
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Execute resolve imports for compiler processing.

        Returns:
            list[str]: Computed `list[str]` output.
        """
        import yaml

        tmpl_path: str = os.path.join(os.path.dirname(__file__), "keras_prefix.yaml")
        with open(tmpl_path) as f:
            data: dict[str, BackendArg] = yaml.safe_load(f)

        lines: list[str] = ["import keras\n"]
        if "imports" in data and data["imports"]:
            lines.extend(str(data["imports"]).split("\n"))
        funcs_dict = data.get("functions")
        if isinstance(funcs_dict, dict) and funcs_dict:
            for func_code in funcs_dict.values():
                lines.extend(str(func_code).split("\n"))
        return lines

    def _generate_function_signature(self) -> None:
        """Generate the model function signature."""
        self.indent_level = 0
        self.add_line("def get_model() -> object:")
        self.keras_input_vars = []
        self.keras_output_vars = []
        self.indent_level = self.indent_level + 1

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Compile IRGraph into ahead-of-time Keras executable artifact.

        Args:
            graph (IRGraph): Target computation graph to compile.
            **kwargs (BackendArg): Compiler options.

        Returns:
            BackendArg: Executable compiled artifact.
        """
        from ml_switcheroo_compiler.backends._module_loader import load_generated_code
        from ml_switcheroo_compiler.backends.base_generator import CompiledArtifact

        source_code = self.generate()
        model_obj = None
        try:
            scope = load_generated_code(source_code)
            if "get_model" in scope:
                model_obj = require_callable(scope["get_model"])()
        except (
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            ArithmeticError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ):
            model_obj = None

        def runner(*args: BackendArg, **runner_kwargs: BackendArg) -> BackendArg:
            """Execute compiled Keras artifact using instantiated model or graph fallback.

            Args:
                *args (BackendArg): Input tensor arguments.
                **runner_kwargs (BackendArg): Optional execution options.

            Returns:
                BackendArg: Model execution output or evaluated graph output.
            """
            if model_obj is not None:
                return require_callable(model_obj)(*args, **runner_kwargs)

            from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

            feed_dict: dict[str, BackendArg] = {}
            for idx, arg in enumerate(args):
                if idx < len(graph.inputs):
                    feed_dict[graph.inputs[idx]] = extract_data(arg)
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

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for Keras (`keras.ops`).

        Returns:
            EagerBackendAdapter: Concrete Keras eager backend adapter.
        """
        import keras

        return StaticEagerBackendAdapter(
            backend_name="keras",
            ops={
                "abs": keras.ops.abs,
                "add": keras.ops.add,
                "subtract": keras.ops.subtract,
                "multiply": keras.ops.multiply,
                "divide": keras.ops.divide,
                "sin": keras.ops.sin,
                "cos": keras.ops.cos,
                "tanh": keras.ops.tanh,
                "exp": keras.ops.exp,
                "log": keras.ops.log,
                "sqrt": keras.ops.sqrt,
                "matmul": keras.ops.matmul,
                "sum": keras.ops.sum,
                "mean": keras.ops.mean,
                "reshape": keras.ops.reshape,
                "transpose": keras.ops.transpose,
                "asarray": keras.ops.convert_to_tensor,
                "array": keras.ops.convert_to_tensor,
                "zeros": keras.ops.zeros,
                "ones": keras.ops.ones,
                "empty": keras.ops.empty,
            },
            dtypes={
                "float16": "float16",
                "float32": "float32",
                "float64": "float64",
                "int32": "int32",
                "int64": "int64",
                "bool": "bool",
            },
            submodules={
                "ops": keras.ops,
                "linalg": keras.ops.linalg,
                "nn": keras.ops.nn,
                "image": keras.ops.image,
            },
            module=keras,
        )
