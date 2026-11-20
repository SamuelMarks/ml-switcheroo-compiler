"""TensorFlow Target Emission."""

from __future__ import annotations

from ml_switcheroo_ir import IRGraph, LogicalNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.protocols import EagerBackendAdapter, StaticEagerBackendAdapter
from ml_switcheroo_compiler.backends.registry import register_backend

from .eager import execute_op as _tf_execute_op


@register_backend("tensorflow")
class TensorFlowCodeGenerator(BaseGenerator):
    """Emit TensorFlow-compatible code from IR."""

    @classmethod
    def execute_op(cls, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an eager operation natively on the TensorFlow backend.

        Args:
            op_type (str): Canonical IR operation name.
            *args (BackendArg): Positional tensor or scalar operands.
            **kwargs (BackendArg): Keyword operation attributes.

        Returns:
            BackendArg: Evaluated TensorFlow tensor or collection.
        """
        return _tf_execute_op(cls, op_type, *args, **kwargs)

    @classmethod
    def load(
        cls: type, filepath: str, allow_pickle: bool = False, fix_imports: bool = True, encoding: str = "ASCII"
    ) -> BackendArg:
        """Load.

        Args:
        filepath (str): Input `filepath` argument (filepath).
        allow_pickle (bool): Input `allow_pickle` argument (allow pickle).
        fix_imports (bool): Input `fix_imports` argument (fix imports).
        encoding (str): Input `encoding` argument (encoding).

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        import pickle

        with open(filepath, "rb") as f:
            loaded: BackendArg = pickle.load(f)
            return loaded

    @classmethod
    def save(cls: type, file: str, arr: BackendArg, allow_pickle: bool = True, fix_imports: bool = True) -> None:
        """Save.

        Args:
            file (str): Input `file` argument (file).
            arr (BackendArg): Input `arr` argument (arr).
            allow_pickle (bool): Input `allow_pickle` argument (allow pickle).
            fix_imports (bool): Input `fix_imports` argument (fix imports).
        """
        import pickle

        with open(file, "wb") as f:
            pickle.dump(arr, f)

    @classmethod
    def savez(cls: type, file: str, *args: BackendArg, **kwds: BackendArg) -> None:
        """Savez.

        Args:
            file (str): Input `file` argument (file).
            *args (BackendArg): Positional args.
            **kwds (BackendArg): Keyword args.
        """
        import pickle

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        with open(file, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def savez_compressed(cls: type, file: str, *args: BackendArg, **kwds: BackendArg) -> None:
        """Savez compressed.

        Args:
            file (str): Input `file` argument (file).
            *args (BackendArg): Positional args.
            **kwds (BackendArg): Keyword args.
        """
        import gzip
        import pickle

        data = {f"arr_{i}": arg for i, arg in enumerate(args)}
        data.update(kwds)
        with gzip.open(file, "wb") as f:
            pickle.dump(data, f)

    def __init__(self, graph: IRGraph) -> None:
        """Init.

        Args:
            graph (IRGraph): Input `graph` argument (graph).
        """
        super().__init__(graph)
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def _format_zeros_like(self, op: str, kwargs: BackendArg) -> str:
        """Execute format zeros like for compiler processing.

        Args:
        op (str): Input `op` argument (op).
        kwargs (BackendArg): Input `kwargs` argument (kwargs).

        Returns:
        str: Computed `str` output.
        """
        res = f"tf.{op}({{shape}})"
        if isinstance(kwargs, dict) and "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_full(self, kwargs: BackendArg) -> str:
        """Execute format full for compiler processing.

        Args:
        kwargs (BackendArg): Input `kwargs` argument (kwargs).

        Returns:
        str: Computed `str` output.
        """
        res = "tf.full({shape}, {fill_value})"
        if isinstance(kwargs, dict) and "dtype" in kwargs:
            res += f", dtype='{kwargs['dtype']}'"
        return res

    def _format_transpose(self, kwargs: BackendArg) -> str:
        """Execute format transpose for compiler processing.

        Args:
        kwargs (BackendArg): Input `kwargs` argument (kwargs).

        Returns:
        str: Computed `str` output.
        """
        if isinstance(kwargs, dict) and "axes" in kwargs:
            return "tf.transpose({0}, perm={axes})"
        return "tf.transpose({0})"

    def visit_RaggedDot(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Emit target backend code or IR transformation for the RaggedDot operation.

        Args:
        node (LogicalNode): Input `node` argument (node).
        input_vars (list[str]): Input `input_vars` argument (input vars).
        **kwargs (BackendArg): Keyword args.

        Returns:
        str: Computed `str` output.
        """
        return f"tf_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Handle Einsum nodes.

        Args:
            node (LogicalNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Extra arguments.

        Returns:
            str: The code string.
        """
        args_str = ", ".join(input_vars)
        eq = kwargs.get("equation", "")
        return f"tf.einsum('{eq}', {args_str})"

    def generate(self) -> str:
        """Generate TensorFlow code directly from the IR graph without NumPy bypass.

        Returns:
            str: Generated TensorFlow source code.
        """
        self.code = [self.header]
        self.code.extend(self._resolve_imports())
        self._generate_function_signature()
        self._generate_body("args")
        return "\n".join(self.code)

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
        str: Computed `str` output.
        """
        return "tf.math"

    def _get_creation_ops(self, kwargs: BackendArg) -> dict[str, str]:
        """Execute get creation ops for compiler processing.

        Args:
            kwargs (BackendArg): Input `kwargs` argument (kwargs).

        Returns:
            dict: Computed `dict` output.
        """
        return {
            "Arange": "tf.range({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
        }

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Execute emit constant assignment for compiler processing.

        Args:
            var_name (str): Input `var_name` argument (var name).
            val_repr (str): Input `val_repr` argument (val repr).
        """
        self.add_line(f"{var_name} = tf.constant({val_repr})")

    def _generate_file_header(self) -> list[str]:
        """Execute generate file header for compiler processing.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Execute resolve imports for compiler processing.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ["import tensorflow as tf\n"]

    def _generate_function_signature(self) -> None:
        """Execute generate function signature for compiler processing.

        Args:
        self (OpValue): Input `self` argument (self).

        Returns:
        NoneType: Computed `NoneType` output.
        """
        self.indent_level = 0
        self.add_line("@tf.function")
        self.add_line("def apply_model(*args, **kwargs) -> object:")
        self.indent_level += 1

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> BackendArg:
        """Compile IRGraph into an ahead-of-time (AOT) callable using tf.function(jit_compile=True).

        Args:
            graph (IRGraph): Target computation graph.
            **kwargs (BackendArg): Optional compilation options ('jit_compile').

        Returns:
            BackendArg: Compiled executable callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        def forward_fn(*fn_args: BackendArg) -> BackendArg:
            """Evaluate the graph with TensorFlow tensors.

            Args:
                *fn_args (BackendArg): Input tensor values.

            Returns:
                BackendArg: Computed tensor or collection of tensors.
            """
            input_nodes = [n for n in graph.nodes.values() if n.op_type == "Input"]
            inputs: dict[str, BackendArg] = {}
            for i, inp_node in enumerate(input_nodes):
                if i < len(fn_args):
                    arg_val = fn_args[i]
                    inputs[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
            evaluated = evaluate_graph(graph, inputs=inputs)
            if graph.outputs:
                if len(graph.outputs) == 1:
                    return evaluated.get(graph.outputs[0])
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return evaluated

        try:
            import tensorflow as tf

            jit_compile = bool(kwargs.get("jit_compile", True))
            from ml_switcheroo_compiler.backends.eager_registry import require_callable

            compiled_fn = require_callable(tf.function)(forward_fn, jit_compile=jit_compile)

            def aot_tf_runner(*w_args: BackendArg, **w_kwargs: BackendArg) -> BackendArg:
                """Execute AOT compiled TensorFlow function.

                Args:
                    *w_args (BackendArg): Input tensors.
                    **w_kwargs (BackendArg): Keyword arguments.

                Returns:
                    BackendArg: Computed outputs.
                """
                conv_fn = require_callable(tf.convert_to_tensor)
                tf_args = [conv_fn(a.data if isinstance(a, Tensor) else a) for a in w_args]
                return require_callable(compiled_fn)(*tf_args)

            return aot_tf_runner
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
            return forward_fn

    @classmethod
    def get_eager_adapter(cls) -> EagerBackendAdapter:
        """Return the explicitly bound `EagerBackendAdapter` for TensorFlow.

        Returns:
            EagerBackendAdapter: Concrete TensorFlow eager backend adapter.
        """
        import tensorflow as tf

        return StaticEagerBackendAdapter(
            backend_name="tensorflow",
            ops={
                "abs": tf.abs,
                "add": tf.add,
                "subtract": tf.subtract,
                "multiply": tf.multiply,
                "divide": tf.divide,
                "sin": tf.math.sin,
                "cos": tf.math.cos,
                "tanh": tf.math.tanh,
                "exp": tf.math.exp,
                "log": tf.math.log,
                "sqrt": tf.math.sqrt,
                "matmul": tf.matmul,
                "asarray": tf.convert_to_tensor,
                "array": tf.constant,
                "zeros": tf.zeros,
                "ones": tf.ones,
            },
            dtypes={
                "float16": tf.float16,
                "float32": tf.float32,
                "float64": tf.float64,
                "bfloat16": tf.bfloat16,
                "int8": tf.int8,
                "int16": tf.int16,
                "int32": tf.int32,
                "int64": tf.int64,
                "uint8": tf.uint8,
                "bool": tf.bool,
                "complex64": tf.complex64,
                "complex128": tf.complex128,
            },
            submodules={
                "linalg": tf.linalg,
                "math": tf.math,
                "nn": tf.nn,
                "image": tf.image,
                "signal": tf.signal,
            },
            module=tf,
        )


TensorFlowGenerator = TensorFlowCodeGenerator
