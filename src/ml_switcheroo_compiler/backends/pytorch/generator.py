"""PyTorch Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("pytorch")
class PyTorchCodeGenerator(BaseGenerator):
    """Emit PyTorch-compatible code from IR."""

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit a node and return the PyTorch code string.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated PyTorch Python code
        """
        op_type = getattr(node, "op_type", "")

        ops_map = self._get_ops_map(kwargs)

        if op_type in ops_map:
            return self._format_mapped_op(ops_map[op_type], input_vars, kwargs)

        return self._format_generic_fallback(op_type, input_vars, kwargs)

    def _get_ops_map(self, kwargs: dict) -> dict:
        """Execute _get_ops_map.

        Args:
            kwargs (Any): Argument kwargs.

        Returns:
        Any: The result.
        """
        return {
            "Matmul": "torch.matmul({0}, {1})",
            "Dot": "torch.dot({0}, {1})",
            "BroadcastTo": "{0}.expand({shape})",
            "Reshape": "torch.reshape({0}, {shape})",
            "TrueDivide": "torch.true_divide({0}, {1})",
            "Zeros": "torch.zeros({shape})",
            "Ones": "torch.ones({shape})",
            "Full": "torch.full({shape}, {fill_value})",
            "Arange": "torch.arange({0})",
            "AssignVariable": "{0}",
            "ReadVariable": "{0}",
            "Transpose": "torch.permute({0}, {axes})" if "axes" in kwargs else "{0}.t()",
            "Einsum": "torch.einsum({subscripts}, {0})",
            "Sum": "torch.sum({0}, dim={axis}, keepdim={keepdims})",
            "Mean": "torch.mean({0}, dim={axis}, keepdim={keepdims})",
            "Max": "torch.max({0}, dim={axis}, keepdim={keepdims})",
        }

    def _format_mapped_op(self, fmt: str, input_vars: list[str], kwargs: dict) -> str:
        """Execute _format_mapped_op.

        Args:
            fmt (Any): Argument fmt.
            input_vars (Any): Argument input_vars.
            kwargs (Any): Argument kwargs.

        Returns:
        Any: The result.
        """
        # Replace kwargs placeholders
        for k, v in kwargs.items():
            if f"{{{k}}}" in fmt:
                fmt = fmt.replace(f"{{{k}}}", str(v))
        # Special case for keepdims
        if "keepdims" in fmt and "keepdims" not in kwargs:
            fmt = fmt.replace("keepdim={keepdims}", "keepdim=False")
        if "axis" in fmt and "axis" not in kwargs:
            fmt = fmt.replace(", dim={axis}", "")
        # Replace args placeholders
        for i, var in enumerate(input_vars):
            fmt = fmt.replace(f"{{{i}}}", var)
        return fmt

    def _format_generic_fallback(self, op_type: str, input_vars: list[str], kwargs: dict) -> str:
        """Execute _format_generic_fallback.

        Args:
            op_type (Any): Argument op_type.
            input_vars (Any): Argument input_vars.
            kwargs (Any): Argument kwargs.

        Returns:
        Any: The result.
        """
        return super()._format_generic_fallback(
            "torch", op_type, input_vars, kwargs, axis_kwarg="dim", keepdims_kwarg="keepdim"
        )

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = self.{var_name}")

    def generate(self) -> str:
        """Generate PyTorch model code from the IR graph.

        Returns:
            str: The generated PyTorch Python code
        """
        self.code = [
            self.header.strip(),
            "import torch",
            "import torch.nn as nn\n",
            "class CompiledModel(nn.Module):",
        ]

        # __init__
        self.indent_level = 1
        self.add_line("def __init__(self):")
        self.indent_level += 1
        self.add_line("super().__init__()")

        has_params = False
        for node in self.sorted_nodes:
            if node.op_type == "Constant":
                val_repr = self.emit_constant(node)
                var_name = self.assign_var_name(node.id, "const")
                self.add_line(
                    f"self.register_parameter('{var_name}', "
                    f"nn.Parameter(torch.tensor({val_repr})))",
                )
                has_params = True

        if not has_params:
            self.add_line("pass")

        self.add_line("")
        self.indent_level -= 1

        # forward
        self.add_line("def forward(self, *args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)
