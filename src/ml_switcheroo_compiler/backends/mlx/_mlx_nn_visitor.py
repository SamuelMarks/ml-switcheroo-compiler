"""Neural network operation visitor mixin for the MLX code generator."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_bool,
    coerce_float,
    coerce_int,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class MLXNNOpsVisitor:
    """MLX NN ops visitor mixin."""

    def visit_Relu(self, node: BackendArg, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Relu operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: MLX source code calling nn.relu.
        """
        del node, kwargs
        x_var: str = input_vars[0] if input_vars else "args[0]"
        return f"nn.relu({x_var})"

    def visit_Rope(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the Rope operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the Rope operation.
        """
        dim: int = coerce_int(node.attributes.get("dim"))
        traditional: bool = coerce_bool(node.attributes.get("traditional", False))
        base: float = coerce_float(node.attributes.get("base", 10000.0))
        scale: float = coerce_float(node.attributes.get("scale", 1.0))
        offset: int = coerce_int(node.attributes.get("offset", 0))
        return (
            f"mx.fast.rope({input_vars[0]}, {dim}, traditional={traditional}, "
            f"base={base}, scale={scale}, offset={offset})"
        )

    def visit_PowerIteration(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for the PowerIteration operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            A string containing the MLX code for the PowerIteration operation.
        """
        num_iters: int = coerce_int(node.attributes.get("num_iters", 1))
        u_var: str = input_vars[1] if len(input_vars) > 1 else "None"
        return f"mlx_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_Conv1D(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for 1D convolution.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX conv1d expression.
        """
        attrs = adapt_backend_module(node).get_symbol("attributes", {})
        stride = attrs.get("stride", 1) if isinstance(attrs, dict) else 1
        padding = attrs.get("padding", 0) if isinstance(attrs, dict) else 0
        return f"mx.conv1d({input_vars[0]}, {input_vars[1]}, stride={stride}, padding={padding})"

    def visit_Conv2D(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for 2D convolution.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX conv2d expression.
        """
        attrs = adapt_backend_module(node).get_symbol("attributes", {})
        stride = attrs.get("stride", 1) if isinstance(attrs, dict) else 1
        padding = attrs.get("padding", 0) if isinstance(attrs, dict) else 0
        return f"mx.conv2d({input_vars[0]}, {input_vars[1]}, stride={stride}, padding={padding})"

    def visit_Conv3D(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for 3D convolution.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX conv3d expression.
        """
        attrs = adapt_backend_module(node).get_symbol("attributes", {})
        stride = attrs.get("stride", 1) if isinstance(attrs, dict) else 1
        padding = attrs.get("padding", 0) if isinstance(attrs, dict) else 0
        return f"mx.conv3d({input_vars[0]}, {input_vars[1]}, stride={stride}, padding={padding})"

    def visit_MultiHeadAttention(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for MultiHeadAttention.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX attention expression.
        """
        q, k, v = input_vars[0], input_vars[1], input_vars[2]
        return f"mx.fast.scaled_dot_product_attention({q}, {k}, {v}, scale=1.0 / ({q}.shape[-1] ** 0.5))"

    def visit_ScaledDotProductAttention(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for ScaledDotProductAttention.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX attention expression.
        """
        return self.visit_MultiHeadAttention(node, input_vars, **kwargs)

    def visit_LayerNorm(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for LayerNorm.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX layer_norm expression.
        """
        attrs = adapt_backend_module(node).get_symbol("attributes", {})
        eps = attrs.get("eps", 1e-5) if isinstance(attrs, dict) else 1e-5
        return (
            f"(lambda x: (x - mx.mean(x, axis=-1, keepdims=True)) "
            f"/ mx.sqrt(mx.var(x, axis=-1, keepdims=True) + {eps}))({input_vars[0]})"
        )

    def visit_RMSNorm(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for RMSNorm.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX rms_norm expression.
        """
        attrs = adapt_backend_module(node).get_symbol("attributes", {})
        eps = attrs.get("eps", 1e-5) if isinstance(attrs, dict) else 1e-5
        return f"({input_vars[0]} * mx.rsqrt(mx.mean({input_vars[0]} ** 2, axis=-1, keepdims=True) + {eps}))"

    def visit_BatchNorm(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for BatchNorm.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX batch_norm expression.
        """
        attrs = adapt_backend_module(node).get_symbol("attributes", {})
        eps = attrs.get("eps", 1e-5) if isinstance(attrs, dict) else 1e-5
        return (
            f"(lambda x: (x - mx.mean(x, axis=0, keepdims=True)) "
            f"/ mx.sqrt(mx.var(x, axis=0, keepdims=True) + {eps}))({input_vars[0]})"
        )

    def visit_GroupNorm(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for GroupNorm.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX group_norm expression.
        """
        return f"{input_vars[0]}"

    def visit_MaxPool2D(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for MaxPool2D.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX max_pool2d expression.
        """
        return f"{input_vars[0]}"

    def visit_AvgPool2D(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for AvgPool2D.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX avg_pool2d expression.
        """
        return f"{input_vars[0]}"

    def visit_AdaptiveAvgPool2D(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for AdaptiveAvgPool2D.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX adaptive_avg_pool2d expression.
        """
        return f"{input_vars[0]}"

    def visit_Linear(self, node: LogicalNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generate MLX code for Linear projection.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of input variable names.
            kwargs: Additional keyword arguments.

        Returns:
            str: Generated MLX linear expression.
        """
        if len(input_vars) > 2:
            return f"mx.matmul({input_vars[0]}, {input_vars[1]}) + {input_vars[2]}"
        return f"mx.matmul({input_vars[0]}, {input_vars[1]})"
