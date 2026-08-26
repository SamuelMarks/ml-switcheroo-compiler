"""Distributed AST Mixin."""

from typing import Any

from ml_switcheroo_compiler.ir.core import IRNode


class DistributedASTVisitor:
    """Provide shared AST visitors for distributed pipeline primitives."""

    def __init__(self, generator: Any) -> None:
        """Initialize the visitor.

        Args:
            generator (Any): The generator instance.
        """
        self.generator = generator

    def visit_Send(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate code for the Send operation.

        Args:
            node (IRNode): The node.
            input_vars (list[str]): Input variable names.
            **kwargs (Any): Extra arguments.

        Returns:
            str: Generated code.
        """
        target: int = int(node.attributes.get("target_stage", 0))
        self.generator.code.append(f"    # Send tensor to pipeline stage {target}")
        gen_class_name = self.generator.__class__.__name__.replace("CodeGenerator", "").replace("Generator", "").lower()
        if gen_class_name == "numpy":
            return f"_numpy_send({input_vars[0]}, target={target})"
        elif gen_class_name in ("pytorch", "torch"):
            return f"torch.distributed.isend({input_vars[0]}, dst={target})"
        elif gen_class_name == "jax":
            return f"jax.lax.send({input_vars[0]}, dst={target})"
        elif gen_class_name == "mlx":
            return f"mlx.core.distributed.send({input_vars[0]}, dst={target})"
        elif gen_class_name == "keras":
            return f"keras.distribution.send({input_vars[0]}, target={target})"
        return f"send({input_vars[0]}, target={target})"

    def visit_Recv(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate code for the Recv operation.

        Args:
            node (IRNode): The node.
            input_vars (list[str]): Input variable names.
            **kwargs (Any): Extra arguments.

        Returns:
            str: Generated code.
        """
        source: int = int(node.attributes.get("source_stage", 0))
        shape = node.shape_metadata
        dtype: str = getattr(node, "dtype", "float32")
        self.generator.code.append(f"    # Recv tensor from pipeline stage {source}")

        gen_class_name = self.generator.__class__.__name__.replace("CodeGenerator", "").replace("Generator", "").lower()
        if gen_class_name == "numpy":
            return f"_numpy_recv(source={source}, shape={shape}, dtype='{dtype}')"
        elif gen_class_name in ("pytorch", "torch"):
            return f"torch.distributed.irecv(src={source})"
        elif gen_class_name == "jax":
            return f"jax.lax.recv(src={source})"
        elif gen_class_name == "mlx":
            return f"mlx.core.distributed.recv(src={source})"
        elif gen_class_name == "keras":
            return f"keras.distribution.recv(source={source})"
        return f"recv(source={source})"
