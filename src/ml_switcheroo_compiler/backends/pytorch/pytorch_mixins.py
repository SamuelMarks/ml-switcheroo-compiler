"""Mixins."""


class PyTorchScatterVisitor:
    """Mixin."""

    def visit_TensorScatterUpdate(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterUpdate nodes."""
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]})"  # pragma: no cover

    def visit_TensorScatterAdd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Handle TensorScatterAdd nodes."""
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]}, accumulate=True)"  # pragma: no cover

    def visit_TensorScatterMax(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate tensor scatter max."""
        return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amax', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"  # pragma: no cover

    def visit_TensorScatterMin(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate tensor scatter min."""
        return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amin', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"  # pragma: no cover


class PyTorchDistributedVisitor:
    """Mixin."""

    def visit_all_gather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_gather."""
        tensor = input_vars[0]
        return f"torch.distributed.all_gather_into_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_reduce_scatter(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for reduce_scatter."""
        tensor = input_vars[0]
        return f"torch.distributed.reduce_scatter_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_all_reduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate code for all_reduce."""
        tensor = input_vars[0]
        return f"torch.distributed.all_reduce({tensor})"
