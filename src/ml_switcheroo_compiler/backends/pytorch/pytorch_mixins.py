# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Mixins."""


class PyTorchScatterVisitor:
    """Mixin."""

    def visit_TensorScatterUpdate(self, node, input_vars: list[str], **kwargs) -> str:
        """Handle TensorScatterUpdate nodes.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]})"

    def visit_TensorScatterAdd(self, node, input_vars: list[str], **kwargs) -> str:
        """Handle TensorScatterAdd nodes.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"{input_vars[0]}.clone().index_put_(tuple({input_vars[1]}.unbind(-1)), {input_vars[2]}, accumulate=True)"

    def visit_TensorScatterMax(self, node, input_vars: list[str], **kwargs) -> str:
        """Evaluate visit_TensorScatterMax operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amax', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMin(self, node, input_vars: list[str], **kwargs) -> str:
        """Evaluate visit_TensorScatterMin operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"(lambda t, i, u: t.clone().flatten().scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amin', include_self=True).reshape(t.shape))({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"


class PyTorchDistributedVisitor:
    """Mixin."""

    def visit_AllGather(self, node, input_vars: list[str], **kwargs) -> str:
        """Generate code for all_gather.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        tensor = input_vars[0]
        return f"torch.distributed.all_gather_into_tensor(output, {tensor})"

    def visit_ReduceScatter(self, node, input_vars: list[str], **kwargs) -> str:
        """Generate code for reduce_scatter.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        tensor = input_vars[0]
        return f"torch.distributed.reduce_scatter_tensor(torch.empty_like({tensor}), {tensor})"

    def visit_AllReduce(self, node, input_vars: list[str], **kwargs) -> str:
        """Generate code for all_reduce.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        tensor = input_vars[0]
        return f"torch.distributed.all_reduce({tensor})"


class PyTorchLinalgMixin:
    """Linalg."""

    """Linalg Mixin."""


class PyTorchNNMixin:
    """NN Mixin."""

    def visit_Send(self, node, input_vars: list[str], **kwargs) -> str:
        """Send tensor.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: PyTorch code for send.
        """
        dst = node.attributes.get("dst_rank", 0)
        tag = node.attributes.get("tag", 0)
        self.add_line(f"        torch.distributed.isend({input_vars[0]}, dst={dst}, tag={tag})")
        return ""

    def visit_Recv(self, node, input_vars: list[str], **kwargs) -> str:
        """Receive tensor.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs (object): Additional attributes.

        Returns:
            str: PyTorch code for recv.
        """
        src = node.attributes.get("src_rank", 0)
        tag = node.attributes.get("tag", 0)
        shape = node.attributes.get("shape", ())
        dtype = "torch." + str(node.attributes.get("dtype", "float32")).lower()
        nid = getattr(node, "id", "")
        res_var = f"v_{nid.replace('-', '_')}"
        self.add_line(f"        {res_var} = torch.empty({list(shape)}, dtype={dtype}, device=self.device)")
        self.add_line(f"        torch.distributed.irecv({res_var}, src={src}, tag={tag})")
        return res_var
