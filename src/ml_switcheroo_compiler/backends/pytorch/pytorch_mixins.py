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

    def visit_AllToAll(self, node, input_vars: list[str], **kwargs) -> str:
        """Generate code for all_to_all.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        tensor = input_vars[0]
        return f"torch.distributed.all_to_all_single(torch.empty_like({tensor}), {tensor})"

    def visit_Broadcast(self, node, input_vars: list[str], **kwargs) -> str:
        """Generate code for broadcast.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        tensor = input_vars[0]
        src = getattr(node, "attributes", {}).get("root", 0)
        return f"torch.distributed.broadcast({tensor}, src={src})"


class PyTorchLinalgMixin:
    """Linalg."""

    """Linalg Mixin."""


class PyTorchNNMixin:
    """NN Mixin."""

    def __init__(self, *args: object, generator: object = None, **kwargs: object) -> None:
        """Initialize PyTorchNNMixin.

        Args:
            *args (object): Variable length arguments.
            generator (object): The parent code generator instance.
            **kwargs (object): Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.generator = generator

    def _add_line(self, line: str) -> None:
        """Add a line to code output.

        Args:
            line (str): Line of code to append.
        """
        if hasattr(self, "add_line"):
            self.add_line(line)
        elif self.generator and hasattr(self.generator, "add_line"):
            self.generator.add_line(line)

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
        self._add_line(f"        torch.distributed.isend({input_vars[0]}, dst={dst}, tag={tag})")
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
        self._add_line(f"        {res_var} = torch.empty({list(shape)}, dtype={dtype}, device=self.device)")
        self._add_line(f"        torch.distributed.irecv({res_var}, src={src}, tag={tag})")
        return res_var

    def visit_Conv1D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for 1D convolution.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch conv1d expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        stride = attrs.get("stride", 1)
        padding = attrs.get("padding", 0)
        dilation = attrs.get("dilation", 1)
        groups = attrs.get("groups", 1)
        bias = input_vars[2] if len(input_vars) > 2 else "None"
        return f"torch.nn.functional.conv1d({input_vars[0]}, {input_vars[1]}, bias={bias}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups})"

    def visit_Conv2D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for 2D convolution.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch conv2d expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        stride = attrs.get("stride", 1)
        padding = attrs.get("padding", 0)
        dilation = attrs.get("dilation", 1)
        groups = attrs.get("groups", 1)
        bias = input_vars[2] if len(input_vars) > 2 else "None"
        return f"torch.nn.functional.conv2d({input_vars[0]}, {input_vars[1]}, bias={bias}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups})"

    def visit_Conv3D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for 3D convolution.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch conv3d expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        stride = attrs.get("stride", 1)
        padding = attrs.get("padding", 0)
        dilation = attrs.get("dilation", 1)
        groups = attrs.get("groups", 1)
        bias = input_vars[2] if len(input_vars) > 2 else "None"
        return f"torch.nn.functional.conv3d({input_vars[0]}, {input_vars[1]}, bias={bias}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups})"

    def visit_MultiHeadAttention(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for MultiHeadAttention / ScaledDotProductAttention.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Query, Key, Value input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch attention expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        is_causal = bool(attrs.get("is_causal", False))
        q, k, v = input_vars[0], input_vars[1], input_vars[2]
        return f"torch.nn.functional.scaled_dot_product_attention({q}, {k}, {v}, is_causal={is_causal})"

    def visit_ScaledDotProductAttention(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for scaled dot product attention.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Query, Key, Value input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch attention expression.
        """
        return self.visit_MultiHeadAttention(node, input_vars, **kwargs)

    def visit_LayerNorm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for LayerNorm.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch layer_norm expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        shape = attrs.get("normalized_shape", [1])
        eps = attrs.get("eps", 1e-5)
        weight = input_vars[1] if len(input_vars) > 1 else "None"
        bias = input_vars[2] if len(input_vars) > 2 else "None"
        return f"torch.nn.functional.layer_norm({input_vars[0]}, {shape}, weight={weight}, bias={bias}, eps={eps})"

    def visit_RMSNorm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for RMSNorm.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch rms_norm expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        shape = attrs.get("normalized_shape", [1])
        eps = attrs.get("eps", 1e-5)
        weight = input_vars[1] if len(input_vars) > 1 else "None"
        return f"(torch.nn.functional.rms_norm({input_vars[0]}, {shape}, weight={weight}, eps={eps}) if hasattr(torch.nn.functional, 'rms_norm') else ({input_vars[0]} * torch.rsqrt(torch.mean({input_vars[0]} ** 2, dim=-1, keepdim=True) + {eps})))"

    def visit_BatchNorm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for BatchNorm.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch batch_norm expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        eps = attrs.get("eps", 1e-5)
        mean = input_vars[1] if len(input_vars) > 1 else "None"
        var = input_vars[2] if len(input_vars) > 2 else "None"
        return f"torch.nn.functional.batch_norm({input_vars[0]}, running_mean={mean}, running_var={var}, eps={eps})"

    def visit_GroupNorm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for GroupNorm.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch group_norm expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        num_groups = attrs.get("num_groups", 1)
        eps = attrs.get("eps", 1e-5)
        return f"torch.nn.functional.group_norm({input_vars[0]}, num_groups={num_groups}, eps={eps})"

    def visit_MaxPool2D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for MaxPool2D.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch max_pool2d expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        kernel_size = attrs.get("kernel_size", 2)
        stride = attrs.get("stride", kernel_size)
        return f"torch.nn.functional.max_pool2d({input_vars[0]}, kernel_size={kernel_size}, stride={stride})"

    def visit_AvgPool2D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for AvgPool2D.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch avg_pool2d expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        kernel_size = attrs.get("kernel_size", 2)
        stride = attrs.get("stride", kernel_size)
        return f"torch.nn.functional.avg_pool2d({input_vars[0]}, kernel_size={kernel_size}, stride={stride})"

    def visit_AdaptiveAvgPool2D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for AdaptiveAvgPool2D.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch adaptive_avg_pool2d expression.
        """
        attrs = getattr(node, "attributes", {}) or {}
        output_size = attrs.get("output_size", [1, 1])
        return f"torch.nn.functional.adaptive_avg_pool2d({input_vars[0]}, output_size={output_size})"

    def visit_Linear(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate PyTorch code for Linear projection.

        Args:
            node (object): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs (object): Additional operation attributes.

        Returns:
            str: Generated PyTorch linear expression.
        """
        bias = input_vars[2] if len(input_vars) > 2 else "None"
        return f"torch.nn.functional.linear({input_vars[0]}, {input_vars[1]}, bias={bias})"
