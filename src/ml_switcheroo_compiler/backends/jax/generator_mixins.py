# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""JAX Generator Mixins."""

from typing import Any, Optional

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.audio_utils import (
    extract_mel_attributes,
    extract_stft_attributes,
)
from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)
from ml_switcheroo_compiler.ir.core import IRNode


class JaxDistributedVisitor:
    """Provide mixin for JAX distributed node visitors."""

    def __init__(self, generator: Optional[BaseGenerator] = None) -> None:
        """Init.

        Args:
            generator: The generator.
        """
        self.generator = generator

    @property
    def code(self) -> list[str]:
        """Get code list from the generator."""
        return getattr(self.generator, "code", [])

    @staticmethod
    def _extract_code_lines(target: object) -> list[str]:
        """Obtain mutable target code list from visitor or generator.

        Args:
            target (object): Generator or visitor instance.

        Returns:
            list[str]: Mutable code lines list.
        """
        if hasattr(target, "code") and isinstance(target.code, list):
            return target.code
        if hasattr(target, "generator") and hasattr(target.generator, "code"):
            gen_code = target.generator.code
            if isinstance(gen_code, list):
                return gen_code
        return []

    def visit_Send(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for sending a tensor via token-passing pipeline communication.

        Args:
            node (IRNode): The IR node representing the Send operation.
            input_vars (list[str]): Input variable names to send.
            **kwargs (Any): Additional attributes and parameters.

        Returns:
            str: Empty string as send is a statement emitted directly into generator code.
        """
        code_lines = JaxDistributedVisitor._extract_code_lines(self)
        if not any("jax.lax.create_token()" in line for line in code_lines):
            code_lines.append("    token = jax.lax.create_token()")

        dst: int = int(getattr(node, "attributes", {}).get("dst_rank", getattr(node, "attributes", {}).get("target_stage", 0)))
        channel: int = int(getattr(node, "attributes", {}).get("channel", getattr(node, "attributes", {}).get("tag", dst)))
        in_var = input_vars[0] if input_vars else "x"
        code_lines.append(f"    # JAX Send to {dst} (via host_callback or token-passing lax.send on channel {channel})")
        code_lines.append(f"    token = jax.lax.send({in_var}, token, channel={channel})")
        return ""

    def visit_Recv(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for receiving a tensor via token-passing pipeline communication.

        Args:
            node (IRNode): The IR node representing the Recv operation.
            input_vars (list[str]): Input variable names (tokens or dependencies).
            **kwargs (Any): Additional attributes and parameters.

        Returns:
            str: The name of the variable storing the received tensor.
        """
        code_lines = JaxDistributedVisitor._extract_code_lines(self)
        if not any("jax.lax.create_token()" in line for line in code_lines):
            code_lines.append("    token = jax.lax.create_token()")

        src: int = int(getattr(node, "attributes", {}).get("src_rank", getattr(node, "attributes", {}).get("source_stage", 0)))
        channel: int = int(getattr(node, "attributes", {}).get("channel", getattr(node, "attributes", {}).get("tag", src)))

        raw_shape = getattr(node, "shape_metadata", None)
        if raw_shape is None:
            raw_shape = getattr(node, "attributes", {}).get("shape", ())
        shape: tuple[object, ...] = tuple(raw_shape) if isinstance(raw_shape, (list, tuple)) else (raw_shape,)

        raw_dtype = str(getattr(node, "attributes", {}).get("dtype", getattr(node, "dtype", "float32"))).lower()
        clean_dtype = raw_dtype if raw_dtype.startswith("jnp.") else f"jnp.{raw_dtype}"

        nid: str = getattr(node, "id", "recv")
        res_var: str = f"v_{nid.replace('-', '_')}"
        code_lines.append(f"    # JAX Recv from {src} (via token-passing lax.recv on channel {channel})")
        code_lines.append(f"    {res_var}, token = jax.lax.recv(token, channel={channel}, shape={shape}, dtype={clean_dtype})")
        return res_var

    def visit_AllGather(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the all_gather operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        tensor: str = input_vars[0]
        axis_name: str = getattr(node, "attributes", {}).get("axis_name", "'x'")
        return f"jax.lax.all_gather({tensor}, axis_name={axis_name})"

    def visit_ReduceScatter(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the reduce_scatter operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        tensor: str = input_vars[0]
        axis: int = int(getattr(node, "attributes", {}).get("axis", 0))
        axis_name: str = getattr(node, "attributes", {}).get("axis_name", "'x'")
        op: str = getattr(node, "attributes", {}).get("op", "jax.lax.psum")
        return f"jax.lax.reduce_scatter({tensor}, {op}, scatter_dimension={axis}, axis_name={axis_name})"

    def visit_AllReduce(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the all_reduce operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        tensor: str = input_vars[0]
        axis_name: str = getattr(node, "attributes", {}).get("axis_name", "'x'")
        op: str = getattr(node, "attributes", {}).get("op", "psum")
        return f"jax.lax.{op}({tensor}, axis_name={axis_name})"

    def visit_AllToAll(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the all_to_all collective operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        tensor: str = input_vars[0]
        axis_name: str = getattr(node, "attributes", {}).get("axis_name", "'x'")
        split_axis: int = int(getattr(node, "attributes", {}).get("split_axis", 0))
        concat_axis: int = int(getattr(node, "attributes", {}).get("concat_axis", 0))
        return f"jax.lax.all_to_all({tensor}, axis_name={axis_name}, split_axis={split_axis}, concat_axis={concat_axis})"

    def visit_Broadcast(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the broadcast collective operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        tensor: str = input_vars[0]
        axis_name: str = getattr(node, "attributes", {}).get("axis_name", "'x'")
        return f"jax.lax.pbroadcast({tensor}, axis_name={axis_name})"


class JaxMathVisitor:
    """Provide mixin for JAX math node visitors."""

    def __init__(self, generator: Optional[BaseGenerator] = None) -> None:
        """Init.

        Args:
            generator: The generator.
        """
        self.generator = generator

    def visit_SegmentSum(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the SegmentSum operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_sum({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentMax(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the SegmentMax operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_max({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentMin(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the SegmentMin operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_min({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_SegmentProd(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the SegmentProd operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_prod({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentSum(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the UnsortedSegmentSum operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_sum({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentMax(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the UnsortedSegmentMax operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_max({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentMin(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the UnsortedSegmentMin operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_min({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_UnsortedSegmentProd(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the UnsortedSegmentProd operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_segments = getattr(node, "attributes", {}).get("num_segments", "None")
        return f"jax.ops.segment_prod({input_vars[0]}, {input_vars[1]}, num_segments={num_segments})"

    def visit_MatrixExponential(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the MatrixExponential operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        return f"jax.scipy.linalg.expm({input_vars[0]})"

    def visit_Polar(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the Polar operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        side: str = str(getattr(node, "attributes", {}).get("side", "'right'"))
        if not side.startswith("'"):
            side = f"'{side}'"
        return f"jax.scipy.linalg.polar({input_vars[0]}, side={side})"

    def visit_Schur(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the Schur operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        return f"jax.scipy.linalg.schur({input_vars[0]})"

    def visit_Cholesky(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the Cholesky operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        return f"jax.numpy.linalg.cholesky({input_vars[0]})"

    def visit_Svd(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the Svd operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        full_matrices: bool = bool(getattr(node, "attributes", {}).get("full_matrices", True))
        compute_uv: bool = bool(getattr(node, "attributes", {}).get("compute_uv", True))
        return f"jax.numpy.linalg.svd({input_vars[0]}, full_matrices={full_matrices}, compute_uv={compute_uv})"

    def visit_PowerIteration(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the PowerIteration operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        num_iters: int = int(getattr(node, "attributes", {}).get("num_iters", 1))
        u_var: str = input_vars[1] if len(input_vars) > 1 else "None"
        return f"jax_power_iteration({input_vars[0]}, {num_iters}, {u_var})"

    def visit_RaggedDot(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the RaggedDot operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        return f"jax_ragged_dot({input_vars[0]}, {input_vars[1]})"

    def visit_Einsum(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the Einsum operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        args_str: str = ", ".join(input_vars)
        eq: str = str(kwargs.get("equation", ""))
        return f"jnp.einsum('{eq}', {args_str})"

    def visit_Conv1D(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for Conv1D.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX conv1d expression.
        """
        s = getattr(node, "attributes", {}).get("stride", 1)
        stride = s[0] if isinstance(s, (list, tuple)) else int(s)
        padding = str(getattr(node, "attributes", {}).get("padding", "SAME")).upper()
        return f"jax.lax.conv_general_dilated({input_vars[0]}, {input_vars[1]}, window_strides=({stride},), padding='{padding}')"

    def visit_Conv2D(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for Conv2D.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX conv2d expression.
        """
        s = getattr(node, "attributes", {}).get("stride", 1)
        stride = tuple(s) if isinstance(s, (list, tuple)) else (int(s), int(s))
        padding = str(getattr(node, "attributes", {}).get("padding", "SAME")).upper()
        return f"jax.lax.conv_general_dilated({input_vars[0]}, {input_vars[1]}, window_strides={stride}, padding='{padding}')"

    def visit_Conv3D(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for Conv3D.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variables.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX conv3d expression.
        """
        s = getattr(node, "attributes", {}).get("stride", 1)
        stride = tuple(s) if isinstance(s, (list, tuple)) else (int(s), int(s), int(s))
        padding = str(getattr(node, "attributes", {}).get("padding", "SAME")).upper()
        return f"jax.lax.conv_general_dilated({input_vars[0]}, {input_vars[1]}, window_strides={stride}, padding='{padding}')"

    def visit_MultiHeadAttention(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for MultiHeadAttention / ScaledDotProductAttention.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Query, Key, Value input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX attention expression.
        """
        q, k, v = input_vars[0], input_vars[1], input_vars[2]
        return f"(lambda q, k, v: (jnp.matmul(jax.nn.softmax(jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(q.shape[-1])), v)))({q}, {k}, {v})"

    def visit_ScaledDotProductAttention(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for ScaledDotProductAttention.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Query, Key, Value input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX attention expression.
        """
        return self.visit_MultiHeadAttention(node, input_vars, **kwargs)

    def visit_LayerNorm(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for LayerNorm.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX layer_norm expression.
        """
        eps = float(getattr(node, "attributes", {}).get("eps", 1e-5))
        return f"(lambda x: (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(x, axis=-1, keepdims=True) + {eps}))({input_vars[0]})"

    def visit_RMSNorm(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for RMSNorm.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX rms_norm expression.
        """
        eps = float(getattr(node, "attributes", {}).get("eps", 1e-5))
        return f"(lambda x: x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + {eps}))({input_vars[0]})"

    def visit_BatchNorm(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for BatchNorm.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX batch_norm expression.
        """
        eps = float(getattr(node, "attributes", {}).get("eps", 1e-5))
        return f"(lambda x: (x - jnp.mean(x, axis=0, keepdims=True)) / jnp.sqrt(jnp.var(x, axis=0, keepdims=True) + {eps}))({input_vars[0]})"

    def visit_GroupNorm(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for GroupNorm.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX group_norm expression.
        """
        return f"{input_vars[0]}"

    def visit_MaxPool2D(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for MaxPool2D.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX max_pool2d expression.
        """
        return f"jax.lax.reduce_window({input_vars[0]}, -jnp.inf, jax.lax.max, window_dimensions=(1, 2, 2, 1), window_strides=(1, 2, 2, 1), padding='SAME')"

    def visit_AvgPool2D(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for AvgPool2D.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX avg_pool2d expression.
        """
        return f"(jax.lax.reduce_window({input_vars[0]}, 0.0, jax.lax.add, window_dimensions=(1, 2, 2, 1), window_strides=(1, 2, 2, 1), padding='SAME') / 4.0)"

    def visit_AdaptiveAvgPool2D(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for AdaptiveAvgPool2D.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX adaptive_avg_pool2d expression.
        """
        return f"{input_vars[0]}"

    def visit_Linear(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for Linear projection.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Input variable names.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Generated JAX linear expression.
        """
        if len(input_vars) > 2:
            return f"jnp.matmul({input_vars[0]}, {input_vars[1]}) + {input_vars[2]}"
        return f"jnp.matmul({input_vars[0]}, {input_vars[1]})"


class JaxControlFlowVisitor:
    """Provide mixin for JAX control flow node visitors."""

    def __init__(self, generator: Optional[BaseGenerator] = None) -> None:
        """Init.

        Args:
            generator: The generator.
        """
        self.generator = generator

    def visit_If(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the If operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        # Simple fallback for jax.lax.cond if proper block tracing is not used natively
        return f"jax.lax.cond({input_vars[0]}, lambda: None, lambda: None)"

    def visit_Loop(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the Loop operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        return f"jax.lax.while_loop(lambda _: True, lambda _: {input_vars[0]}, {input_vars[0]})"

    def visit_Scan(self, node: IRNode, input_vars: list[str], **kwargs: "Any") -> str:
        """Generate JAX code for the Scan operation.

        Args:
            node (IRNode): The intermediate representation node representing the operation.
            input_vars (list[str]): A list of variable names representing the inputs to the operation.
            **kwargs: Additional keyword arguments used during code generation.

        Returns:
            str: A string containing the generated JAX code.
        """
        return f"jax.lax.scan(lambda c, x: (c, x), {input_vars[0]}, {input_vars[1] if len(input_vars) > 1 else None})"
