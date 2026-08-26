# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for kernels.py."""

from dataclasses import dataclass
from typing import Optional

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@register_op("CudaKernel")
class CudaKernelOp(OpDef):
    """Cuda kernel operation."""

    op_name = "CudaKernel"

    def infer_shape(self, *args, **kwargs):
        """Infer shape for PrecompiledCudaKernel.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("MetalKernel")
class MetalKernelOp(OpDef):
    """Metal kernel operation."""

    op_name = "MetalKernel"

    def infer_shape(self, *args, **kwargs):
        """Infer shape for MetalKernel.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("PrecompiledCudaKernel")
class PrecompiledCudaKernelOp(OpDef):
    """Precompiled Cuda kernel operation."""

    op_name = "PrecompiledCudaKernel"

    def infer_shape(self, *args, **kwargs):
        """Infer shape for PrecompiledCudaKernel.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@dataclass
class KernelLaunchConfig:
    """Configuration for kernel launch."""

    grid: tuple[int, ...]
    block: tuple[int, ...]
    name: str = "custom_kernel"


@dataclass
class KernelContext:
    """Provide context for kernel execution."""

    op_type: str
    output_shapes: list[tuple[int, ...]]
    output_dtypes: list[DType]
    launch_config: KernelLaunchConfig


def _eager_custom_kernel(
    inputs: list[Tensor],
    ctx: KernelContext,
) -> list[Tensor]:
    """Execute a custom kernel eagerly.

    Args:
        inputs (list[Tensor]): The input tensors.
        ctx (KernelContext): The kernel context.

    Returns:
        list[Tensor]: The output tensors.
    """
    data = get_active_backend().execute_op(
        ctx.op_type,
        ctx.code_or_binary,
        [getattr(t, "data", t) for t in inputs],
        output_shapes=ctx.output_shapes,
        output_dtypes=[getattr(dt, "value", dt) for dt in ctx.output_dtypes],
        grid=ctx.launch_config.grid,
        block=ctx.launch_config.block,
        name=ctx.launch_config.name,
    )
    device = inputs[0].device if inputs else config.default_device
    return [Tensor(d, TensorConfig(s, dt, device)) for d, s, dt in zip(data, ctx.output_shapes, ctx.output_dtypes)]


def cuda_kernel(
    source: str,
    inputs: list[Tensor],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[DType],
    launch_config: Optional[KernelLaunchConfig] = None,
) -> list[Tensor]:
    """Injects and compiles an inline CUDA kernel.

    Args:
        source (str): The source parameter.
        inputs (list): The inputs parameter.
        output_shapes (list): The output_shapes parameter.
        output_dtypes (list): The output_dtypes parameter.
        launch_config (Optional): The launch_config parameter.

    Returns:
        list: Result.
    """
    conf = launch_config if launch_config is not None else KernelLaunchConfig((1,), (1,), "custom_kernel")
    if config.eager_mode:
        return _eager_custom_kernel(inputs, KernelContext("CudaKernel", source, output_shapes, output_dtypes, conf))

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs = {
            "source": source,
            "grid": conf.grid,
            "block": conf.block,
            "name": conf.name,
            "output_idx": i,
            "num_outputs": len(output_shapes),
        }
        outputs.append(_emit_shape_node("CudaKernel", inputs, attrs, shape, dtype))
    return outputs


def metal_kernel(
    source: str,
    inputs: list[Tensor],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[DType],
    launch_config: Optional[KernelLaunchConfig] = None,
) -> list[Tensor]:
    """Injects and compiles an inline Metal kernel.

    Args:
        source (str): The source parameter.
        inputs (list): The inputs parameter.
        output_shapes (list): The output_shapes parameter.
        output_dtypes (list): The output_dtypes parameter.
        launch_config (Optional): The launch_config parameter.

    Returns:
        list: Result.
    """
    conf = launch_config if launch_config is not None else KernelLaunchConfig((1,), (1,), "custom_kernel")
    if config.eager_mode:
        return _eager_custom_kernel(inputs, KernelContext("MetalKernel", source, output_shapes, output_dtypes, conf))

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs = {
            "source": source,
            "grid": conf.grid,
            "block": conf.block,
            "name": conf.name,
            "output_idx": i,
            "num_outputs": len(output_shapes),
        }
        outputs.append(_emit_shape_node("MetalKernel", inputs, attrs, shape, dtype))
    return outputs


def precompiled_cuda_kernel(
    binary: bytes,
    inputs: list[Tensor],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[DType],
    launch_config: Optional[KernelLaunchConfig] = None,
) -> list[Tensor]:
    """Injects and executes a precompiled CUDA binary (PTX/CUBIN).

    Args:
        binary (bytes): The binary parameter.
        inputs (list): The inputs parameter.
        output_shapes (list): The output_shapes parameter.
        output_dtypes (list): The output_dtypes parameter.
        launch_config (Optional): The launch_config parameter.

    Returns:
        list: Result.
    """
    conf = launch_config if launch_config is not None else KernelLaunchConfig((1,), (1,), "custom_kernel")
    if config.eager_mode:
        return _eager_custom_kernel(inputs, KernelContext("PrecompiledCudaKernel", binary, output_shapes, output_dtypes, conf))

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs = {
            "binary": binary,
            "grid": conf.grid,
            "block": conf.block,
            "name": conf.name,
            "output_idx": i,
            "num_outputs": len(output_shapes),
        }
        outputs.append(_emit_shape_node("PrecompiledCudaKernel", inputs, attrs, shape, dtype))
    return outputs
