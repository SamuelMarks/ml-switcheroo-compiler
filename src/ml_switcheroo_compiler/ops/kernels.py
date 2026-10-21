"""Implementation of kernels operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.kernels`)."""

from __future__ import annotations

import pydantic

from ml_switcheroo_compiler.backends.registry import as_str, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@register_op("CudaKernel")
class CudaKernelOp(OpDef):
    """Cuda kernel operation."""

    op_name = "CudaKernel"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape for PrecompiledCudaKernel.

        Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("MetalKernel")
class MetalKernelOp(OpDef):
    """Metal kernel operation."""

    op_name = "MetalKernel"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape for MetalKernel.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("PrecompiledCudaKernel")
class PrecompiledCudaKernelOp(OpDef):
    """Precompiled Cuda kernel operation."""

    op_name = "PrecompiledCudaKernel"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape for PrecompiledCudaKernel.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


class KernelLaunchConfig(pydantic.BaseModel):
    """Configuration for kernel launch."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    grid: tuple[int, ...]
    block: tuple[int, ...]
    name: str = "custom_kernel"


class KernelContext(pydantic.BaseModel):
    """Provide context for kernel execution."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    op_type: str
    code_or_binary: str
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
    return [Tensor(d, TensorConfig(s, dt, device)) for d, s, dt in zip(getattr(data, "__iter__")(), ctx.output_shapes, ctx.output_dtypes)]


def cuda_kernel(
    source: str,
    inputs: list[Tensor],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[DType],
    launch_config: KernelLaunchConfig | None = None,
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
    conf = launch_config if launch_config is not None else KernelLaunchConfig(grid=(1,), block=(1,), name="custom_kernel")
    if config.eager_mode:
        return _eager_custom_kernel(inputs, KernelContext(op_type="CudaKernel", code_or_binary=source, output_shapes=output_shapes, output_dtypes=output_dtypes, launch_config=conf))

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs: dict[str, TensorOrScalar] = {
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
    launch_config: KernelLaunchConfig | None = None,
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
    conf = launch_config if launch_config is not None else KernelLaunchConfig(grid=(1,), block=(1,), name="custom_kernel")
    if config.eager_mode:
        return _eager_custom_kernel(inputs, KernelContext(op_type="MetalKernel", code_or_binary=source, output_shapes=output_shapes, output_dtypes=output_dtypes, launch_config=conf))

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs: dict[str, TensorOrScalar] = {
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
    launch_config: KernelLaunchConfig | None = None,
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
    conf = launch_config if launch_config is not None else KernelLaunchConfig(grid=(1,), block=(1,), name="custom_kernel")
    if config.eager_mode:
        return _eager_custom_kernel(inputs, KernelContext(op_type="PrecompiledCudaKernel", code_or_binary=as_str(binary), output_shapes=output_shapes, output_dtypes=output_dtypes, launch_config=conf))

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs: dict[str, TensorOrScalar] = {
            "binary": binary,
            "grid": conf.grid,
            "block": conf.block,
            "name": conf.name,
            "output_idx": i,
            "num_outputs": len(output_shapes),
        }
        outputs.append(_emit_shape_node("PrecompiledCudaKernel", inputs, attrs, shape, dtype))
    return outputs
