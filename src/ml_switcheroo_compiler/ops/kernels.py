"""Module docstring."""

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

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape for CudaKernel.

        Args:
            *args (object): Argument *args.
            **kwargs (object): Argument **kwargs.

            launch_config (object): The launch config.\

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("MetalKernel")
class MetalKernelOp(OpDef):
    """Metal kernel operation."""

    op_name = "MetalKernel"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape for MetalKernel.

        Args:
            *args (object): Argument *args.
            **kwargs (object): Argument **kwargs.

            launch_config (object): The launch config.\

        Returns:
            object: The inferred shape.
        """
        return ()


@register_op("PrecompiledCudaKernel")
class PrecompiledCudaKernelOp(OpDef):
    """Precompiled Cuda kernel operation."""

    op_name = "PrecompiledCudaKernel"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape for PrecompiledCudaKernel.

        Args:
            *args (object): Argument *args.
            **kwargs (object): Argument **kwargs.

            launch_config (object): The launch config.\

        Returns:
            object: The inferred shape.
        """
        return ()


@dataclass
class KernelLaunchConfig:
    """Configuration for kernel launch."""

    grid: tuple[int, ...]
    block: tuple[int, ...]
    name: str = "custom_kernel"


def cuda_kernel(
    source: str,
    inputs: list[Tensor],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[DType],
    launch_config: Optional[KernelLaunchConfig] = None,
) -> list[Tensor]:
    """Injects and compiles an inline CUDA kernel.

    Args:
        source (str): The CUDA C++ source code.
        inputs (list[Tensor]): Input tensors.
        output_shapes (list[tuple[int, ...]]): Expected output shapes.
        output_dtypes (list[DType]): Expected output data types.
        config: Kernel configuration.

        launch_config (object): The launch config.\

    Returns:
        list[Tensor]: Output tensors.
    """
    conf = launch_config if launch_config is not None else KernelLaunchConfig((1,), (1,), "custom_kernel")
    grid = conf.grid
    block = conf.block
    name = conf.name
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "CudaKernel",
            source,
            [getattr(t, "data", t) for t in inputs],
            output_shapes=output_shapes,
            output_dtypes=[getattr(dt, "value", dt) for dt in output_dtypes],
            grid=grid,
            block=block,
            name=name,
        )
        return [Tensor(d, TensorConfig(s, dt, inputs[0].device if inputs else config.default_device)) for d, s, dt in zip(data, output_shapes, output_dtypes)]

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs = {
            "source": source,
            "grid": grid,
            "block": block,
            "name": name,
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
        source (str): The Metal source code.
        inputs (list[Tensor]): Input tensors.
        output_shapes (list[tuple[int, ...]]): Expected output shapes.
        output_dtypes (list[DType]): Expected output data types.
        config: Kernel configuration.

        launch_config (object): The launch config.\

    Returns:
        list[Tensor]: Output tensors.
    """
    conf = launch_config if launch_config is not None else KernelLaunchConfig((1,), (1,), "custom_kernel")
    grid = conf.grid
    block = conf.block
    name = conf.name
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "MetalKernel",
            source,
            [getattr(t, "data", t) for t in inputs],
            output_shapes=output_shapes,
            output_dtypes=[getattr(dt, "value", dt) for dt in output_dtypes],
            grid=grid,
            block=block,
            name=name,
        )
        return [Tensor(d, TensorConfig(s, dt, inputs[0].device if inputs else config.default_device)) for d, s, dt in zip(data, output_shapes, output_dtypes)]

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs = {
            "source": source,
            "grid": grid,
            "block": block,
            "name": name,
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
        binary (bytes): The precompiled binary.
        inputs (list[Tensor]): Input tensors.
        output_shapes (list[tuple[int, ...]]): Expected output shapes.
        output_dtypes (list[DType]): Expected output data types.
        config: Kernel configuration.

        launch_config (object): The launch config.\

    Returns:
        list[Tensor]: Output tensors.
    """
    conf = launch_config if launch_config is not None else KernelLaunchConfig((1,), (1,), "custom_kernel")
    grid = conf.grid
    block = conf.block
    name = conf.name
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "PrecompiledCudaKernel",
            binary,
            [getattr(t, "data", t) for t in inputs],
            output_shapes=output_shapes,
            output_dtypes=[getattr(dt, "value", dt) for dt in output_dtypes],
            grid=grid,
            block=block,
            name=name,
        )
        return [Tensor(d, TensorConfig(s, dt, inputs[0].device if inputs else config.default_device)) for d, s, dt in zip(data, output_shapes, output_dtypes)]

    outputs = []
    for i, (shape, dtype) in enumerate(zip(output_shapes, output_dtypes)):
        attrs = {
            "binary": binary,
            "grid": grid,
            "block": block,
            "name": name,
            "output_idx": i,
            "num_outputs": len(output_shapes),
        }
        outputs.append(_emit_shape_node("PrecompiledCudaKernel", inputs, attrs, shape, dtype))
    return outputs
