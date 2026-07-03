"""Tests for custom hardware kernels."""

from unittest.mock import patch

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.kernels import (
    CudaKernelOp,
    KernelLaunchConfig,
    MetalKernelOp,
    PrecompiledCudaKernelOp,
    cuda_kernel,
    metal_kernel,
    precompiled_cuda_kernel,
)


@patch("ml_switcheroo_compiler.ops.kernels._emit_shape_node")
def test_cuda_kernel_tracing(mock_emit: object) -> None:
    """Test cuda_kernel tracing.

    Args:
        mock_emit (object): Mocked _emit_shape_node function.
    """
    mock_emit.return_value = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    t1 = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    with ConfigContext(eager_mode=False):
        outs = cuda_kernel(
            source="void kernel() {}",
            inputs=[t1],
            output_shapes=[(2, 2)],
            output_dtypes=[DType.Float32],
            launch_config=KernelLaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), name="test"),
        )
        assert len(outs) == 1
        assert mock_emit.called


@patch("ml_switcheroo_compiler.ops.kernels._emit_shape_node")
def test_metal_kernel_tracing(mock_emit: object) -> None:
    """Test metal_kernel tracing.

    Args:
        mock_emit (object): Mocked _emit_shape_node function.
    """
    mock_emit.return_value = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    t1 = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    with ConfigContext(eager_mode=False):
        outs = metal_kernel(
            source="kernel void foo() {}",
            inputs=[t1],
            output_shapes=[(2, 2)],
            output_dtypes=[DType.Float32],
            launch_config=KernelLaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), name="test"),
        )
        assert len(outs) == 1
        assert mock_emit.called


@patch("ml_switcheroo_compiler.ops.kernels._emit_shape_node")
def test_precompiled_cuda_kernel_tracing(mock_emit: object) -> None:
    """Test precompiled_cuda_kernel tracing.

    Args:
        mock_emit (object): Mocked _emit_shape_node function.
    """
    mock_emit.return_value = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    t1 = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    with ConfigContext(eager_mode=False):
        outs = precompiled_cuda_kernel(
            binary=b"dummy_binary",
            inputs=[t1],
            output_shapes=[(2, 2)],
            output_dtypes=[DType.Float32],
            launch_config=KernelLaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), name="test"),
        )
        assert len(outs) == 1
        assert mock_emit.called


@patch("ml_switcheroo_compiler.backends.numpy.generator.NumpyGenerator.execute_op")
def test_cuda_kernel_eager(mock_execute_op: object) -> None:
    """Test cuda_kernel eager execution.

    Args:
        mock_execute_op (object): Mocked execute_op function.
    """
    mock_execute_op.return_value = [None]
    t1 = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    with ConfigContext(eager_mode=True):
        outs = cuda_kernel(
            source="void kernel() {}",
            inputs=[t1],
            output_shapes=[(2, 2)],
            output_dtypes=[DType.Float32],
            launch_config=KernelLaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), name="test"),
        )
        assert len(outs) == 1
        assert mock_execute_op.called


@patch("ml_switcheroo_compiler.backends.numpy.generator.NumpyGenerator.execute_op")
def test_metal_kernel_eager(mock_execute_op: object) -> None:
    """Test metal_kernel eager execution.

    Args:
        mock_execute_op (object): Mocked execute_op function.
    """
    mock_execute_op.return_value = [None]
    t1 = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    with ConfigContext(eager_mode=True):
        outs = metal_kernel(
            source="kernel void foo() {}",
            inputs=[t1],
            output_shapes=[(2, 2)],
            output_dtypes=[DType.Float32],
            launch_config=KernelLaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), name="test"),
        )
        assert len(outs) == 1
        assert mock_execute_op.called


@patch("ml_switcheroo_compiler.backends.numpy.generator.NumpyGenerator.execute_op")
def test_precompiled_cuda_kernel_eager(mock_execute_op: object) -> None:
    """Test precompiled_cuda_kernel eager execution.

    Args:
        mock_execute_op (object): Mocked execute_op function.
    """
    mock_execute_op.return_value = [None]
    t1 = Tensor(None, TensorConfig((2, 2), DType.Float32, "cpu"))
    with ConfigContext(eager_mode=True):
        outs = precompiled_cuda_kernel(
            binary=b"dummy_binary",
            inputs=[t1],
            output_shapes=[(2, 2)],
            output_dtypes=[DType.Float32],
            launch_config=KernelLaunchConfig(grid=(1, 1, 1), block=(1, 1, 1), name="test"),
        )
        assert len(outs) == 1
        assert mock_execute_op.called


def test_op_defs() -> None:
    """Test custom kernel OpDefs."""
    op1 = CudaKernelOp()
    assert op1.infer_shape() == ()
    op2 = MetalKernelOp()
    assert op2.infer_shape() == ()
    op3 = PrecompiledCudaKernelOp()
    assert op3.infer_shape() == ()
