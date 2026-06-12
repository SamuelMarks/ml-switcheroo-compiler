# Detailed MLX Compiler Implementation Plan

This document serves as the exhaustive roadmap for everything `ml-switcheroo-compiler` must implement to support `zero-mlx`. The goal is to guarantee that the `zero-mlx` shim achieves 100% semantic and syntactic test-passing parity with the official Apple `mlx` library when running on the `ml-switcheroo` backend.

---

## 1. Core IR & Execution Engine (Lazy Evaluation & Streams)
Apple MLX differs fundamentally from JAX and PyTorch by natively using a lazy computation graph with implicit evaluation.

- [ ] **Lazy Graph Evaluation:** Ensure `ProxyTensor` correctly delays execution. The IR compiler must support triggering compilation and execution *only* when `.eval()` is explicitly called or when host-data is requested (e.g., printing or converting to numpy).
- [ ] **Stream Contexts & Device Management:**
  - [ ] Implement support for multi-stream execution logic (`mlx.core.StreamContext`, `mlx.core.Stream`, `mlx.core.Device`).
  - [ ] The IR must allow assigning Logical Nodes to specific execution streams or devices (e.g., CPU vs GPU).
- [ ] **Cache Management:** Expose an API for the memory allocator to safely free unused buffers manually (`mlx.core.clear_cache()`).
- [ ] **Function Exporter:** Support dumping the compiler's `LogicalGraph` trace into an exportable format compatible with `mlx.core.FunctionExporter`.

## 2. Advanced Automatic Differentiation & Transforms
MLX provides a robust set of functional transformations.

- [ ] **Forward-Mode AD (JVP):** Extend `compiler.grad` (which currently supports reverse-mode VJPs) to support Jacobian-Vector Products (`mlx.core.jvp`) and forward-mode sensitivities.
- [ ] **Value and Grad:** Ensure `value_and_grad` is highly optimized to avoid duplicate forward passes when calculating both the primal and the gradient.
- [ ] **Custom Vectorization (VMAP):**
  - [ ] Ensure `vmap` handles complex shapes and multi-axis vectorization.
  - [ ] Support vectorization of stateful operations (like random generation or stream-dependent ops).
- [ ] **Custom Gradients:** Ensure `custom_vjp` and equivalent IR node overrides allow defining custom gradient functions natively inside the trace.

---

## 3. Specific Target Implementation Matrix
The following tables list the exact primitives that the compiler backend must support to enable `zero-mlx` to pass all unit tests.

### Core Operations (mlx.core)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `mlx.core.ArrayAt` | `(...)` | A helper object to apply updates at specific indices. | Needs mapping & tests. |
| [ ] | `mlx.core.ArrayIterator` | `(...)` | A helper object to iterate over the 1st dimension of an array. | Needs mapping & tests. |
| [ ] | `mlx.core.ArrayLike` | `(...)` | Any Python object which has an __mlx__array__ method that | Needs mapping & tests. |
| [x] | `mlx.core.DeviceType` | `(value, names=None, *, module=None, qualname=None, type=None, start=1)` | No documentation available. | Needs mapping & tests. |
| [ ] | `mlx.core.Dtype` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [ ] | `mlx.core.DtypeCategory` | `(value, names=None, *, module=None, qualname=None, type=None, start=1)` | Type to hold categories of :class:dtypes <Dtype>. | Needs mapping & tests. |
| [ ] | `mlx.core.FunctionExporter` | `(...)` | A context managing class for exporting multiple traces of the same | Needs mapping & tests. |
| [ ] | `mlx.core.StreamContext` | `(...)` | A context manager for setting the current device and stream. | Needs mapping & tests. |
| [ ] | `mlx.core.arccosh` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse hyperbolic cosine. | Needs mapping & tests. |
| [ ] | `mlx.core.arctan2` | `(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse tangent of the ratio of two arrays. | Needs mapping & tests. |
| [ ] | `mlx.core.bitwise_invert` | `(a: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise bitwise inverse. | Needs mapping & tests. |
| [ ] | `mlx.core.clear_cache` | `() -> None` | Clear the memory cache. | Needs mapping & tests. |
| [ ] | `mlx.core.complexfloating` | `(...)` | No documentation available. | Needs mapping & tests. |
| [ ] | `mlx.core.concat` | `(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array` | See :func:concatenate. | Needs mapping & tests. |
| [x] | `mlx.core.conj` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> array` | Return the elementwise complex conjugate of the input. | Needs mapping & tests. |
| [ ] | `mlx.core.contiguous` | `(a: array, /, allow_col_major: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Force an array to be row contiguous. Copy if necessary. | Needs mapping & tests. |
| [x] | `mlx.core.conv1d` | `(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array` | 1D convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv2d` | `(input: array, weight: array, /, stride: Union[int, tuple[int, int]] = 1, padding: Union[int, tuple[int, int]] = 0, dilation: Union[int, tuple[int, int]] = 1, groups: int = 1, *, stream: Union[None...` | 2D convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv3d` | `(input: array, weight: array, /, stride: Union[int, tuple[int, int, int]] = 1, padding: Union[int, tuple[int, int, int]] = 0, dilation: Union[int, tuple[int, int, int]] = 1, groups: int = 1, *, str...` | 3D convolution over an input with several channels | Needs mapping & tests. |
| [ ] | `mlx.core.conv_general` | `(input: array, weight: array, /, stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int], tuple[Sequence[int], Sequence[int]]] = 0, kernel_dilation: Union[int, Sequence[int]] = 1, ...` | General convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv_transpose1d` | `(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array` | 1D transposed convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv_transpose2d` | `(input: array, weight: array, /, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, output_padding: Union[int, Tuple[int, ...` | 2D transposed convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv_transpose3d` | `(input: array, weight: array, /, stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int]] = 0, dilation: Union[int, Tuple[int, int, int]] = 1, output_padding: Union[i...` | 3D transposed convolution over an input with several channels | Needs mapping & tests. |
| [ ] | `mlx.core.convolve` | `(a: array, v: array, /, mode: str = "full", *, stream: Union[None, Stream, Device] = None) -> array` | The discrete convolution of 1D arrays. | Needs mapping & tests. |
| [ ] | `mlx.core.dequantize` | `(w: array, /, scales: array, biases: Optional[array] = None, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = None) -> array` | Dequantize the matrix w using quantization parameters. | Needs mapping & tests. |
| [ ] | `mlx.core.distributed` | `(...)` | mlx.core.distributed: Communication operations | Needs mapping & tests. |
| [x] | `mlx.core.einsum` | `(subscripts: str, *operands, stream: Union[None, Stream, Device] = None) -> array` | Perform the Einstein summation convention on the operands. | Needs mapping & tests. |
| [ ] | `mlx.core.einsum_path` | `(subscripts: str, *operands)` | Compute the contraction order for the given Einstein summation. | Needs mapping & tests. |
| [ ] | `mlx.core.export_function` | `(arg0: object, fun: collections.abc.Callable, *args, shapeless: bool = False, **kwargs) -> None` | Export an MLX function. | Needs mapping & tests. |
| [ ] | `mlx.core.exporter` | `(file: str, fun: collections.abc.Callable, *, shapeless: bool = False) -> mlx.core.FunctionExporter` | Make a callable object to export multiple traces of a function to a file. | Needs mapping & tests. |
| [ ] | `mlx.core.fast` | `(...)` | mlx.core.fast: fast operations | Needs mapping & tests. |
| [x] | `mlx.core.flatten` | `(a: array, /, start_axis: int = 0, end_axis: int = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Flatten an array. | Needs mapping & tests. |
| [ ] | `mlx.core.floating` | `(...)` | No documentation available. | Needs mapping & tests. |
| [ ] | `mlx.core.gather_qmm` | `(x: array, w: array, /, scales: array, biases: Optional[array] = None, lhs_indices: Optional[array] = None, rhs_indices: Optional[array] = None, transpose: bool = True, group_size: int = 64, bits: ...` | Perform quantized matrix multiplication with matrix-level gather. | Needs mapping & tests. |
| [ ] | `mlx.core.generic` | `(...)` | No documentation available. | Needs mapping & tests. |
| [ ] | `mlx.core.get_active_memory` | `() -> int` | Get the actively used memory in bytes. | Needs mapping & tests. |
| [ ] | `mlx.core.get_cache_memory` | `() -> int` | Get the cache size in bytes. | Needs mapping & tests. |
| [ ] | `mlx.core.hadamard_transform` | `(a: array, scale: Optional[float] = None, stream: Union[None, Stream, Device] = None) -> array` | Perform the Walsh-Hadamard transform along the final axis. | Needs mapping & tests. |
| [x] | `mlx.core.identity` | `(n: int, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array` | Create a square identity matrix. | Needs mapping & tests. |
| [ ] | `mlx.core.import_function` | `(file: str) -> Callable` | Import a function from a file. | Needs mapping & tests. |
| [ ] | `mlx.core.inexact` | `(...)` | No documentation available. | Needs mapping & tests. |
| [ ] | `mlx.core.integer` | `(...)` | No documentation available. | Needs mapping & tests. |
| [ ] | `mlx.core.load` | `(file: Union[file, str, pathlib.Path], /, format: Optional[str] = None, return_metadata: bool = False, *, stream: Union[None, Stream, Device] = None) -> Union[array, dict[str, array]]` | Load array(s) from a binary file. | Needs mapping & tests. |
| [ ] | `mlx.core.metal` | `(...)` | mlx.metal | Needs mapping & tests. |
| [ ] | `mlx.core.number` | `(...)` | No documentation available. | Needs mapping & tests. |
| [ ] | `mlx.core.permute_dims` | `(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | See :func:transpose. | Needs mapping & tests. |
| [ ] | `mlx.core.quantize` | `(w: array, /, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = None) -> tuple[array, array, array]` | Quantize the matrix w using bits bits per element. | Needs mapping & tests. |
| [ ] | `mlx.core.quantized_matmul` | `(x: array, w: array, /, scales: array, biases: Optional[array] = None, transpose: bool = True, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = No...` | Perform the matrix multiplication with the quantized matrix w. The | Needs mapping & tests. |
| [ ] | `mlx.core.reset_peak_memory` | `() -> None` | Reset the peak memory to zero. | Needs mapping & tests. |
| [ ] | `mlx.core.save` | `(file: Union[file, str, pathlib.Path], arr: array) -> None` | Save the array to a binary file in .npy format. | Needs mapping & tests. |
| [ ] | `mlx.core.save_gguf` | `(file: Union[file, str, pathlib.Path], arrays: dict[str, array], metadata: dict[str, Union[array, str, list[str]]])` | Save array(s) to a binary file in .gguf format. | Needs mapping & tests. |
| [ ] | `mlx.core.save_safetensors` | `(file: Union[file, str, pathlib.Path], arrays: dict[str, array], metadata: Optional[dict[str, str]] = None)` | Save array(s) to a binary file in .safetensors format. | Needs mapping & tests. |
| [ ] | `mlx.core.savez` | `(file: Union[file, str, pathlib.Path], *args, **kwargs)` | Save several arrays to a binary file in uncompressed .npz | Needs mapping & tests. |
| [ ] | `mlx.core.savez_compressed` | `(file: Union[file, str, pathlib.Path], *args, **kwargs)` | Save several arrays to a binary file in compressed .npz format. | Needs mapping & tests. |
| [ ] | `mlx.core.set_cache_limit` | `(limit: int) -> int` | Set the free cache limit. | Needs mapping & tests. |
| [ ] | `mlx.core.set_default_stream` | `(stream: mlx.core.Stream) -> None` | Set the default stream. | Needs mapping & tests. |
| [ ] | `mlx.core.set_memory_limit` | `(limit: int) -> int` | Set the memory limit. | Needs mapping & tests. |
| [ ] | `mlx.core.set_wired_limit` | `(limit: int) -> int` | Set the wired size limit. | Needs mapping & tests. |
| [ ] | `mlx.core.signedinteger` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.slice` | `(a: array, start_indices: array, axes: Sequence[int], slice_size: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Extract a sub-array from the input array. | Needs mapping & tests. |
| [ ] | `mlx.core.slice_update` | `(a: array, update: array, start_indices: array, axes: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Update a sub-array of the input array. | Needs mapping & tests. |
| [x] | `mlx.core.tan` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise tangent. | Needs mapping & tests. |
| [ ] | `mlx.core.topk` | `(a: array, /, k: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Returns the k largest elements from the input along a given axis. | Needs mapping & tests. |
| [ ] | `mlx.core.unflatten` | `(a: array, /, axis: int, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Unflatten an axis of an array to a shape. | Needs mapping & tests. |
| [ ] | `mlx.core.unsignedinteger` | `(...)` | No documentation available. | Needs mapping & tests. |
| [ ] | `mlx.nn.AllToShardedLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation such | Needs mapping & tests. |
| [ ] | `mlx.nn.QuantizedAllToShardedLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation with | Needs mapping & tests. |
| [ ] | `mlx.nn.QuantizedShardedToAllLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation using | Needs mapping & tests. |
| [ ] | `mlx.nn.ShardedToAllLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation and | Needs mapping & tests. |
| [ ] | `mlx.nn.average_gradients` | `(gradients: Any, group: Optional[mlx.core.distributed.Group] = None, all_reduce_size: int = 33554432, communication_type: Optional[mlx.core.Dtype] = None, communication_stream: Optional[mlx.core.St...` | Average the gradients across the distributed processes in the passed group. | Needs mapping & tests. |
| [x] | `mlx.nn.gelu` | `(x) -> mlx.core.array` | Applies the Gaussian Error Linear Units function. | Needs mapping & tests. |
| [x] | `mlx.nn.glu` | `(x: mlx.core.array, axis: int = -1) -> mlx.core.array` | Applies the gated linear unit function. | Needs mapping & tests. |
| [x] | `mlx.nn.mish` | `(x: mlx.core.array) -> mlx.core.array` | Applies the Mish function, element-wise. | Needs mapping & tests. |
| [ ] | `mlx.nn.prelu` | `(x: mlx.core.array, alpha: mlx.core.array) -> mlx.core.array` | Applies the element-wise parametric ReLU. | Needs mapping & tests. |
| [x] | `mlx.nn.step` | `(x: mlx.core.array, threshold: float = 0.0)` | Applies the Step Activation Function. | Needs mapping & tests. |

| [ ] | `mlx.core.linalg.cholesky_inv` | `(a: array, /, upper: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the inverse of a real symmetric positive definite matrix using its Cholesky decomposition. | Needs mapping & tests. |
| [ ] | `mlx.core.linalg.cross` | `(a: array, b: array, /, axis: int = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Return the cross product of two arrays. | Needs mapping & tests. |
| [ ] | `mlx.core.linalg.pinv` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the (Moore-Penrose) pseudo-inverse of a matrix. | Needs mapping & tests. |
| [ ] | `mlx.core.linalg.svd` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array, array]` | Singular Value Decomposition. | Needs mapping & tests. |
| [ ] | `mlx.core.fft.fft2` | `(a: array, /, n: Optional[Sequence[int]] = None, axes: Sequence[int] = (-2, -1), *, stream: Union[None, Stream, Device] = None) -> array` | 2D discrete Fourier Transform. | Needs mapping & tests. |
| [ ] | `mlx.core.fft.fftn` | `(a: array, /, n: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | nD discrete Fourier Transform. | Needs mapping & tests. |
| [ ] | `mlx.core.fft.ifft2` | `(a: array, /, n: Optional[Sequence[int]] = None, axes: Sequence[int] = (-2, -1), *, stream: Union[None, Stream, Device] = None) -> array` | 2D inverse discrete Fourier Transform. | Needs mapping & tests. |
| [ ] | `mlx.core.fft.ifftn` | `(a: array, /, n: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | nD inverse discrete Fourier Transform. | Needs mapping & tests. |
| [ ] | `mlx.core.fft.irfft2` | `(a: array, /, n: Optional[Sequence[int]] = None, axes: Sequence[int] = (-2, -1), *, stream: Union[None, Stream, Device] = None) -> array` | 2D inverse real discrete Fourier Transform. | Needs mapping & tests. |
| [ ] | `mlx.core.fft.irfftn` | `(a: array, /, n: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | nD inverse real discrete Fourier Transform. | Needs mapping & tests. |
| [ ] | `mlx.core.fft.rfft2` | `(a: array, /, n: Optional[Sequence[int]] = None, axes: Sequence[int] = (-2, -1), *, stream: Union[None, Stream, Device] = None) -> array` | 2D real discrete Fourier Transform. | Needs mapping & tests. |
| [ ] | `mlx.core.fft.rfftn` | `(a: array, /, n: Optional[Sequence[int]] = None, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | nD real discrete Fourier Transform. | Needs mapping & tests. |

| [ ] | `mlx.core.random.categorical` | `(logits: array, axis: int = -1, shape: Optional[Sequence[int]] = None, num_samples: Optional[int] = None, key: Optional[array] = None, *, stream: Union[None, S...` | Sample from a categorical distribution. | Needs mapping & tests. |
| [ ] | `mlx.core.random.dirichlet` | `(alpha: array, shape: Optional[Sequence[int]] = None, key: Optional[array] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Sample from a Dirichlet distribution. | Needs mapping & tests. |
| [ ] | `mlx.core.random.gumbel` | `(shape: Sequence[int], dtype: Optional[Dtype] = float32, key: Optional[array] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Sample from a standard Gumbel distribution. | Needs mapping & tests. |
| [ ] | `mlx.core.random.laplace` | `(shape: Sequence[int], dtype: Optional[Dtype] = float32, key: Optional[array] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Sample from a standard Laplace distribution. | Needs mapping & tests. |
| [ ] | `mlx.core.random.multivariate_normal` | `(mean: array, cov: array, shape: Optional[Sequence[int]] = None, dtype: Optional[Dtype] = float32, key: Optional[array] = None, *, stream: Union[None, Stream, D...` | Sample from a multivariate normal distribution. | Needs mapping & tests. |
| [ ] | `mlx.core.random.state` | `(...)` | The PRNG state. | Needs mapping & tests. |

### Neural Networks (mlx.nn)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `mlx.nn.ALiBi` | ` () ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.AvgPool1d` | ` (kernel_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int], NoneType] = None, padding: Union[int, Tuple[int]] = 0) ` |  Applies 1-dimensional average pooling.  | Needs mapping & tests. |
| [ ] | `mlx.nn.AvgPool2d` | ` (kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int], NoneType] = None, padding: Union[int, Tuple[int, int], NoneType] = 0) ` |  Applies 2-dimensional average pooling.  | Needs mapping & tests. |
| [ ] | `mlx.nn.AvgPool3d` | ` (kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int], NoneType] = None, padding: Union[int, Tuple[int, int, int], NoneType] = 0) ` |  Applies 3-dimensional average pooling.  | Needs mapping & tests. |
| [x] | `mlx.nn.BatchNorm` | ` (num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True) ` |  Applies Batch Normalization over a 2D or 3D input.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Bilinear` | ` (input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True) -> None ` |  Applies a bilinear transformation to the inputs.  | Needs mapping & tests. |
| [ ] | `mlx.nn.CELU` | ` (alpha=1.0) ` |  Applies the Continuously Differentiable Exponential Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Conv1d` | ` (in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True) ` |  Applies a 1-dimensional convolution over the multi-channel input sequence.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Conv2d` | ` (in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, groups: int = 1, bias: bool = T... ` |  Applies a 2-dimensional convolution over the multi-channel input image.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Conv3d` | ` (in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, bias: bool = True) ` |  Applies a 3-dimensional convolution over the multi-channel input image.  | Needs mapping & tests. |
| [ ] | `mlx.nn.ConvTranspose1d` | ` (in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, bias: bool = True) ` |  Applies a 1-dimensional transposed convolution over the multi-channel input sequence.  | Needs mapping & tests. |
| [ ] | `mlx.nn.ConvTranspose2d` | ` (in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, output_padding: Union[int, tupl... ` |  Applies a 2-dimensional transposed convolution over the multi-channel input image.  | Needs mapping & tests. |
| [ ] | `mlx.nn.ConvTranspose3d` | ` (in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, output_padding: Union[int, tupl... ` |  Applies a 3-dimensional transposed convolution over the multi-channel input image.  | Needs mapping & tests. |
| [x] | `mlx.nn.Dropout` | ` (p: float = 0.5) ` |  Randomly zero a portion of the elements during training.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Dropout2d` | ` (p: float = 0.5) ` |  Apply 2D channel-wise dropout during training.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Dropout3d` | ` (p: float = 0.5) ` |  Apply 3D channel-wise dropout during training.  | Needs mapping & tests. |
| [ ] | `mlx.nn.ELU` | ` (alpha=1.0) ` |  Applies the Exponential Linear Unit.  | Needs mapping & tests. |
| [x] | `mlx.nn.Embedding` | ` (num_embeddings: int, dims: int) ` |  Implements a simple lookup table that maps each input integer to a  | Needs mapping & tests. |
| [ ] | `mlx.nn.GELU` | ` (approx='none') ` |  Applies the Gaussian Error Linear Units.  | Needs mapping & tests. |
| [ ] | `mlx.nn.GLU` | ` (axis: int = -1) ` |  Applies the gated linear unit function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.GRU` | ` (input_size: int, hidden_size: int, bias: bool = True) ` |  A gated recurrent unit (GRU) RNN layer.  | Needs mapping & tests. |
| [x] | `mlx.nn.GroupNorm` | ` (num_groups: int, dims: int, eps: float = 1e-05, affine: bool = True, pytorch_compatible: bool = False) ` |  Applies Group Normalization [1] to the inputs.  | Needs mapping & tests. |
| [ ] | `mlx.nn.HardShrink` | ` () ` |  Applies the HardShrink function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.HardTanh` | ` () ` |  Applies the HardTanh function.  | Needs mapping & tests. |
| [x] | `mlx.nn.Hardswish` | ` () ` |  Applies the hardswish function, element-wise.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Identity` | ` (*args: Any, **kwargs: Any) -> None ` |  A placeholder identity operator that is argument-insensitive.  | Needs mapping & tests. |
| [x] | `mlx.nn.InstanceNorm` | ` (dims: int, eps: float = 1e-05, affine: bool = False) ` |  Applies instance normalization [1] on the inputs.  | Needs mapping & tests. |
| [ ] | `mlx.nn.LSTM` | ` (input_size: int, hidden_size: int, bias: bool = True) ` |  An LSTM recurrent layer.  | Needs mapping & tests. |
| [x] | `mlx.nn.LayerNorm` | ` (dims: int, eps: float = 1e-05, affine: bool = True, bias: bool = True) ` |  Applies layer normalization [1] on the inputs.  | Needs mapping & tests. |
| [ ] | `mlx.nn.LeakyReLU` | ` (negative_slope=0.01) ` |  Applies the Leaky Rectified Linear Unit.  | Needs mapping & tests. |
| [x] | `mlx.nn.Linear` | ` (input_dims: int, output_dims: int, bias: bool = True) -> None ` |  Applies an affine transformation to the input.  | Needs mapping & tests. |
| [ ] | `mlx.nn.LogSigmoid` | ` () ` |  Applies the Log Sigmoid function.  | Needs mapping & tests. |
| [x] | `mlx.nn.LogSoftmax` | ` () ` |  Applies the Log Softmax function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.MaxPool1d` | ` (kernel_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int], NoneType] = None, padding: Union[int, Tuple[int]] = 0) ` |  Applies 1-dimensional max pooling.  | Needs mapping & tests. |
| [ ] | `mlx.nn.MaxPool2d` | ` (kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int], NoneType] = None, padding: Union[int, Tuple[int, int], NoneType] = 0) ` |  Applies 2-dimensional max pooling.  | Needs mapping & tests. |
| [ ] | `mlx.nn.MaxPool3d` | ` (kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int], NoneType] = None, padding: Union[int, Tuple[int, int, int], NoneType] = 0) ` |  Applies 3-dimensional max pooling.  | Needs mapping & tests. |
| [x] | `mlx.nn.Mish` | ` () ` |  Applies the Mish function, element-wise.  | Needs mapping & tests. |
| [x] | `mlx.nn.Module` | ` () ` |  Base class for building neural networks with MLX.  | Needs mapping & tests. |
| [ ] | `mlx.nn.MultiHeadAttention` | ` (dims: int, num_heads: int, query_input_dims: Optional[int] = None, key_input_dims: Optional[int] = None, value_input_dims: Optional[int] = None, value_dims: Optional[int] = None, value_output_dims... ` |  Implements the scaled dot product attention with multiple heads.  | Needs mapping & tests. |
| [ ] | `mlx.nn.PReLU` | ` (num_parameters=1, init=0.25) ` |  Applies the element-wise parametric ReLU.  | Needs mapping & tests. |
| [ ] | `mlx.nn.QuantizedEmbedding` | ` (num_embeddings: int, dims: int, group_size: int = 64, bits: int = 4, mode: str = 'affine') ` |  The same as :obj:Embedding but with a  quantized weight matrix.  | Needs mapping & tests. |
| [ ] | `mlx.nn.QuantizedLinear` | ` (input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, mode: str = 'affine') ` |  Applies an affine transformation to the input using a quantized weight matrix.  | Needs mapping & tests. |
| [ ] | `mlx.nn.RMSNorm` | ` (dims: int, eps: float = 1e-05) ` |  Applies Root Mean Square normalization [1] to the inputs.  | Needs mapping & tests. |
| [ ] | `mlx.nn.RNN` | ` (input_size: int, hidden_size: int, bias: bool = True, nonlinearity: Optional[Callable] = None) ` |  An Elman recurrent layer.  | Needs mapping & tests. |
| [ ] | `mlx.nn.ReLU` | ` () ` |  Applies the Rectified Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.ReLU2` | ` () ` |  Applies the ReLU² activation function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.ReLU6` | ` () ` |  Applies the Rectified Linear Unit 6.  | Needs mapping & tests. |
| [ ] | `mlx.nn.RoPE` | ` (dims: int, traditional: bool = False, base: float = 10000, scale: float = 1.0) ` |  Implements the rotary positional encoding.  | Needs mapping & tests. |
| [ ] | `mlx.nn.SELU` | ` () ` |  Applies the Scaled Exponential Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Sequential` | ` (*modules) ` |  A layer that calls the passed callables in order.  | Needs mapping & tests. |
| [ ] | `mlx.nn.SiLU` | ` () ` |  Applies the Sigmoid Linear Unit. Also known as Swish.  | Needs mapping & tests. |
| [x] | `mlx.nn.Sigmoid` | ` () ` |  Applies the sigmoid function, element-wise.  | Needs mapping & tests. |
| [ ] | `mlx.nn.SinusoidalPositionalEncoding` | ` (dims: int, min_freq: float = 0.0001, max_freq: float = 1, scale: Optional[float] = None, cos_first: bool = False, full_turns: bool = False) ` |  Implements sinusoidal positional encoding.  | Needs mapping & tests. |
| [x] | `mlx.nn.Softmax` | ` () ` |  Applies the Softmax function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Softmin` | ` () ` |  Applies the Softmin function.  | Needs mapping & tests. |
| [x] | `mlx.nn.Softplus` | ` () ` |  Applies the Softplus function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Softshrink` | ` (lambd=0.5) ` |  Applies the Softshrink function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Softsign` | ` () ` |  Applies the Softsign function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Step` | ` (threshold: float = 0.0) ` |  Applies the Step Activation Function.  | Needs mapping & tests. |
| [x] | `mlx.nn.Tanh` | ` () ` |  Applies the hyperbolic tangent function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Transformer` | ` (dims: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func... ` |  Implements a standard Transformer model.  | Needs mapping & tests. |
| [ ] | `mlx.nn.TransformerDecoder` | ` (num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<mlx.gc_func object at 0x10484bf90>, norm_first: bool = True, checkpoint: bool = False) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.TransformerDecoderLayer` | ` (dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func object at 0x10484bf90>, norm_first: bool = True) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.TransformerEncoder` | ` (num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<mlx.gc_func object at 0x10484bf90>, norm_first: bool = True, checkpoint: bool = False) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.TransformerEncoderLayer` | ` (dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func object at 0x10484bf90>, norm_first: bool = True) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.Upsample` | ` (scale_factor: Union[float, Tuple], mode: Literal['nearest', 'linear', 'cubic'] = 'nearest', align_corners: bool = False) ` |  Upsample the input signal spatially.  | Needs mapping & tests. |
| [ ] | `mlx.nn.activations` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.base` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [x] | `mlx.nn.celu` | ` (x, alpha=1.0) ` |  Applies the Continuously Differentiable Exponential Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.containers` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.convolution` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.convolution_transpose` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [x] | `mlx.nn.dropout` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [x] | `mlx.nn.elu` | ` (x, alpha=1.0) ` |  Applies the Exponential Linear Unit.  | Needs mapping & tests. |
| [x] | `mlx.nn.embedding` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.gelu_approx` | ` (x) ` |  An approximation to Gaussian Error Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.gelu_fast_approx` | ` (x) ` |  A fast approximation to Gaussian Error Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.hard_shrink` | ` (x, lambd=0.5) ` |  Applies the HardShrink activation function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.hard_tanh` | ` (x, min_val=-1.0, max_val=1.0) ` |  Applies the HardTanh function.  | Needs mapping & tests. |
| [x] | `mlx.nn.hardswish` | ` (x) ` |  Applies the hardswish function, element-wise.  | Needs mapping & tests. |
| [ ] | `mlx.nn.init` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.layers` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [x] | `mlx.nn.leaky_relu` | ` (x, negative_slope=0.01) ` |  Applies the Leaky Rectified Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.linear` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.log_sigmoid` | ` (x) ` |  Applies the Log Sigmoid function.  | Needs mapping & tests. |
| [x] | `mlx.nn.log_softmax` | ` (x, axis=-1) ` |  Applies the Log Softmax function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.normalization` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.pooling` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.positional_encoding` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.quantized` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.recurrent` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [x] | `mlx.nn.relu` | ` (x) ` |  Applies the Rectified Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.relu2` | ` (x) ` |  Applies the ReLU² activation function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.relu6` | ` (x) ` |  Applies the Rectified Linear Unit 6.  | Needs mapping & tests. |
| [x] | `mlx.nn.selu` | ` (x) ` |  Applies the Scaled Exponential Linear Unit.  | Needs mapping & tests. |
| [ ] | `mlx.nn.silu` | ` (x) ` |  Applies the Sigmoid Linear Unit. Also known as Swish.  | Needs mapping & tests. |
| [ ] | `mlx.nn.softmin` | ` (x, axis=-1) ` |  Applies the Softmin function.  | Needs mapping & tests. |
| [x] | `mlx.nn.softplus` | ` (x) ` |  Applies the Softplus function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.softshrink` | ` (x, lambd: float = 0.5) ` |  Applies the Softshrink activation function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.softsign` | ` (x) ` |  Applies the Softsign function.  | Needs mapping & tests. |
| [ ] | `mlx.nn.transformer` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.upsample` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.utils` | ` (...) ` |  No documentation available.  | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.cosine_similarity_loss` | `(x1: mx.array, x2: mx.array, axis: int / None=1, eps: float / None=1e-08, reduction: str / None='none') -> mlx.core.array` | Computes the cosine similarity between the two inputs. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.gaussian_nll_loss` | `(inputs: array, targets: array, vars: array, full: bool / None=False, eps: float / None=1e-06, reduction: str / None='none') -> mlx.core.array` | Computes the negative log likelihood loss for a Gaussian distribution. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.hinge_loss` | `(inputs: array, targets: array, reduction: str / None='none') -> mlx.core.array` | Computes the hinge loss between inputs and targets. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.huber_loss` | `(inputs: array, targets: array, delta: float / None=1.0, reduction: str / None='none') -> mlx.core.array` | Computes the Huber loss between inputs and targets. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.kl_div_loss` | `(inputs: array, targets: array, axis: int / None=-1.0, reduction: str / None='none') -> mlx.core.array` | Computes the Kullback-Leibler divergence loss. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.l1_loss` | `(predictions: array, targets: array, reduction: str / None='mean') -> mlx.core.array` | Computes the L1 loss. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.log_cosh_loss` | `(inputs: array, targets: array, reduction: str / None='none') -> mlx.core.array` | Computes the log cosh loss between inputs and targets. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.margin_ranking_loss` | `(inputs1: array, inputs2: array, targets: array, margin: float / None=0.0, reduction: str / None='none') -> mlx.core.array` | Calculate the margin ranking loss that loss given inputs :math:`x_1`, :math:`x_2` and a label :ma... | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.mse_loss` | `(predictions: array, targets: array, reduction: str / None='mean') -> mlx.core.array` | Computes the mean squared error loss. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.nll_loss` | `(inputs: array, targets: array, axis: int / None=-1.0, reduction: str / None='none') -> mlx.core.array` | Computes the negative log likelihood loss. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.smooth_l1_loss` | `(predictions: array, targets: array, beta: float / None=1.0, reduction: str / None='mean') -> mlx.core.array` | Computes the smooth L1 loss. | Needs mapping & tests. |
| [ ] | `mlx.nn.losses.triplet_loss` | `(anchors: array, positives: array, negatives: array, axis: int / None=-1.0, p: int / None=2, margin: float / None=1.0, eps: float / None=1e-06, reduction: str / None='none') -> mlx.core.array` | Computes the triplet loss for a set of anchor, positive, and negative samples. Margin is represen... | Needs mapping & tests. |

### Optimizers (mlx.optimizers)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `mlx.optimizers.SGD` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False)` | The stochastic gradient descent optimizer. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.RMSprop` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], alpha: float = 0.99, eps: float = 1e-08)` | The RMSprop optimizer [1]. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.Muon` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], momentum: float = 0.95, weight_decay: float = 0.01, nesterov: bool = True, ns_steps: int = 5)` | The Muon optimizer. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.Lion` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.99], weight_decay: float = 0.0)` | The Lion optimizer [1]. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.Adamax` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08)` | The Adamax optimizer, a variant of Adam based on the infinity norm [1]. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.AdamW` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08, weight_decay: float = 0.01, bias_correction: bool = False)` | The AdamW optimizer [1]. We update the weights with a weight_decay | Needs mapping & tests. |
| [ ] | `mlx.optimizers.Adam` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08, bias_correction: bool = False)` | The Adam optimizer [1]. In detail, | Needs mapping & tests. |
| [ ] | `mlx.optimizers.Adagrad` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], eps: float = 1e-08)` | The Adagrad optimizer [1]. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.Adafactor` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array], NoneType] = None, eps: Tuple[float, float] = (1e-30, 0.001), clip_threshold: float = 1.0, decay_rate: float = -0.8, beta_1: ...` | The Adafactor optimizer. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.AdaDelta` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], rho: float = 0.9, eps: float = 1e-06)` | The AdaDelta optimizer with a learning rate [1]. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.MultiOptimizer` | `(optimizers, filters: list = [])` | Wraps a list of optimizers with corresponding weight predicates/filters | Needs mapping & tests. |
| [ ] | `mlx.optimizers.Optimizer` | `(schedulers=None)` | The base class for all optimizers. It allows us to implement an | Needs mapping & tests. |
| [ ] | `mlx.optimizers.clip_grad_norm` | `(grads, max_norm)` | Clips the global norm of the gradients. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.cosine_decay` | `(init: float, decay_steps: int, end: float = 0.0) -> Callable` | Make a cosine decay scheduler. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.exponential_decay` | `(init: float, decay_rate: float) -> Callable` | Make an exponential decay scheduler. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.join_schedules` | `(schedules: List[Callable], boundaries: List[int]) -> Callable` | Join multiple schedules to create a new schedule. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.linear_schedule` | `(init: float, end: float, steps: int) -> Callable` | Make a linear scheduler. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.step_decay` | `(init: float, decay_rate: float, step_size: int) -> Callable` | Make a step decay scheduler. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.tree_map` | `(fn: Callable, tree: Any, *rest: Any, is_leaf: Optional[Callable] = None) -> Any` | Applies fn to the leaves of the Python tree tree and | Needs mapping & tests. |
| [ ] | `mlx.optimizers.tree_merge` | `(tree_a, tree_b, merge_fn=None)` | Merge two Python trees in one containing the values of both. It can be | Needs mapping & tests. |
| [ ] | `mlx.optimizers.tree_reduce` | `(fn, tree, initializer=None, is_leaf=None)` | Applies a reduction to the leaves of a Python tree. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.tree_unflatten` | `(tree: Union[List[Tuple[str, Any]], Dict[str, Any]]) -> Any` | Recreate a Python tree from its flat representation. | Needs mapping & tests. |
| [ ] | `mlx.optimizers.tree_flatten` | `(tree: Any, prefix: str = '', is_leaf: Optional[Callable] = None, destination: Union[List[Tuple[str, Any]], Dict[str, Any], NoneType] = None) -> Union[List[Tuple[str, Any]], Dict[str, Any]]` | Flattens a Python tree to a list of key, value tuples. | Needs mapping & tests. |
