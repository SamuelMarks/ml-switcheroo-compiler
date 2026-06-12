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
| [x] | `mlx.core.ArrayIterator` | `(...)` | A helper object to iterate over the 1st dimension of an array. | Needs mapping & tests. |
| [x] | `mlx.core.ArrayLike` | `(...)` | Any Python object which has an __mlx__array__ method that | Needs mapping & tests. |
| [x] | `mlx.core.Device` | `(...)` | A device to run operations on. | Needs mapping & tests. |
| [x] | `mlx.core.DeviceType` | `(value, names=None, *, module=None, qualname=None, type=None, start=1)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.Dtype` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.DtypeCategory` | `(value, names=None, *, module=None, qualname=None, type=None, start=1)` | Type to hold categories of :class:dtypes <Dtype>. | Needs mapping & tests. |
| [x] | `mlx.core.FunctionExporter` | `(...)` | A context managing class for exporting multiple traces of the same | Needs mapping & tests. |
| [x] | `mlx.core.Stream` | `(...)` | A stream for running operations on a given device. | Needs mapping & tests. |
| [ ] | `mlx.core.StreamContext` | `(...)` | A context manager for setting the current device and stream. | Needs mapping & tests. |
| [x] | `mlx.core.abs` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise absolute value. | Needs mapping & tests. |
| [x] | `mlx.core.add` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise addition. | Needs mapping & tests. |
| [x] | `mlx.core.addmm` | `(c: array, a: array, b: array, /, alpha: float = 1.0, beta: float = 1.0,  *, stream: Union[None, Stream, Device] = None) -> array` | Matrix multiplication with addition and optional scaling. | Needs mapping & tests. |
| [x] | `mlx.core.all` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | An and reduction over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.allclose` | `(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array` | Approximate comparison of two arrays. | Needs mapping & tests. |
| [x] | `mlx.core.any` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | An or reduction over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.arange` | `(start : Union[int, float], stop : Union[int, float], step : Union[None, int, float], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Overloaded function. | Needs mapping & tests. |
| [x] | `mlx.core.arccos` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse cosine. | Needs mapping & tests. |
| [x] | `mlx.core.arccosh` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse hyperbolic cosine. | Needs mapping & tests. |
| [x] | `mlx.core.arcsin` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse sine. | Needs mapping & tests. |
| [x] | `mlx.core.arcsinh` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse hyperbolic sine. | Needs mapping & tests. |
| [x] | `mlx.core.arctan` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse tangent. | Needs mapping & tests. |
| [x] | `mlx.core.arctan2` | `(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse tangent of the ratio of two arrays. | Needs mapping & tests. |
| [x] | `mlx.core.arctanh` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse hyperbolic tangent. | Needs mapping & tests. |
| [x] | `mlx.core.argmax` | `(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Indices of the maximum values along the axis. | Needs mapping & tests. |
| [x] | `mlx.core.argmin` | `(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Indices of the minimum values along the axis. | Needs mapping & tests. |
| [x] | `mlx.core.argpartition` | `(a: array, /, kth: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Returns the indices that partition the array. | Needs mapping & tests. |
| [x] | `mlx.core.argsort` | `(a: array, /, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Returns the indices that sort the array. | Needs mapping & tests. |
| [x] | `mlx.core.array` | `(...)` | An N-dimensional array object. | Needs mapping & tests. |
| [x] | `mlx.core.array_equal` | `(a: Union[scalar, array], b: Union[scalar, array], equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array` | Array equality check. | Needs mapping & tests. |
| [x] | `mlx.core.as_strided` | `(a: array, /, shape: Optional[Sequence[int]] = None, strides: Optional[Sequence[int]] = None, offset: int = 0, *, stream: Union[None, Stream, Device] = None) -> array` | Create a view into the array with the given shape and strides. | Needs mapping & tests. |
| [x] | `mlx.core.async_eval` | `(*args)` | Asynchronously evaluate an :class:array or tree of :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.atleast_1d` | `(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]` | Convert all arrays to have at least one dimension. | Needs mapping & tests. |
| [x] | `mlx.core.atleast_2d` | `(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]` | Convert all arrays to have at least two dimensions. | Needs mapping & tests. |
| [x] | `mlx.core.atleast_3d` | `(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]` | Convert all arrays to have at least three dimensions. | Needs mapping & tests. |
| [x] | `mlx.core.bfloat16` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.bitwise_and` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise bitwise and. | Needs mapping & tests. |
| [x] | `mlx.core.bitwise_invert` | `(a: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise bitwise inverse. | Needs mapping & tests. |
| [x] | `mlx.core.bitwise_or` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise bitwise or. | Needs mapping & tests. |
| [x] | `mlx.core.bitwise_xor` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise bitwise xor. | Needs mapping & tests. |
| [x] | `mlx.core.block_masked_mm` | `(a: array, b: array, /, block_size: int = 64, mask_out: Optional[array] = None, mask_lhs: Optional[array] = None, mask_rhs: Optional[array] = None, *, stream: Union[None, Stream, Device] = None) ->...` | Matrix multiplication with block masking. | Needs mapping & tests. |
| [x] | `mlx.core.bool_` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.broadcast_arrays` | `(*arrays: array, stream: Union[None, Stream, Device] = None) -> Tuple[array, ...]` | Broadcast arrays against one another. | Needs mapping & tests. |
| [x] | `mlx.core.broadcast_shapes` | `(*shapes: Sequence[int]) -> Tuple[int]` | Broadcast shapes. | Needs mapping & tests. |
| [x] | `mlx.core.broadcast_to` | `(a: Union[scalar, array], /, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Broadcast an array to the given shape. | Needs mapping & tests. |
| [x] | `mlx.core.ceil` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise ceil. | Needs mapping & tests. |
| [x] | `mlx.core.checkpoint` | `(fun: collections.abc.Callable) -> collections.abc.Callable` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.clear_cache` | `() -> None` | Clear the memory cache. | Needs mapping & tests. |
| [x] | `mlx.core.clip` | `(a: array, /, a_min: Union[scalar, array, None], a_max: Union[scalar, array, None], *, stream: Union[None, Stream, Device] = None) -> array` | Clip the values of the array between the given minimum and maximum. | Needs mapping & tests. |
| [x] | `mlx.core.compile` | `(fun: Callable, inputs: Optional[object] = None, outputs: Optional[object] = None, shapeless: bool = False) -> Callable` | Returns a compiled function which produces the same output as fun. | Needs mapping & tests. |
| [x] | `mlx.core.complex64` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.complexfloating` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.concat` | `(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array` | See :func:concatenate. | Needs mapping & tests. |
| [x] | `mlx.core.concatenate` | `(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array` | Concatenate the arrays along the given axis. | Needs mapping & tests. |
| [x] | `mlx.core.conj` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> array` | Return the elementwise complex conjugate of the input. | Needs mapping & tests. |
| [x] | `mlx.core.conjugate` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> array` | Return the elementwise complex conjugate of the input. | Needs mapping & tests. |
| [x] | `mlx.core.contiguous` | `(a: array, /, allow_col_major: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Force an array to be row contiguous. Copy if necessary. | Needs mapping & tests. |
| [x] | `mlx.core.conv1d` | `(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array` | 1D convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv2d` | `(input: array, weight: array, /, stride: Union[int, tuple[int, int]] = 1, padding: Union[int, tuple[int, int]] = 0, dilation: Union[int, tuple[int, int]] = 1, groups: int = 1, *, stream: Union[None...` | 2D convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv3d` | `(input: array, weight: array, /, stride: Union[int, tuple[int, int, int]] = 1, padding: Union[int, tuple[int, int, int]] = 0, dilation: Union[int, tuple[int, int, int]] = 1, groups: int = 1, *, str...` | 3D convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv_general` | `(input: array, weight: array, /, stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int], tuple[Sequence[int], Sequence[int]]] = 0, kernel_dilation: Union[int, Sequence[int]] = 1, ...` | General convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv_transpose1d` | `(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array` | 1D transposed convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv_transpose2d` | `(input: array, weight: array, /, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, output_padding: Union[int, Tuple[int, ...` | 2D transposed convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.conv_transpose3d` | `(input: array, weight: array, /, stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int]] = 0, dilation: Union[int, Tuple[int, int, int]] = 1, output_padding: Union[i...` | 3D transposed convolution over an input with several channels | Needs mapping & tests. |
| [x] | `mlx.core.convolve` | `(a: array, v: array, /, mode: str = "full", *, stream: Union[None, Stream, Device] = None) -> array` | The discrete convolution of 1D arrays. | Needs mapping & tests. |
| [x] | `mlx.core.cos` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise cosine. | Needs mapping & tests. |
| [x] | `mlx.core.cosh` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise hyperbolic cosine. | Needs mapping & tests. |
| [x] | `mlx.core.cpu` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.cummax` | `(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array` | Return the cumulative maximum of the elements along the given axis. | Needs mapping & tests. |
| [x] | `mlx.core.cummin` | `(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array` | Return the cumulative minimum of the elements along the given axis. | Needs mapping & tests. |
| [x] | `mlx.core.cumprod` | `(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array` | Return the cumulative product of the elements along the given axis. | Needs mapping & tests. |
| [x] | `mlx.core.cumsum` | `(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array` | Return the cumulative sum of the elements along the given axis. | Needs mapping & tests. |
| [x] | `mlx.core.custom_function` | `(...)` | Set up a function for custom gradient and vmap definitions. | Needs mapping & tests. |
| [x] | `mlx.core.default_device` | `() -> mlx.core.Device` | Get the default device. | Needs mapping & tests. |
| [x] | `mlx.core.default_stream` | `(device: mlx.core.Device) -> mlx.core.Stream` | Get the device's default stream. | Needs mapping & tests. |
| [x] | `mlx.core.degrees` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Convert angles from radians to degrees. | Needs mapping & tests. |
| [x] | `mlx.core.depends` | `(inputs: Union[array, Sequence[array]], dependencies: Union[array, Sequence[array]])` | Insert dependencies between arrays in the graph. The outputs are | Needs mapping & tests. |
| [x] | `mlx.core.dequantize` | `(w: array, /, scales: array, biases: Optional[array] = None, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = None) -> array` | Dequantize the matrix w using quantization parameters. | Needs mapping & tests. |
| [x] | `mlx.core.diag` | `(a: array, /, k: int = 0, *, stream: Union[None, Stream, Device] = None) -> array` | Extract a diagonal or construct a diagonal matrix. | Needs mapping & tests. |
| [x] | `mlx.core.diagonal` | `(a: array, offset: int = 0, axis1: int = 0, axis2: int = 1, stream: Union[None, Stream, Device] = None) -> array` | Return specified diagonals. | Needs mapping & tests. |
| [x] | `mlx.core.disable_compile` | `() -> None` | Globally disable compilation. Setting the environment variable | Needs mapping & tests. |
| [x] | `mlx.core.divide` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise division. | Needs mapping & tests. |
| [x] | `mlx.core.divmod` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise quotient and remainder. | Needs mapping & tests. |
| [x] | `mlx.core.e` | `(...)` | Convert a string or number to a floating point number, if possible. | Needs mapping & tests. |
| [x] | `mlx.core.einsum` | `(subscripts: str, *operands, stream: Union[None, Stream, Device] = None) -> array` | Perform the Einstein summation convention on the operands. | Needs mapping & tests. |
| [x] | `mlx.core.einsum_path` | `(subscripts: str, *operands)` | Compute the contraction order for the given Einstein summation. | Needs mapping & tests. |
| [x] | `mlx.core.enable_compile` | `() -> None` | Globally enable compilation. This will override the environment | Needs mapping & tests. |
| [x] | `mlx.core.equal` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise equality. | Needs mapping & tests. |
| [x] | `mlx.core.erf` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise error function. | Needs mapping & tests. |
| [x] | `mlx.core.erfinv` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse of :func:erf. | Needs mapping & tests. |
| [x] | `mlx.core.euler_gamma` | `(...)` | Convert a string or number to a floating point number, if possible. | Needs mapping & tests. |
| [x] | `mlx.core.eval` | `(*args) -> None` | Evaluate an :class:array or tree of :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.exp` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise exponential. | Needs mapping & tests. |
| [x] | `mlx.core.expand_dims` | `(a: array, /, axis: Union[int, Sequence[int]], *, stream: Union[None, Stream, Device] = None) -> array` | Add a size one dimension at the given axis. | Needs mapping & tests. |
| [x] | `mlx.core.expm1` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise exponential minus 1. | Needs mapping & tests. |
| [x] | `mlx.core.export_function` | `(arg0: object, fun: collections.abc.Callable, *args, shapeless: bool = False, **kwargs) -> None` | Export an MLX function. | Needs mapping & tests. |
| [x] | `mlx.core.export_to_dot` | `(file: object, *args, **kwargs) -> None` | Export a graph to DOT format for visualization. | Needs mapping & tests. |
| [x] | `mlx.core.exporter` | `(file: str, fun: collections.abc.Callable, *, shapeless: bool = False) -> mlx.core.FunctionExporter` | Make a callable object to export multiple traces of a function to a file. | Needs mapping & tests. |
| [x] | `mlx.core.eye` | `(n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array` | Create an identity matrix or a general diagonal matrix. | Needs mapping & tests. |
| [x] | `mlx.core.finfo` | `(...)` | Get information on floating-point types. | Needs mapping & tests. |
| [x] | `mlx.core.flatten` | `(a: array, /, start_axis: int = 0, end_axis: int = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Flatten an array. | Needs mapping & tests. |
| [x] | `mlx.core.float16` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.float32` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.float64` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.floating` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.floor` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise floor. | Needs mapping & tests. |
| [x] | `mlx.core.floor_divide` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise integer division. | Needs mapping & tests. |
| [x] | `mlx.core.full` | `(shape: Union[int, Sequence[int]], vals: Union[scalar, array], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Construct an array with the given value. | Needs mapping & tests. |
| [x] | `mlx.core.gather_mm` | `(a: array, b: array, /, lhs_indices: array, rhs_indices: array, *, sorted_indices: bool = False, stream: Union[None, Stream, Device] = None) -> array` | Matrix multiplication with matrix-level gather. | Needs mapping & tests. |
| [x] | `mlx.core.gather_qmm` | `(x: array, w: array, /, scales: array, biases: Optional[array] = None, lhs_indices: Optional[array] = None, rhs_indices: Optional[array] = None, transpose: bool = True, group_size: int = 64, bits: ...` | Perform quantized matrix multiplication with matrix-level gather. | Needs mapping & tests. |
| [x] | `mlx.core.generic` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.get_active_memory` | `() -> int` | Get the actively used memory in bytes. | Needs mapping & tests. |
| [x] | `mlx.core.get_cache_memory` | `() -> int` | Get the cache size in bytes. | Needs mapping & tests. |
| [x] | `mlx.core.get_peak_memory` | `() -> int` | Get the peak amount of used memory in bytes. | Needs mapping & tests. |
| [x] | `mlx.core.gpu` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.grad` | `(fun: Callable, argnums: Optional[Union[int, Sequence[int]]] = None, argnames: Union[str, Sequence[str]] = []) -> Callable` | Returns a function which computes the gradient of fun. | Needs mapping & tests. |
| [x] | `mlx.core.greater` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise greater than. | Needs mapping & tests. |
| [x] | `mlx.core.greater_equal` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise greater or equal. | Needs mapping & tests. |
| [x] | `mlx.core.hadamard_transform` | `(a: array, scale: Optional[float] = None, stream: Union[None, Stream, Device] = None) -> array` | Perform the Walsh-Hadamard transform along the final axis. | Needs mapping & tests. |
| [x] | `mlx.core.identity` | `(n: int, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array` | Create a square identity matrix. | Needs mapping & tests. |
| [x] | `mlx.core.iinfo` | `(...)` | Get information on integer types. | Needs mapping & tests. |
| [x] | `mlx.core.imag` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Returns the imaginary part of a complex array. | Needs mapping & tests. |
| [x] | `mlx.core.import_function` | `(file: str) -> Callable` | Import a function from a file. | Needs mapping & tests. |
| [x] | `mlx.core.inexact` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.inf` | `(...)` | Convert a string or number to a floating point number, if possible. | Needs mapping & tests. |
| [x] | `mlx.core.inner` | `(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Ordinary inner product of vectors for 1-D arrays, in higher dimensions a sum product over the last axes. | Needs mapping & tests. |
| [x] | `mlx.core.int16` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.int32` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.int64` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.int8` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.integer` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.is_available` | `(device: mlx.core.Device) -> bool` | Check if a back-end is available for the given device. | Needs mapping & tests. |
| [x] | `mlx.core.isclose` | `(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array` | Returns a boolean array where two arrays are element-wise equal within a tolerance. | Needs mapping & tests. |
| [x] | `mlx.core.isfinite` | `(a: array, stream: Union[None, Stream, Device] = None) -> array` | Return a boolean array indicating which elements are finite. | Needs mapping & tests. |
| [x] | `mlx.core.isinf` | `(a: array, stream: Union[None, Stream, Device] = None) -> array` | Return a boolean array indicating which elements are +/- inifnity. | Needs mapping & tests. |
| [x] | `mlx.core.isnan` | `(a: array, stream: Union[None, Stream, Device] = None) -> array` | Return a boolean array indicating which elements are NaN. | Needs mapping & tests. |
| [x] | `mlx.core.isneginf` | `(a: array, stream: Union[None, Stream, Device] = None) -> array` | Return a boolean array indicating which elements are negative infinity. | Needs mapping & tests. |
| [x] | `mlx.core.isposinf` | `(a: array, stream: Union[None, Stream, Device] = None) -> array` | Return a boolean array indicating which elements are positive infinity. | Needs mapping & tests. |
| [x] | `mlx.core.issubdtype` | `(arg1: Union[Dtype, DtypeCategory], arg2: Union[Dtype, DtypeCategory]) -> bool` | Check if a :obj:Dtype or :obj:DtypeCategory is a subtype | Needs mapping & tests. |
| [x] | `mlx.core.jvp` | `(fun: Callable, primals: list[array], tangents: list[array]) -> tuple[list[array], list[array]]` | Compute the Jacobian-vector product. | Needs mapping & tests. |
| [x] | `mlx.core.kron` | `(a: array, b: array, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the Kronecker product of two arrays a and b. | Needs mapping & tests. |
| [x] | `mlx.core.left_shift` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise left shift. | Needs mapping & tests. |
| [x] | `mlx.core.less` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise less than. | Needs mapping & tests. |
| [x] | `mlx.core.less_equal` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise less than or equal. | Needs mapping & tests. |
| [x] | `mlx.core.linspace` | `(start, stop, num: Optional[int] = 50, dtype: Optional[Dtype] = float32, stream: Union[None, Stream, Device] = None) -> array` | Generate num evenly spaced numbers over interval [start, stop]. | Needs mapping & tests. |
| [x] | `mlx.core.load` | `(file: Union[file, str, pathlib.Path], /, format: Optional[str] = None, return_metadata: bool = False, *, stream: Union[None, Stream, Device] = None) -> Union[array, dict[str, array]]` | Load array(s) from a binary file. | Needs mapping & tests. |
| [x] | `mlx.core.log` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise natural logarithm. | Needs mapping & tests. |
| [x] | `mlx.core.log10` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise base-10 logarithm. | Needs mapping & tests. |
| [x] | `mlx.core.log1p` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise natural log of one plus the array. | Needs mapping & tests. |
| [x] | `mlx.core.log2` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise base-2 logarithm. | Needs mapping & tests. |
| [x] | `mlx.core.logaddexp` | `(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise log-add-exp. | Needs mapping & tests. |
| [x] | `mlx.core.logcumsumexp` | `(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array` | Return the cumulative logsumexp of the elements along the given axis. | Needs mapping & tests. |
| [x] | `mlx.core.logical_and` | `(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise logical and. | Needs mapping & tests. |
| [x] | `mlx.core.logical_not` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise logical not. | Needs mapping & tests. |
| [x] | `mlx.core.logical_or` | `(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise logical or. | Needs mapping & tests. |
| [x] | `mlx.core.logsumexp` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | A log-sum-exp reduction over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.matmul` | `(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Matrix multiplication. | Needs mapping & tests. |
| [x] | `mlx.core.max` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | A max reduction over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.maximum` | `(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise maximum. | Needs mapping & tests. |
| [x] | `mlx.core.mean` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the mean(s) over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.meshgrid` | `(*arrays: array, sparse: Optional[bool] = False, indexing: Optional[str] = 'xy', stream: Union[None, Stream, Device] = None) -> array` | Generate multidimensional coordinate grids from 1-D coordinate arrays | Needs mapping & tests. |
| [x] | `mlx.core.min` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | A min reduction over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.minimum` | `(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise minimum. | Needs mapping & tests. |
| [x] | `mlx.core.moveaxis` | `(a: array, /, source: int, destination: int, *, stream: Union[None, Stream, Device] = None) -> array` | Move an axis to a new position. | Needs mapping & tests. |
| [x] | `mlx.core.multiply` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise multiplication. | Needs mapping & tests. |
| [x] | `mlx.core.nan` | `(...)` | Convert a string or number to a floating point number, if possible. | Needs mapping & tests. |
| [x] | `mlx.core.nan_to_num` | `(a: Union[scalar, array], nan: float = 0, posinf: Optional[float] = None, neginf: Optional[float] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Replace NaN and Inf values with finite numbers. | Needs mapping & tests. |
| [x] | `mlx.core.negative` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise negation. | Needs mapping & tests. |
| [x] | `mlx.core.new_stream` | `(device: mlx.core.Device) -> mlx.core.Stream` | Make a new stream on the given device. | Needs mapping & tests. |
| [x] | `mlx.core.newaxis` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.not_equal` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise not equal. | Needs mapping & tests. |
| [x] | `mlx.core.number` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.ones` | `(shape: Union[int, Sequence[int]], dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array` | Construct an array of ones. | Needs mapping & tests. |
| [x] | `mlx.core.ones_like` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | An array of ones like the input. | Needs mapping & tests. |
| [x] | `mlx.core.outer` | `(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the outer product of two 1-D arrays, if the array's passed are not 1-D a flatten op will be run beforehand. | Needs mapping & tests. |
| [x] | `mlx.core.pad` | `(a: array, pad_width: Union[int, tuple[int], tuple[int, int], list[tuple[int, int]]], mode: Literal['constant', 'edge'] = 'constant', constant_values: Union[scalar, array] = 0, *, stream: Union[Non...` | Pad an array with a constant value | Needs mapping & tests. |
| [x] | `mlx.core.partition` | `(a: array, /, kth: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Returns a partitioned copy of the array such that the smaller kth | Needs mapping & tests. |
| [x] | `mlx.core.permute_dims` | `(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | See :func:transpose. | Needs mapping & tests. |
| [x] | `mlx.core.pi` | `(...)` | Convert a string or number to a floating point number, if possible. | Needs mapping & tests. |
| [x] | `mlx.core.power` | `(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise power operation. | Needs mapping & tests. |
| [x] | `mlx.core.prod` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | An product reduction over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.put_along_axis` | `(a: array, /, indices: array, values: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Put values along an axis at the specified indices. | Needs mapping & tests. |
| [x] | `mlx.core.quantize` | `(w: array, /, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = None) -> tuple[array, array, array]` | Quantize the matrix w using bits bits per element. | Needs mapping & tests. |
| [x] | `mlx.core.quantized_matmul` | `(x: array, w: array, /, scales: array, biases: Optional[array] = None, transpose: bool = True, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = No...` | Perform the matrix multiplication with the quantized matrix w. The | Needs mapping & tests. |
| [x] | `mlx.core.radians` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Convert angles from degrees to radians. | Needs mapping & tests. |
| [x] | `mlx.core.real` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Returns the real part of a complex array. | Needs mapping & tests. |
| [x] | `mlx.core.reciprocal` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise reciprocal. | Needs mapping & tests. |
| [x] | `mlx.core.remainder` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise remainder of division. | Needs mapping & tests. |
| [x] | `mlx.core.repeat` | `(array: array, repeats: int, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Repeat an array along a specified axis. | Needs mapping & tests. |
| [x] | `mlx.core.reset_peak_memory` | `() -> None` | Reset the peak memory to zero. | Needs mapping & tests. |
| [x] | `mlx.core.reshape` | `(a: array, /, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Reshape an array while preserving the size. | Needs mapping & tests. |
| [x] | `mlx.core.right_shift` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise right shift. | Needs mapping & tests. |
| [x] | `mlx.core.roll` | `(a: array, shift: Union[int, Tuple[int]], axis: Union[None, int, Tuple[int]] = None, /, *, stream: Union[None, Stream, Device] = None) -> array` | Roll array elements along a given axis. | Needs mapping & tests. |
| [x] | `mlx.core.round` | `(a: array, /, decimals: int = 0, stream: Union[None, Stream, Device] = None) -> array` | Round to the given number of decimals. | Needs mapping & tests. |
| [x] | `mlx.core.rsqrt` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise reciprocal and square root. | Needs mapping & tests. |
| [x] | `mlx.core.save` | `(file: Union[file, str, pathlib.Path], arr: array) -> None` | Save the array to a binary file in .npy format. | Needs mapping & tests. |
| [x] | `mlx.core.save_gguf` | `(file: Union[file, str, pathlib.Path], arrays: dict[str, array], metadata: dict[str, Union[array, str, list[str]]])` | Save array(s) to a binary file in .gguf format. | Needs mapping & tests. |
| [x] | `mlx.core.save_safetensors` | `(file: Union[file, str, pathlib.Path], arrays: dict[str, array], metadata: Optional[dict[str, str]] = None)` | Save array(s) to a binary file in .safetensors format. | Needs mapping & tests. |
| [x] | `mlx.core.savez` | `(file: Union[file, str, pathlib.Path], *args, **kwargs)` | Save several arrays to a binary file in uncompressed .npz | Needs mapping & tests. |
| [x] | `mlx.core.savez_compressed` | `(file: Union[file, str, pathlib.Path], *args, **kwargs)` | Save several arrays to a binary file in compressed .npz format. | Needs mapping & tests. |
| [x] | `mlx.core.segmented_mm` | `(a: array, b: array, /, segments: array, *, stream: Union[None, Stream, Device] = None) -> array` | Perform a matrix multiplication but segment the inner dimension and | Needs mapping & tests. |
| [x] | `mlx.core.set_cache_limit` | `(limit: int) -> int` | Set the free cache limit. | Needs mapping & tests. |
| [x] | `mlx.core.set_default_device` | `(device: mlx.core.Device) -> None` | Set the default device. | Needs mapping & tests. |
| [x] | `mlx.core.set_default_stream` | `(stream: mlx.core.Stream) -> None` | Set the default stream. | Needs mapping & tests. |
| [x] | `mlx.core.set_memory_limit` | `(limit: int) -> int` | Set the memory limit. | Needs mapping & tests. |
| [x] | `mlx.core.set_wired_limit` | `(limit: int) -> int` | Set the wired size limit. | Needs mapping & tests. |
| [x] | `mlx.core.sigmoid` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise logistic sigmoid. | Needs mapping & tests. |
| [x] | `mlx.core.sign` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise sign. | Needs mapping & tests. |
| [x] | `mlx.core.signedinteger` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.sin` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise sine. | Needs mapping & tests. |
| [x] | `mlx.core.sinh` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise hyperbolic sine. | Needs mapping & tests. |
| [x] | `mlx.core.slice` | `(a: array, start_indices: array, axes: Sequence[int], slice_size: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Extract a sub-array from the input array. | Needs mapping & tests. |
| [x] | `mlx.core.slice_update` | `(a: array, update: array, start_indices: array, axes: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Update a sub-array of the input array. | Needs mapping & tests. |
| [x] | `mlx.core.softmax` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Perform the softmax along the given axis. | Needs mapping & tests. |
| [x] | `mlx.core.sort` | `(a: array, /, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Returns a sorted copy of the array. | Needs mapping & tests. |
| [x] | `mlx.core.split` | `(a: array, /, indices_or_sections: Union[int, Sequence[int]], axis: int = 0, *, stream: Union[None, Stream, Device] = None) -> array` | Split an array along a given axis. | Needs mapping & tests. |
| [x] | `mlx.core.sqrt` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise square root. | Needs mapping & tests. |
| [x] | `mlx.core.square` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise square. | Needs mapping & tests. |
| [x] | `mlx.core.squeeze` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Remove length one axes from an array. | Needs mapping & tests. |
| [x] | `mlx.core.stack` | `(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array` | Stacks the arrays along a new axis. | Needs mapping & tests. |
| [x] | `mlx.core.std` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the standard deviation(s) over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.stop_gradient` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Stop gradients from being computed. | Needs mapping & tests. |
| [x] | `mlx.core.stream` | `(s: typing.Union[mlx.core.Stream, mlx.core.Device]) -> mlx.core.StreamContext` | Create a context manager to set the default device and stream. | Needs mapping & tests. |
| [x] | `mlx.core.subtract` | `(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise subtraction. | Needs mapping & tests. |
| [x] | `mlx.core.sum` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Sum reduce the array over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.swapaxes` | `(a: array, /, axis1 : int, axis2: int, *, stream: Union[None, Stream, Device] = None) -> array` | Swap two axes of an array. | Needs mapping & tests. |
| [x] | `mlx.core.synchronize` | `(stream: typing.Optional[mlx.core.Stream] = None) -> None` | Synchronize with the given stream. | Needs mapping & tests. |
| [x] | `mlx.core.take` | `(a: array, /, indices: Union[int, array], axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Take elements along an axis. | Needs mapping & tests. |
| [x] | `mlx.core.take_along_axis` | `(a: array, /, indices: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Take values along an axis at the specified indices. | Needs mapping & tests. |
| [x] | `mlx.core.tan` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise tangent. | Needs mapping & tests. |
| [x] | `mlx.core.tanh` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise hyperbolic tangent. | Needs mapping & tests. |
| [x] | `mlx.core.tensordot` | `(a: array, b: array, /, axes: Union[int, list[Sequence[int]]] = 2, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the tensor dot product along the specified axes. | Needs mapping & tests. |
| [x] | `mlx.core.tile` | `(a: array, reps: Union[int, Sequence[int]], /, *, stream: Union[None, Stream, Device] = None) -> array` | Construct an array by repeating a the number of times given by reps. | Needs mapping & tests. |
| [x] | `mlx.core.topk` | `(a: array, /, k: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Returns the k largest elements from the input along a given axis. | Needs mapping & tests. |
| [x] | `mlx.core.trace` | `(a: array, /, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Return the sum along a specified diagonal in the given array. | Needs mapping & tests. |
| [x] | `mlx.core.transpose` | `(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | Transpose the dimensions of the array. | Needs mapping & tests. |
| [x] | `mlx.core.tri` | `(n: int, m: int, k: int, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array` | An array with ones at and below the given diagonal and zeros elsewhere. | Needs mapping & tests. |
| [x] | `mlx.core.tril` | `(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array` | Zeros the array above the given diagonal. | Needs mapping & tests. |
| [x] | `mlx.core.triu` | `(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array` | Zeros the array below the given diagonal. | Needs mapping & tests. |
| [x] | `mlx.core.uint16` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.uint32` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.uint64` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.uint8` | `(...)` | An object to hold the type of a :class:array. | Needs mapping & tests. |
| [x] | `mlx.core.unflatten` | `(a: array, /, axis: int, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Unflatten an axis of an array to a shape. | Needs mapping & tests. |
| [x] | `mlx.core.unsignedinteger` | `(...)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.core.value_and_grad` | `(fun: Callable, argnums: Optional[Union[int, Sequence[int]]] = None, argnames: Union[str, Sequence[str]] = []) -> Callable` | Returns a function which computes the value and gradient of fun. | Needs mapping & tests. |
| [x] | `mlx.core.var` | `(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the variance(s) over the given axes. | Needs mapping & tests. |
| [x] | `mlx.core.view` | `(a: Union[scalar, array], dtype: Dtype, stream: Union[None, Stream, Device] = None) -> array` | View the array as a different type. | Needs mapping & tests. |
| [x] | `mlx.core.vjp` | `(fun: Callable, primals: list[array], cotangents: list[array]) -> tuple[list[array], list[array]]` | Compute the vector-Jacobian product. | Needs mapping & tests. |
| [x] | `mlx.core.vmap` | `(fun: Callable, in_axes: object = 0, out_axes: object = 0) -> Callable` | Returns a vectorized version of fun. | Needs mapping & tests. |
| [x] | `mlx.core.where` | `(condition: Union[scalar, array], x: Union[scalar, array], y: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array` | Select from x or y according to condition. | Needs mapping & tests. |
| [x] | `mlx.core.zeros` | `(shape: Union[int, Sequence[int]], dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array` | Construct an array of zeros. | Needs mapping & tests. |
| [x] | `mlx.core.zeros_like` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | An array of zeros like the input. | Needs mapping & tests. |

### FFT Operations (mlx.core.fft)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `mlx.core.fft.fft` | `(a: mlx.core.array, n: typing.Optional[int] = None, axis: int = -1, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] = None) -> mlx.core.array` | One dimensional discrete Fourier Transform. | Needs mapping & tests. |
| [x] | `mlx.core.fft.fft2` | `(a: mlx.core.array, s: typing.Optional[tuple[int, ...]] = None, axes: typing.Optional[collections.abc.Sequence[int]] = [-2, -1], stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Devic...` | Two dimensional discrete Fourier Transform. | Needs mapping & tests. |
| [x] | `mlx.core.fft.fftn` | `(a: mlx.core.array, s: typing.Optional[tuple[int, ...]] = None, axes: typing.Optional[collections.abc.Sequence[int]] = None, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] ...` | n-dimensional discrete Fourier Transform. | Needs mapping & tests. |
| [x] | `mlx.core.fft.fftshift` | `(a: mlx.core.array, axes: typing.Optional[collections.abc.Sequence[int]] = None, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] = None) -> mlx.core.array` | Shift the zero-frequency component to the center of the spectrum. | Needs mapping & tests. |
| [x] | `mlx.core.fft.ifft` | `(a: mlx.core.array, n: typing.Optional[int] = None, axis: int = -1, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] = None) -> mlx.core.array` | One dimensional inverse discrete Fourier Transform. | Needs mapping & tests. |
| [x] | `mlx.core.fft.ifft2` | `(a: mlx.core.array, s: typing.Optional[tuple[int, ...]] = None, axes: typing.Optional[collections.abc.Sequence[int]] = [-2, -1], stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Devic...` | Two dimensional inverse discrete Fourier Transform. | Needs mapping & tests. |
| [x] | `mlx.core.fft.ifftn` | `(a: mlx.core.array, s: typing.Optional[tuple[int, ...]] = None, axes: typing.Optional[collections.abc.Sequence[int]] = None, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] ...` | n-dimensional inverse discrete Fourier Transform. | Needs mapping & tests. |
| [x] | `mlx.core.fft.ifftshift` | `(a: mlx.core.array, axes: typing.Optional[collections.abc.Sequence[int]] = None, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] = None) -> mlx.core.array` | The inverse of :func:fftshift. While identical to :func:fftshift for even-length axes, | Needs mapping & tests. |
| [x] | `mlx.core.fft.irfft` | `(a: mlx.core.array, n: typing.Optional[int] = None, axis: int = -1, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] = None) -> mlx.core.array` | The inverse of :func:rfft. | Needs mapping & tests. |
| [x] | `mlx.core.fft.irfft2` | `(a: mlx.core.array, s: typing.Optional[tuple[int, ...]] = None, axes: typing.Optional[collections.abc.Sequence[int]] = [-2, -1], stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Devic...` | The inverse of :func:rfft2. | Needs mapping & tests. |
| [x] | `mlx.core.fft.irfftn` | `(a: mlx.core.array, s: typing.Optional[tuple[int, ...]] = None, axes: typing.Optional[collections.abc.Sequence[int]] = None, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] ...` | The inverse of :func:rfftn. | Needs mapping & tests. |
| [x] | `mlx.core.fft.rfft` | `(a: mlx.core.array, n: typing.Optional[int] = None, axis: int = -1, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] = None) -> mlx.core.array` | One dimensional discrete Fourier Transform on a real input. | Needs mapping & tests. |
| [x] | `mlx.core.fft.rfft2` | `(a: mlx.core.array, s: typing.Optional[tuple[int, ...]] = None, axes: typing.Optional[collections.abc.Sequence[int]] = [-2, -1], stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Devic...` | Two dimensional real discrete Fourier Transform. | Needs mapping & tests. |
| [x] | `mlx.core.fft.rfftn` | `(a: mlx.core.array, s: typing.Optional[tuple[int, ...]] = None, axes: typing.Optional[collections.abc.Sequence[int]] = None, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] ...` | n-dimensional real discrete Fourier Transform. | Needs mapping & tests. |

### Linear Algebra Operations (mlx.core.linalg)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `mlx.core.linalg.cholesky` | `(a: array, upper: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the Cholesky decomposition of a real symmetric positive semi-definite matrix. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.cholesky_inv` | `(L: array, upper: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the inverse of a real symmetric positive semi-definite matrix using it's Cholesky decomposition. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.cross` | `(a: array, b: array, axis: int = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the cross product of two arrays along a specified axis. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.eig` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array]` | Compute the eigenvalues and eigenvectors of a square matrix. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.eigh` | `(a: array, UPLO: str = 'L', *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array]` | Compute the eigenvalues and eigenvectors of a complex Hermitian or | Needs mapping & tests. |
| [x] | `mlx.core.linalg.eigvals` | `(a: mlx.core.array, *, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] = None) -> mlx.core.array` | Compute the eigenvalues of a square matrix. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.eigvalsh` | `(a: mlx.core.array, UPLO: str = 'L', *, stream: typing.Optional[typing.Union[mlx.core.Stream, mlx.core.Device]] = None) -> mlx.core.array` | Compute the eigenvalues of a complex Hermitian or real symmetric matrix. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.inv` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the inverse of a square matrix. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.lu` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array, array]` | Compute the LU factorization of the given matrix A. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.lu_factor` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array]` | Computes a compact representation of the LU factorization. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.norm` | `(a: array, /, ord: Union[None, int, float, str] = None, axis: Union[None, int, list[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Matrix or vector norm. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.pinv` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the (Moore-Penrose) pseudo-inverse of a matrix. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.qr` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array]` | The QR factorization of the input matrix. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.solve` | `(a: array, b: array, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the solution to a system of linear equations AX = B. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.solve_triangular` | `(a: array, b: array, *, upper: bool = False, stream: Union[None, Stream, Device] = None) -> array` | Computes the solution of a triangular system of linear equations AX = B. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.svd` | `(a: array, compute_uv: bool = True, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array, array]` | The Singular Value Decomposition (SVD) of the input matrix. | Needs mapping & tests. |
| [x] | `mlx.core.linalg.tri_inv` | `(a: array, upper: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Compute the inverse of a triangular square matrix. | Needs mapping & tests. |

### Random Operations (mlx.core.random)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `mlx.core.random.bernoulli` | `(p: Union[scalar, array] = 0.5, shape: Optional[Sequence[int]] = None, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array` | Generate Bernoulli random values. | Needs mapping & tests. |
| [x] | `mlx.core.random.categorical` | `(logits: array, axis: int = -1, shape: Optional[Sequence[int]] = None, num_samples: Optional[int] = None, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array` | Sample from a categorical distribution. | Needs mapping & tests. |
| [x] | `mlx.core.random.gumbel` | `(shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, key: Union[None, Stream, Device] = None, stream: Optional[array] = None) -> array` | Sample from the standard Gumbel distribution. | Needs mapping & tests. |
| [x] | `mlx.core.random.key` | `(seed: int) -> mlx.core.array` | Get a PRNG key from a seed. | Needs mapping & tests. |
| [x] | `mlx.core.random.laplace` | `(shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, loc: float = 0.0, scale: float = 1.0, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array` | Sample numbers from a Laplace distribution. | Needs mapping & tests. |
| [x] | `mlx.core.random.multivariate_normal` | `(mean: array, cov: array, shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array` | Generate jointly-normal random samples given a mean and covariance. | Needs mapping & tests. |
| [x] | `mlx.core.random.normal` | `(shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, loc: Union[scalar, array, None] = None, scale: Union[scalar, array, None] = None, key: Optional[array] = None, stream: Union[None, Stre...` | Generate normally distributed random numbers. | Needs mapping & tests. |
| [x] | `mlx.core.random.permutation` | `(x: Union[int, array], axis: int = 0, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array` | Generate a random permutation or permute the entries of an array. | Needs mapping & tests. |
| [x] | `mlx.core.random.randint` | `(low: Union[scalar, array], high: Union[scalar, array], shape: Sequence[int] = [], dtype: Optional[Dtype] = int32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array` | Generate random integers from the given interval. | Needs mapping & tests. |
| [x] | `mlx.core.random.seed` | `(seed: int) -> None` | Seed the global PRNG. | Needs mapping & tests. |
| [x] | `mlx.core.random.state` | `(...)` | Built-in mutable sequence. | Needs mapping & tests. |
| [x] | `mlx.core.random.truncated_normal` | `(lower: Union[scalar, array], upper: Union[scalar, array], shape: Optional[Sequence[int]] = None, dtype: Optional[Dtype] = float32, key: Optional[array] = None, stream: Union[None, Stream, Device] ...` | Generate values from a truncated normal distribution. | Needs mapping & tests. |
| [x] | `mlx.core.random.uniform` | `(low: Union[scalar, array] = 0, high: Union[scalar, array] = 1, shape: Sequence[int] = [], dtype: Optional[Dtype] = float32, key: Optional[array] = None, stream: Union[None, Stream, Device] = None)...` | Generate uniformly distributed random numbers. | Needs mapping & tests. |

### Neural Networks (mlx.nn)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `mlx.nn.ALiBi` | `()` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.nn.AllToShardedLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation such | Needs mapping & tests. |
| [x] | `mlx.nn.AvgPool1d` | `(kernel_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int], NoneType] = None, padding: Union[int, Tuple[int]] = 0)` | Applies 1-dimensional average pooling. | Needs mapping & tests. |
| [x] | `mlx.nn.AvgPool2d` | `(kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int], NoneType] = None, padding: Union[int, Tuple[int, int], NoneType] = 0)` | Applies 2-dimensional average pooling. | Needs mapping & tests. |
| [x] | `mlx.nn.AvgPool3d` | `(kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int], NoneType] = None, padding: Union[int, Tuple[int, int, int], NoneType] = 0)` | Applies 3-dimensional average pooling. | Needs mapping & tests. |
| [x] | `mlx.nn.BatchNorm` | `(num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True)` | Applies Batch Normalization over a 2D or 3D input. | Needs mapping & tests. |
| [x] | `mlx.nn.Bilinear` | `(input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True) -> None` | Applies a bilinear transformation to the inputs. | Needs mapping & tests. |
| [x] | `mlx.nn.CELU` | `(alpha=1.0)` | Applies the Continuously Differentiable Exponential Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.Conv1d` | `(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True)` | Applies a 1-dimensional convolution over the multi-channel input sequence. | Needs mapping & tests. |
| [x] | `mlx.nn.Conv2d` | `(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, groups: int = 1, bias: bool = T...` | Applies a 2-dimensional convolution over the multi-channel input image. | Needs mapping & tests. |
| [x] | `mlx.nn.Conv3d` | `(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, bias: bool = True)` | Applies a 3-dimensional convolution over the multi-channel input image. | Needs mapping & tests. |
| [x] | `mlx.nn.ConvTranspose1d` | `(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, bias: bool = True)` | Applies a 1-dimensional transposed convolution over the multi-channel input sequence. | Needs mapping & tests. |
| [x] | `mlx.nn.ConvTranspose2d` | `(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, output_padding: Union[int, tupl...` | Applies a 2-dimensional transposed convolution over the multi-channel input image. | Needs mapping & tests. |
| [x] | `mlx.nn.ConvTranspose3d` | `(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, output_padding: Union[int, tupl...` | Applies a 3-dimensional transposed convolution over the multi-channel input image. | Needs mapping & tests. |
| [x] | `mlx.nn.Dropout` | `(p: float = 0.5)` | Randomly zero a portion of the elements during training. | Needs mapping & tests. |
| [x] | `mlx.nn.Dropout2d` | `(p: float = 0.5)` | Apply 2D channel-wise dropout during training. | Needs mapping & tests. |
| [x] | `mlx.nn.Dropout3d` | `(p: float = 0.5)` | Apply 3D channel-wise dropout during training. | Needs mapping & tests. |
| [x] | `mlx.nn.ELU` | `(alpha=1.0)` | Applies the Exponential Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.Embedding` | `(num_embeddings: int, dims: int)` | Implements a simple lookup table that maps each input integer to a | Needs mapping & tests. |
| [x] | `mlx.nn.GELU` | `(approx='none')` | Applies the Gaussian Error Linear Units. | Needs mapping & tests. |
| [x] | `mlx.nn.GLU` | `(axis: int = -1)` | Applies the gated linear unit function. | Needs mapping & tests. |
| [x] | `mlx.nn.GRU` | `(input_size: int, hidden_size: int, bias: bool = True)` | A gated recurrent unit (GRU) RNN layer. | Needs mapping & tests. |
| [x] | `mlx.nn.GroupNorm` | `(num_groups: int, dims: int, eps: float = 1e-05, affine: bool = True, pytorch_compatible: bool = False)` | Applies Group Normalization [1] to the inputs. | Needs mapping & tests. |
| [x] | `mlx.nn.HardShrink` | `()` | Applies the HardShrink function. | Needs mapping & tests. |
| [x] | `mlx.nn.HardTanh` | `()` | Applies the HardTanh function. | Needs mapping & tests. |
| [x] | `mlx.nn.Hardswish` | `()` | Applies the hardswish function, element-wise. | Needs mapping & tests. |
| [x] | `mlx.nn.Identity` | `(*args: Any, **kwargs: Any) -> None` | A placeholder identity operator that is argument-insensitive. | Needs mapping & tests. |
| [x] | `mlx.nn.InstanceNorm` | `(dims: int, eps: float = 1e-05, affine: bool = False)` | Applies instance normalization [1] on the inputs. | Needs mapping & tests. |
| [x] | `mlx.nn.LSTM` | `(input_size: int, hidden_size: int, bias: bool = True)` | An LSTM recurrent layer. | Needs mapping & tests. |
| [x] | `mlx.nn.LayerNorm` | `(dims: int, eps: float = 1e-05, affine: bool = True, bias: bool = True)` | Applies layer normalization [1] on the inputs. | Needs mapping & tests. |
| [x] | `mlx.nn.LeakyReLU` | `(negative_slope=0.01)` | Applies the Leaky Rectified Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.Linear` | `(input_dims: int, output_dims: int, bias: bool = True) -> None` | Applies an affine transformation to the input. | Needs mapping & tests. |
| [x] | `mlx.nn.LogSigmoid` | `()` | Applies the Log Sigmoid function. | Needs mapping & tests. |
| [x] | `mlx.nn.LogSoftmax` | `()` | Applies the Log Softmax function. | Needs mapping & tests. |
| [x] | `mlx.nn.MaxPool1d` | `(kernel_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int], NoneType] = None, padding: Union[int, Tuple[int]] = 0)` | Applies 1-dimensional max pooling. | Needs mapping & tests. |
| [x] | `mlx.nn.MaxPool2d` | `(kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int], NoneType] = None, padding: Union[int, Tuple[int, int], NoneType] = 0)` | Applies 2-dimensional max pooling. | Needs mapping & tests. |
| [x] | `mlx.nn.MaxPool3d` | `(kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int], NoneType] = None, padding: Union[int, Tuple[int, int, int], NoneType] = 0)` | Applies 3-dimensional max pooling. | Needs mapping & tests. |
| [x] | `mlx.nn.Mish` | `()` | Applies the Mish function, element-wise. | Needs mapping & tests. |
| [x] | `mlx.nn.Module` | `()` | Base class for building neural networks with MLX. | Needs mapping & tests. |
| [x] | `mlx.nn.MultiHeadAttention` | `(dims: int, num_heads: int, query_input_dims: Optional[int] = None, key_input_dims: Optional[int] = None, value_input_dims: Optional[int] = None, value_dims: Optional[int] = None, value_output_dims...` | Implements the scaled dot product attention with multiple heads. | Needs mapping & tests. |
| [x] | `mlx.nn.PReLU` | `(num_parameters=1, init=0.25)` | Applies the element-wise parametric ReLU. | Needs mapping & tests. |
| [x] | `mlx.nn.QuantizedAllToShardedLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation with | Needs mapping & tests. |
| [x] | `mlx.nn.QuantizedEmbedding` | `(num_embeddings: int, dims: int, group_size: int = 64, bits: int = 4, mode: str = 'affine')` | The same as :obj:Embedding but with a  quantized weight matrix. | Needs mapping & tests. |
| [x] | `mlx.nn.QuantizedLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, mode: str = 'affine')` | Applies an affine transformation to the input using a quantized weight matrix. | Needs mapping & tests. |
| [x] | `mlx.nn.QuantizedShardedToAllLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation using | Needs mapping & tests. |
| [x] | `mlx.nn.RMSNorm` | `(dims: int, eps: float = 1e-05)` | Applies Root Mean Square normalization [1] to the inputs. | Needs mapping & tests. |
| [x] | `mlx.nn.RNN` | `(input_size: int, hidden_size: int, bias: bool = True, nonlinearity: Optional[Callable] = None)` | An Elman recurrent layer. | Needs mapping & tests. |
| [x] | `mlx.nn.ReLU` | `()` | Applies the Rectified Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.ReLU2` | `()` | Applies the ReLU² activation function. | Needs mapping & tests. |
| [x] | `mlx.nn.ReLU6` | `()` | Applies the Rectified Linear Unit 6. | Needs mapping & tests. |
| [x] | `mlx.nn.RoPE` | `(dims: int, traditional: bool = False, base: float = 10000, scale: float = 1.0)` | Implements the rotary positional encoding. | Needs mapping & tests. |
| [x] | `mlx.nn.SELU` | `()` | Applies the Scaled Exponential Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.Sequential` | `(*modules)` | A layer that calls the passed callables in order. | Needs mapping & tests. |
| [x] | `mlx.nn.ShardedToAllLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation and | Needs mapping & tests. |
| [x] | `mlx.nn.SiLU` | `()` | Applies the Sigmoid Linear Unit. Also known as Swish. | Needs mapping & tests. |
| [x] | `mlx.nn.Sigmoid` | `()` | Applies the sigmoid function, element-wise. | Needs mapping & tests. |
| [x] | `mlx.nn.SinusoidalPositionalEncoding` | `(dims: int, min_freq: float = 0.0001, max_freq: float = 1, scale: Optional[float] = None, cos_first: bool = False, full_turns: bool = False)` | Implements sinusoidal positional encoding. | Needs mapping & tests. |
| [x] | `mlx.nn.Softmax` | `()` | Applies the Softmax function. | Needs mapping & tests. |
| [x] | `mlx.nn.Softmin` | `()` | Applies the Softmin function. | Needs mapping & tests. |
| [x] | `mlx.nn.Softplus` | `()` | Applies the Softplus function. | Needs mapping & tests. |
| [x] | `mlx.nn.Softshrink` | `(lambd=0.5)` | Applies the Softshrink function. | Needs mapping & tests. |
| [x] | `mlx.nn.Softsign` | `()` | Applies the Softsign function. | Needs mapping & tests. |
| [x] | `mlx.nn.Step` | `(threshold: float = 0.0)` | Applies the Step Activation Function. | Needs mapping & tests. |
| [x] | `mlx.nn.Tanh` | `()` | Applies the hyperbolic tangent function. | Needs mapping & tests. |
| [x] | `mlx.nn.Transformer` | `(dims: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func...` | Implements a standard Transformer model. | Needs mapping & tests. |
| [x] | `mlx.nn.TransformerDecoder` | `(num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<mlx.gc_func object at 0x1052b3a90>, norm_first: bool = True, checkpoint: bool = False)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.nn.TransformerDecoderLayer` | `(dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func object at 0x1052b3a90>, norm_first: bool = True)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.nn.TransformerEncoder` | `(num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<mlx.gc_func object at 0x1052b3a90>, norm_first: bool = True, checkpoint: bool = False)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.nn.TransformerEncoderLayer` | `(dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func object at 0x1052b3a90>, norm_first: bool = True)` | No documentation available. | Needs mapping & tests. |
| [x] | `mlx.nn.Upsample` | `(scale_factor: Union[float, Tuple], mode: Literal['nearest', 'linear', 'cubic'] = 'nearest', align_corners: bool = False)` | Upsample the input signal spatially. | Needs mapping & tests. |
| [x] | `mlx.nn.average_gradients` | `(gradients: Any, group: Optional[mlx.core.distributed.Group] = None, all_reduce_size: int = 33554432, communication_type: Optional[mlx.core.Dtype] = None, communication_stream: Optional[mlx.core.St...` | Average the gradients across the distributed processes in the passed group. | Needs mapping & tests. |
| [x] | `mlx.nn.celu` | `(x, alpha=1.0)` | Applies the Continuously Differentiable Exponential Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.elu` | `(x, alpha=1.0)` | Applies the Exponential Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.gelu` | `(x) -> mlx.core.array` | Applies the Gaussian Error Linear Units function. | Needs mapping & tests. |
| [x] | `mlx.nn.gelu_approx` | `(x)` | An approximation to Gaussian Error Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.gelu_fast_approx` | `(x)` | A fast approximation to Gaussian Error Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.glu` | `(x: mlx.core.array, axis: int = -1) -> mlx.core.array` | Applies the gated linear unit function. | Needs mapping & tests. |
| [x] | `mlx.nn.hard_shrink` | `(x, lambd=0.5)` | Applies the HardShrink activation function. | Needs mapping & tests. |
| [x] | `mlx.nn.hard_tanh` | `(x, min_val=-1.0, max_val=1.0)` | Applies the HardTanh function. | Needs mapping & tests. |
| [x] | `mlx.nn.hardswish` | `(x)` | Applies the hardswish function, element-wise. | Needs mapping & tests. |
| [x] | `mlx.nn.leaky_relu` | `(x, negative_slope=0.01)` | Applies the Leaky Rectified Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.log_sigmoid` | `(x)` | Applies the Log Sigmoid function. | Needs mapping & tests. |
| [x] | `mlx.nn.log_softmax` | `(x, axis=-1)` | Applies the Log Softmax function. | Needs mapping & tests. |
| [x] | `mlx.nn.mish` | `(x: mlx.core.array) -> mlx.core.array` | Applies the Mish function, element-wise. | Needs mapping & tests. |
| [x] | `mlx.nn.prelu` | `(x: mlx.core.array, alpha: mlx.core.array) -> mlx.core.array` | Applies the element-wise parametric ReLU. | Needs mapping & tests. |
| [x] | `mlx.nn.relu` | `(x)` | Applies the Rectified Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.relu2` | `(x)` | Applies the ReLU² activation function. | Needs mapping & tests. |
| [x] | `mlx.nn.relu6` | `(x)` | Applies the Rectified Linear Unit 6. | Needs mapping & tests. |
| [x] | `mlx.nn.selu` | `(x)` | Applies the Scaled Exponential Linear Unit. | Needs mapping & tests. |
| [x] | `mlx.nn.silu` | `(x)` | Applies the Sigmoid Linear Unit. Also known as Swish. | Needs mapping & tests. |
| [x] | `mlx.nn.softmin` | `(x, axis=-1)` | Applies the Softmin function. | Needs mapping & tests. |
| [x] | `mlx.nn.softplus` | `(x)` | Applies the Softplus function. | Needs mapping & tests. |
| [x] | `mlx.nn.softshrink` | `(x, lambd: float = 0.5)` | Applies the Softshrink activation function. | Needs mapping & tests. |
| [x] | `mlx.nn.softsign` | `(x)` | Applies the Softsign function. | Needs mapping & tests. |
| [x] | `mlx.nn.step` | `(x: mlx.core.array, threshold: float = 0.0)` | Applies the Step Activation Function. | Needs mapping & tests. |

### Optimizers (mlx.optimizers)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `mlx.optimizers.AdaDelta` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], rho: float = 0.9, eps: float = 1e-06)` | The AdaDelta optimizer with a learning rate [1]. | Needs mapping & tests. |
| [x] | `mlx.optimizers.Adafactor` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array], NoneType] = None, eps: Tuple[float, float] = (1e-30, 0.001), clip_threshold: float = 1.0, decay_rate: float = -0.8, beta_1: ...` | The Adafactor optimizer. | Needs mapping & tests. |
| [x] | `mlx.optimizers.Adagrad` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], eps: float = 1e-08)` | The Adagrad optimizer [1]. | Needs mapping & tests. |
| [x] | `mlx.optimizers.Adam` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08, bias_correction: bool = False)` | The Adam optimizer [1]. In detail, | Needs mapping & tests. |
| [x] | `mlx.optimizers.AdamW` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08, weight_decay: float = 0.01, bias_correction: bool = False)` | The AdamW optimizer [1]. We update the weights with a weight_decay | Needs mapping & tests. |
| [x] | `mlx.optimizers.Adamax` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08)` | The Adamax optimizer, a variant of Adam based on the infinity norm [1]. | Needs mapping & tests. |
| [x] | `mlx.optimizers.Lion` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.99], weight_decay: float = 0.0)` | The Lion optimizer [1]. | Needs mapping & tests. |
| [x] | `mlx.optimizers.MultiOptimizer` | `(optimizers, filters: list = [])` | Wraps a list of optimizers with corresponding weight predicates/filters | Needs mapping & tests. |
| [x] | `mlx.optimizers.Muon` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], momentum: float = 0.95, weight_decay: float = 0.01, nesterov: bool = True, ns_steps: int = 5)` | The Muon optimizer. | Needs mapping & tests. |
| [x] | `mlx.optimizers.Optimizer` | `(schedulers=None)` | The base class for all optimizers. It allows us to implement an | Needs mapping & tests. |
| [x] | `mlx.optimizers.RMSprop` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], alpha: float = 0.99, eps: float = 1e-08)` | The RMSprop optimizer [1]. | Needs mapping & tests. |
| [x] | `mlx.optimizers.SGD` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False)` | The stochastic gradient descent optimizer. | Needs mapping & tests. |
| [x] | `mlx.optimizers.clip_grad_norm` | `(grads, max_norm)` | Clips the global norm of the gradients. | Needs mapping & tests. |
| [x] | `mlx.optimizers.cosine_decay` | `(init: float, decay_steps: int, end: float = 0.0) -> Callable` | Make a cosine decay scheduler. | Needs mapping & tests. |
| [x] | `mlx.optimizers.exponential_decay` | `(init: float, decay_rate: float) -> Callable` | Make an exponential decay scheduler. | Needs mapping & tests. |
| [x] | `mlx.optimizers.join_schedules` | `(schedules: List[Callable], boundaries: List[int]) -> Callable` | Join multiple schedules to create a new schedule. | Needs mapping & tests. |
| [x] | `mlx.optimizers.linear_schedule` | `(init: float, end: float, steps: int) -> Callable` | Make a linear scheduler. | Needs mapping & tests. |
| [x] | `mlx.optimizers.step_decay` | `(init: float, decay_rate: float, step_size: int) -> Callable` | Make a step decay scheduler. | Needs mapping & tests. |
| [x] | `mlx.optimizers.tree_flatten` | `(tree: Any, prefix: str = '', is_leaf: Optional[Callable] = None, destination: Union[List[Tuple[str, Any]], Dict[str, Any], NoneType] = None) -> Union[List[Tuple[str, Any]], Dict[str, Any]]` | Flattens a Python tree to a list of key, value tuples. | Needs mapping & tests. |
| [x] | `mlx.optimizers.tree_map` | `(fn: Callable, tree: Any, *rest: Any, is_leaf: Optional[Callable] = None) -> Any` | Applies fn to the leaves of the Python tree tree and | Needs mapping & tests. |
| [x] | `mlx.optimizers.tree_merge` | `(tree_a, tree_b, merge_fn=None)` | Merge two Python trees in one containing the values of both. It can be | Needs mapping & tests. |
| [x] | `mlx.optimizers.tree_reduce` | `(fn, tree, initializer=None, is_leaf=None)` | Applies a reduction to the leaves of a Python tree. | Needs mapping & tests. |
| [x] | `mlx.optimizers.tree_unflatten` | `(tree: Union[List[Tuple[str, Any]], Dict[str, Any]]) -> Any` | Recreate a Python tree from its flat representation. | Needs mapping & tests. |

### Utils (mlx.utils)
| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `mlx.utils.defaultdict` | `(default_factory=None, /, [...]) --> dict with default factory` | The default factory is called without arguments to produce | Needs mapping & tests. |
| [x] | `mlx.utils.tree_map_with_path` | `(fn: Callable, tree: Any, *rest: Any, is_leaf: Optional[Callable] = None, path: Optional[Any] = None) -> Any` | Applies fn to the path and leaves of the Python tree tree and | Needs mapping & tests. |
| [x] | `mlx.utils.zip_longest` | `(iter1 [,iter2 [...]], [fillvalue=None]) --> zip_longest object` | Return a zip_longest object whose .__next__() method returns a tuple where | Needs mapping & tests. |
