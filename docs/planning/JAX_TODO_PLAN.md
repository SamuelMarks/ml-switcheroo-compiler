# `ml-switcheroo-compiler` JAX Parity Implementation Plan

To ensure `zero-jax` can pass 100% of the official JAX test suite semantically and syntactically, `ml-switcheroo-compiler` must implement the following operations, matching their expected inputs and behaviors.

## Tensor Operations (`jax.numpy` / `jax.lax` bindings)

| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `sin` | `sin(x: Any) -> Any` | Compute the trigonometric sine element-wise. | |
| [x] | `cos` | `cos(x: Any) -> Any` | Compute the trigonometric cosine element-wise. | |
| [x] | `exp` | `exp(x: Any) -> Any` | Calculate the exponential of all elements in the input array. | |
| [x] | `log` | `log(x: Any) -> Any` | Natural logarithm, element-wise. | |
| [x] | `transpose` | `transpose(x: Any, axes: Optional[List[int]]=None) -> Any` | Reverse or permute the axes of an array. | |
| [x] | `reshape` | `reshape(x: Any, newshape: Tuple[int, ...]) -> Any` | Gives a new shape to an array without changing its data. | |
| [x] | `broadcast_to` | `broadcast_to(x: Any, shape: Tuple[int, ...]) -> Any` | Broadcast an array to a new shape. | |
| [x] | `concatenate` | `concatenate(arrays: List[Any], axis: int=0) -> Any` | Join a sequence of arrays along an existing axis. | |
| [x] | `where` | `where(condition: Any, x: Any, y: Any) -> Any` | Return elements chosen from x or y depending on condition. | |
| [x] | `einsum` | `einsum(subscripts: str, *operands: Any) -> Any` | Evaluates the Einstein summation convention on the operands. | |
| [x] | `add` | `add(x: Any, y: Any) -> Any` | Add arguments element-wise. | |
| [x] | `multiply` | `multiply(x: Any, y: Any) -> Any` | Multiply arguments element-wise. | |
| [x] | `power` | `power(x: Any, y: Any) -> Any` | First array elements raised to powers from second array, element-wise. | |
| [x] | `maximum` | `maximum(x: Any, y: Any) -> Any` | Element-wise maximum of array elements. | |
| [x] | `minimum` | `minimum(x: Any, y: Any) -> Any` | Element-wise minimum of array elements. | |
| [ ] | `clip` | `clip(a: Any, a_min: Any, a_max: Any) -> Any` | Clip (limit) the values in an array. | |
| [x] | `max` | `max(x: Any, axis: Any=None, keepdims: bool=False, where: Any=None, initial: Any=None) -> Any` | Return the maximum of an array or maximum along an axis. | |
| [x] | `sum` | `sum(x: Any, axis: Any=None, keepdims: bool=False, where: Any=None) -> Any` | Sum of array elements over a given axis. | |
| [x] | `zeros_like` | `zeros_like(x: Any, dtype: Any=None) -> Any` | Return an array of zeros with the same shape and type as a given array. | |
| [x] | `zeros` | `zeros(shape: Any, dtype: Any=None) -> Any` | Return a new array of given shape and type, filled with zeros. | |
| [x] | `abs` | `abs(x: Any) -> Any` | Calculate the absolute value element-wise. | |
| [x] | `mean` | `mean(x: Any, axis: Any=None, keepdims: bool=False) -> Any` | Compute the arithmetic mean along the specified axis. | |
| [ ] | `array` | `array(x: Any, dtype: Any=None) -> Any` | Create an array. | |
| [x] | `dot` | `dot(a: Any, b: Any) -> Any` | Dot product of two arrays. | |
| [x] | `matmul` | `matmul(a: Any, b: Any) -> Any` | Matrix product of two arrays. | |
| [ ] | `expand_dims` | `expand_dims(a: Any, axis: int) -> Any` | Expand the shape of an array. | |
| [x] | `isfinite` | `isfinite(x: Any) -> Any` | Test element-wise for finiteness (not infinity or not Not a Number). | |
| [x] | `allclose` | `allclose(a: Any, b: Any, rtol: Any=1e-05, atol: Any=1e-08, equal_nan: Any=False) -> Any` | Returns True if two arrays are element-wise equal within a tolerance. | |
| [ ] | `array_equal` | `array_equal(a1: Any, a2: Any, equal_nan: Any=False) -> Any` | True if two arrays have the same shape and elements, False otherwise. | |
| [ ] | `broadcast_shapes` | `broadcast_shapes(*shapes: Any) -> Any` | Broadcast the input shapes into a single shape. | |
| [x] | `ones` | `ones(shape: Any, dtype: Any=None) -> Any` | Return a new array of given shape and type, filled with ones. | |
| [x] | `empty` | `empty(shape: Any, dtype: Any=None) -> Any` | Return a new array of given shape and type, without initializing entries. | |
| [x] | `full` | `full(shape: Any, fill_value: Any, dtype: Any=None) -> Any` | Return a new array of given shape and type, filled with fill_value. | |
| [x] | `ones_like` | `ones_like(x: Any, dtype: Any=None) -> Any` | Return an array of ones with the same shape and type as a given array. | |
| [ ] | `empty_like` | `empty_like(x: Any, dtype: Any=None) -> Any` | Return a new array with the same shape and type as a given array. | |
| [x] | `full_like` | `full_like(x: Any, fill_value: Any, dtype: Any=None) -> Any` | Return a full array with the same shape and type as a given array. | |
| [ ] | `asarray` | `asarray(x: Any, dtype: Any=None) -> Any` | Convert the input to an array. | |
| [x] | `arange` | `arange(start: Any, stop: Any=None, step: Any=1, dtype: Any=None) -> Any` | Return evenly spaced values within a given interval. | |
| [x] | `linspace` | `linspace(start: Any, stop: Any, num: int=50, endpoint: bool=True, retstep: bool=False, dtype: Any=None, axis: int=0) -> Any` | Return evenly spaced numbers over a specified interval. | |
| [ ] | `logspace` | `logspace(start: Any, stop: Any, num: int=50, endpoint: bool=True, base: float=10.0, dtype: Any=None, axis: int=0) -> Any` | Return numbers spaced evenly on a log scale. | |
| [x] | `eye` | `eye(N: int, M: int=None, k: int=0, dtype: Any=None) -> Any` | Return a 2-D array with ones on the diagonal and zeros elsewhere. | |
| [x] | `identity` | `identity(n: int, dtype: Any=None) -> Any` | Return the identity array. | |
| [x] | `meshgrid` | `meshgrid(*xi: Any, copy: Any=True, sparse: Any=False, indexing: Any='xy') -> Any` | Return coordinate matrices from coordinate vectors. | |
| [x] | `subtract` | `subtract(x: Any, y: Any) -> Any` | Subtract arguments, element-wise. | |
| [x] | `divide` | `divide(x: Any, y: Any) -> Any` | Divide arguments element-wise. | |
| [ ] | `true_divide` | `true_divide(x: Any, y: Any) -> Any` | Divide arguments element-wise. | |
| [x] | `floor_divide` | `floor_divide(x: Any, y: Any) -> Any` | Return the largest integer smaller or equal to the division of the inputs. | |
| [x] | `mod` | `mod(x: Any, y: Any) -> Any` | Return element-wise remainder of division. | |
| [x] | `remainder` | `remainder(x: Any, y: Any) -> Any` | Return element-wise remainder of division. | |
| [x] | `divmod` | `divmod(x: Any, y: Any) -> Any` | Return element-wise quotient and remainder simultaneously. | |
| [x] | `negative` | `negative(x: Any) -> Any` | Numerical negative, element-wise. | |
| [x] | `positive` | `positive(x: Any) -> Any` | Numerical positive, element-wise. | |
| [x] | `sign` | `sign(x: Any) -> Any` | Returns an element-wise indication of the sign of a number. | |
| [x] | `floor` | `floor(x: Any) -> Any` | Return the floor of the input, element-wise. | |
| [x] | `ceil` | `ceil(x: Any) -> Any` | Return the ceiling of the input, element-wise. | |
| [x] | `trunc` | `trunc(x: Any) -> Any` | Return the truncated value of the input, element-wise. | |
| [ ] | `rint` | `rint(x: Any) -> Any` | Round elements of the array to the nearest integer. | |
| [x] | `tan` | `tan(x: Any) -> Any` | Compute tangent element-wise. | |
| [ ] | `arcsin` | `arcsin(x: Any) -> Any` | Inverse sine, element-wise. | |
| [ ] | `arccos` | `arccos(x: Any) -> Any` | Trigonometric inverse cosine, element-wise. | |
| [ ] | `arctan` | `arctan(x: Any) -> Any` | Trigonometric inverse tangent, element-wise. | |
| [ ] | `arctan2` | `arctan2(x1: Any, x2: Any) -> Any` | Element-wise arc tangent of x1/x2 choosing the quadrant correctly. | |
| [x] | `sinh` | `sinh(x: Any) -> Any` | Hyperbolic sine, element-wise. | |
| [x] | `cosh` | `cosh(x: Any) -> Any` | Hyperbolic cosine, element-wise. | |
| [x] | `tanh` | `tanh(x: Any) -> Any` | Compute hyperbolic tangent element-wise. | |
| [ ] | `arcsinh` | `arcsinh(x: Any) -> Any` | Inverse hyperbolic sine element-wise. | |
| [ ] | `arccosh` | `arccosh(x: Any) -> Any` | Inverse hyperbolic cosine, element-wise. | |
| [ ] | `arctanh` | `arctanh(x: Any) -> Any` | Inverse hyperbolic tangent element-wise. | |
| [x] | `exp2` | `exp2(x: Any) -> Any` | Calculate 2**p for all p in the input array. | |
| [x] | `expm1` | `expm1(x: Any) -> Any` | Calculate exp(x) - 1 for all elements in the array. | |
| [x] | `log2` | `log2(x: Any) -> Any` | Base-2 logarithm of x. | |
| [x] | `log10` | `log10(x: Any) -> Any` | Return the base 10 logarithm of the input array, element-wise. | |
| [x] | `log1p` | `log1p(x: Any) -> Any` | Return the natural logarithm of one plus the input array, element-wise. | |
| [x] | `prod` | `prod(a: Any, axis: Any=None, dtype: Any=None, keepdims: bool=False) -> Any` | Return the product of array elements over a given axis. | |
| [x] | `min` | `min(a: Any, axis: Any=None, keepdims: bool=False) -> Any` | Return the minimum of an array or minimum along an axis. | |
| [ ] | `amin` | `amin(a: Any, axis: Any=None, keepdims: bool=False) -> Any` | Return the minimum of an array or minimum along an axis. | |
| [ ] | `amax` | `amax(a: Any, axis: Any=None, keepdims: bool=False) -> Any` | Return the maximum of an array or maximum along an axis. | |
| [x] | `argmax` | `argmax(a: Any, axis: Any=None, keepdims: bool=False) -> Any` | Returns the indices of the maximum values along an axis. | |
| [x] | `argmin` | `argmin(a: Any, axis: Any=None, keepdims: bool=False) -> Any` | Returns the indices of the minimum values along an axis. | |
| [x] | `any` | `any(a: Any, axis: Any=None, keepdims: bool=False) -> Any` | Test whether any array element along a given axis evaluates to True. | |
| [x] | `all` | `all(a: Any, axis: Any=None, keepdims: bool=False) -> Any` | Test whether all array elements along a given axis evaluate to True. | |
| [ ] | `var` | `var(a: Any, axis: Any=None, dtype: Any=None, keepdims: bool=False, ddof: int=0) -> Any` | Compute the variance along the specified axis. | |
| [x] | `std` | `std(a: Any, axis: Any=None, dtype: Any=None, keepdims: bool=False, ddof: int=0) -> Any` | Compute the standard deviation along the specified axis. | |
| [x] | `ravel` | `ravel(a: Any, order: str='C') -> Any` | Return a contiguous flattened array. | |
| [x] | `squeeze` | `squeeze(a: Any, axis: Any=None) -> Any` | Remove axes of length one from a. | |
| [x] | `swapaxes` | `swapaxes(a: Any, axis1: int, axis2: int) -> Any` | Interchange two axes of an array. | |
| [x] | `moveaxis` | `moveaxis(a: Any, source: Any, destination: Any) -> Any` | Move axes of an array to new positions. | |
| [x] | `stack` | `stack(arrays: Any, axis: int=0) -> Any` | Join a sequence of arrays along a new axis. | |
| [ ] | `vstack` | `vstack(tup: Any) -> Any` | Stack arrays in sequence vertically (row wise). | |
| [ ] | `hstack` | `hstack(tup: Any) -> Any` | Stack arrays in sequence horizontally (column wise). | |
| [ ] | `dstack` | `dstack(tup: Any) -> Any` | Stack arrays in sequence depth wise (along third axis). | |
| [x] | `split` | `split(ary: Any, indices_or_sections: Any, axis: int=0) -> Any` | Split an array into multiple sub-arrays as views into ary. | |
| [ ] | `array_split` | `array_split(ary: Any, indices_or_sections: Any, axis: int=0) -> Any` | Split an array into multiple sub-arrays. | |
| [ ] | `vsplit` | `vsplit(ary: Any, indices_or_sections: Any) -> Any` | Split an array into multiple sub-arrays vertically (row-wise). | |
| [ ] | `hsplit` | `hsplit(ary: Any, indices_or_sections: Any) -> Any` | Split an array into multiple sub-arrays horizontally (column-wise). | |
| [ ] | `dsplit` | `dsplit(ary: Any, indices_or_sections: Any) -> Any` | Split array into multiple sub-arrays along the 3rd axis (depth). | |
| [x] | `tile` | `tile(A: Any, reps: Any) -> Any` | Construct an array by repeating A the number of times given by reps. | |
| [x] | `repeat` | `repeat(a: Any, repeats: Any, axis: Any=None) -> Any` | Repeat elements of an array. | |
| [x] | `pad` | `pad(array: Any, pad_width: Any, mode: str='constant', **kwargs: Any) -> Any` | Pad an array. | |
| [x] | `take` | `take(a: Any, indices: Any, axis: int=None, mode: str=None) -> Any` | Take elements from an array along an axis. | |
| [x] | `take_along_axis` | `take_along_axis(arr: Any, indices: Any, axis: int) -> Any` | Take values from the input array by matching 1d index and data slices. | |
| [x] | `vdot` | `vdot(a: Any, b: Any) -> Any` | Return the dot product of two vectors. | |
| [x] | `inner` | `inner(a: Any, b: Any) -> Any` | Inner product of two arrays. | |
| [x] | `outer` | `outer(a: Any, b: Any) -> Any` | Compute the outer product of two vectors. | |
| [x] | `tensordot` | `tensordot(a: Any, b: Any, axes: Any=2) -> Any` | Compute tensor dot product along specified axes. | |
| [x] | `shape` | `shape(a: Any) -> Any` | Get the shape of the array. | |
| [x] | `sqrt` | `sqrt(x: Any) -> Any` | Return the non-negative square-root of an array, element-wise. | |
| [x] | `square` | `square(x: Any) -> Any` | Return the element-wise square of the input. | |
| [x] | `isnan` | `isnan(x: Any) -> Any` | Test element-wise for NaN and return result as a boolean array. | |
| [x] | `cumsum` | `cumsum(a: Any, axis: Any=None, dtype: Any=None) -> Any` | Return the cumulative sum of the elements along a given axis. | |
| [x] | `sub` | `sub(x: Any, y: Any) -> Any` | Elementwise subtraction. | |
| [x] | `mul` | `mul(x: Any, y: Any) -> Any` | Elementwise multiplication. | |
| [x] | `div` | `div(x: Any, y: Any) -> Any` | Elementwise division. | |
| [ ] | `broadcast` | `broadcast(x: Any, sizes: Any) -> Any` | Broadcasts an array by adding new leading dimensions. | |
| [ ] | `broadcast_in_dim` | `broadcast_in_dim(x: Any, shape: Any, broadcast_dimensions: Any) -> Any` | Broadcasts an array to a specified shape by mapping existing dimensions. | |
| [x] | `slice` | `slice(operand: Any, start_indices: Any, limit_indices: Any, strides: Any=None) -> Any` | Extracts a slice from an array. | |
| [x] | `dynamic_slice` | `dynamic_slice(operand: Any, start_indices: Any, slice_sizes: Any) -> Any` | Extracts a dynamic slice from an array. | |
| [ ] | `dynamic_update_slice` | `dynamic_update_slice(operand: Any, update: Any, start_indices: Any) -> Any` | Updates a dynamic slice of an array. | |
| [ ] | `reduce` | `reduce(operand: Any, init_value: Any, computation: Any, dimensions: Any) -> Any` | Reduces an array along specified dimensions. | |
| [ ] | `select` | `select(pred: Any, on_true: Any, on_false: Any) -> Any` | Elementwise selection based on a predicate. | |
| [ ] | `clamp` | `clamp(min_val: Any, x: Any, max_val: Any) -> Any` | Clamps the values of an array to a specified range. | |

## Control Flow (`jax.lax` control flow)

| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `cond` | `cond(pred: Any, true_fn: Callable, false_fn: Callable, *operands: Any) -> Any` | Conditionally applies one of two functions. | |
| [ ] | `scan` | `scan(f: Callable, init: Any, xs: Any, length: int=None) -> Any` | Scans a function over leading array axes while carrying along state. | |
| [ ] | `stop_gradient` | `stop_gradient(x: Any) -> Any` | Stops the flow of gradients during reverse-mode differentiation. | |

## Random Number Generation (`jax.random`)

| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `split` | `split(key: Any, num: int=2) -> Any` | Splits a PRNG key into `num` new keys. | |
| [ ] | `fold_in` | `fold_in(key: Any, data: Any) -> Any` | Folds in data to a PRNG key to derive a new key. | |
| [x] | `PRNGKey` | `PRNGKey(seed: int) -> Any` | Creates a PRNG key given an integer seed. | |
| [ ] | `uniform` | `uniform(key: Any, shape: Any, dtype: Any=None, minval: float=0.0, maxval: float=1.0) -> Any` | Samples uniform random values from a given key. | |
| [ ] | `normal` | `normal(key: Any, shape: Any, dtype: Any=None) -> Any` | Samples standard normal random values from a given key. | |
| [ ] | `randint` | `randint(key: Any, shape: Any, minval: int, maxval: int, dtype: Any=None) -> Any` | Samples uniform random integers from a given key. | |
| [ ] | `bernoulli` | `bernoulli(key: Any, p: float=0.5, shape: Any=None) -> Any` | Samples Bernoulli random variables from a given key. | |
| [ ] | `categorical` | `categorical(key: Any, logits: Any, axis: int=-1, shape: Any=None) -> Any` | Samples categorical random variables from a given key. | |
| [ ] | `permutation` | `permutation(key: Any, x: Any, axis: int=0, independent: bool=False) -> Any` | Randomly permutes a sequence or array. | |
| [ ] | `choice` | `choice(key: Any, a: Any, shape: Any=(), replace: bool=True, p: Any=None, axis: int=0) -> Any` | Generates a random sample from a given 1-D array. | |

## Transformations & API (`jax.api`)

| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `jit` | `jit(fun: Callable) -> Callable` | Compiles a function to execute faster, in our parity layer this currently acts as an eager wrapper. | |
| [x] | `grad` | `grad(fun: Callable, argnums: Any=0) -> Callable` | Creates a function that evaluates the gradient of fun. | |
| [ ] | `value_and_grad` | `value_and_grad(fun: Callable, argnums: Any=0) -> Callable` | Creates a function that evaluates both the value and gradient of fun. | |
| [ ] | `vmap` | `vmap(fun: Callable) -> Callable` | Vectorizing map. Creates a function which maps fun over argument axes. | |
| [ ] | `disable_jit` | `@contextlib.contextmanager` | A context manager to temporarily disable JIT compilation. | |
| [ ] | `pmap` | `pmap(fun: Any, axis_name: Any=None, in_axes: Any=0, out_axes: Any=0, static_broadcasted_argnums: Any=(), devices: Any=None, backend: Any=None, axis_size: Any=None, donate_argnums: Any=(), global_arg_shapes: Any=None) -> Any` | Parallel map. Creates a function which evaluates fun in parallel on multiple XLA devices. | |
| [ ] | `eval_shape` | `eval_shape(fun: Callable, *args: Any, **kwargs: Any) -> Any` | Evaluates the shape and dtype of the output of fun without computing its values. | |

## Neural Network Primitives (`jax.nn`)

| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `gelu` | `gelu(x: ArrayLike, approximate: bool=False) -> Any` | Computes the Gaussian Error Linear Unit (GELU) activation function. | |
| [x] | `logsumexp` | `logsumexp(a: ArrayLike, axis: Any=None, b: Optional[ArrayLike]=None, keepdims: bool=False, return_sign: bool=False, where: Optional[ArrayLike]=None) -> Any` | Computes the log of the sum of exponentials of input elements. | |
| [ ] | `one_hot` | `one_hot(x: Any, num_classes: int, *, dtype: Any=float, axis: Any=-1) -> Any` | Creates a one-hot encoding of the given integer array. | |
| [x] | `softmax` | `softmax(x: ArrayLike, axis: Any=-1, where: Optional[Any]=None, initial: Any=None) -> Any` | Computes the softmax activation function over the given axis. | |
| [x] | `sigmoid` | `sigmoid(x: Any) -> Any` | Computes the sigmoid activation function. | |
| [ ] | `log_sigmoid` | `log_sigmoid(x: Any) -> Any` | Computes the logarithm of the sigmoid function. | |
| [x] | `relu` | `relu(x: ArrayLike) -> Any` | Computes the Rectified Linear Unit (ReLU) activation function. | |
| [ ] | `relu6` | `relu6(x: ArrayLike) -> Any` | Computes the ReLU6 activation function, capping at 6. | |
| [ ] | `hard_sigmoid` | `hard_sigmoid(x: ArrayLike) -> Any` | Computes the hard sigmoid activation function. | |
| [ ] | `hard_tanh` | `hard_tanh(x: ArrayLike) -> Any` | Computes the hard tanh activation function, bounding the input between -1 and 1. | |
| [x] | `swish` | `swish(x: ArrayLike) -> Any` | Computes the Swish activation function (x * sigmoid(x)). | |
| [ ] | `silu` | `silu(x: ArrayLike) -> Any` | Computes the SiLU (Sigmoid Linear Unit) activation function, which is identical to Swish. | |
| [x] | `elu` | `elu(x: ArrayLike, alpha: float=1.0) -> Any` | Computes the Exponential Linear Unit (ELU) activation function. | |
| [x] | `celu` | `celu(x: ArrayLike, alpha: float=1.0) -> Any` | Computes the Continuously Differentiable Exponential Linear Unit (CELU) activation function. | |
| [x] | `selu` | `selu(x: ArrayLike) -> Any` | Computes the Scaled Exponential Linear Unit (SELU) activation function. | |
| [x] | `log_softmax` | `log_softmax(x: ArrayLike, axis: int=-1) -> Any` | Computes the logarithm of the softmax activation function. | |
| [x] | `zeros` | `zeros(key: KeyArray, shape: Shape, dtype: Any=float) -> Array` | Initializes an array with all zeros. | |
| [x] | `ones` | `ones(key: KeyArray, shape: Shape, dtype: Any=float) -> Array` | Initializes an array with all ones. | |
| [x] | `constant` | `constant(value: RealNumeric, dtype: Any=float) -> Initializer` | Returns an initializer that generates arrays filled with a constant value. | |
| [ ] | `uniform` | `uniform(scale: RealNumeric=0.01, dtype: Any=float) -> Initializer` | Returns an initializer that generates arrays from a uniform distribution. | |
| [ ] | `normal` | `normal(stddev: RealNumeric=0.01, dtype: Any=float) -> Initializer` | Returns an initializer that generates arrays from a normal distribution. | |
| [ ] | `truncated_normal` | `truncated_normal(stddev: RealNumeric=0.01, dtype: Any=float, lower: RealNumeric=-2.0, upper: RealNumeric=2.0) -> Initializer` | Returns an initializer that generates arrays from a truncated normal distribution. | |
| [ ] | `variance_scaling` | `variance_scaling(scale: RealNumeric, mode: str, distribution: str, in_axis: Union[int, Sequence[int]]=-2, out_axis: Union[int, Sequence[int]]=-1, batch_axis: Sequence[int]=(), dtype: Any=float) -> Initializer` | Returns an initializer that scales its variance based on weight shape. | |
| [ ] | `glorot_uniform` | `glorot_uniform(in_axis: Union[int, Sequence[int]]=-2, out_axis: Union[int, Sequence[int]]=-1, batch_axis: Sequence[int]=(), dtype: Any=float) -> Initializer` | Returns an initializer for the Glorot (Xavier) uniform initialization. | |
| [ ] | `glorot_normal` | `glorot_normal(in_axis: Union[int, Sequence[int]]=-2, out_axis: Union[int, Sequence[int]]=-1, batch_axis: Sequence[int]=(), dtype: Any=float) -> Initializer` | Returns an initializer for the Glorot (Xavier) normal initialization. | |
| [ ] | `lecun_uniform` | `lecun_uniform(in_axis: Union[int, Sequence[int]]=-2, out_axis: Union[int, Sequence[int]]=-1, batch_axis: Sequence[int]=(), dtype: Any=float) -> Initializer` | Returns an initializer for the LeCun uniform initialization. | |
| [ ] | `lecun_normal` | `lecun_normal(in_axis: Union[int, Sequence[int]]=-2, out_axis: Union[int, Sequence[int]]=-1, batch_axis: Sequence[int]=(), dtype: Any=float) -> Initializer` | Returns an initializer for the LeCun normal initialization. | |
| [ ] | `he_uniform` | `he_uniform(in_axis: Union[int, Sequence[int]]=-2, out_axis: Union[int, Sequence[int]]=-1, batch_axis: Sequence[int]=(), dtype: Any=float) -> Initializer` | Returns an initializer for the He (Kaiming) uniform initialization. | |
| [ ] | `he_normal` | `he_normal(in_axis: Union[int, Sequence[int]]=-2, out_axis: Union[int, Sequence[int]]=-1, batch_axis: Sequence[int]=(), dtype: Any=float) -> Initializer` | Returns an initializer for the He (Kaiming) normal initialization. | |
| [ ] | `orthogonal` | `orthogonal(scale: RealNumeric=1.0, column_axis: int=-1, dtype: Any=float) -> Initializer` | Returns an initializer that generates orthogonally initialized weight arrays. | |
| [ ] | `delta_orthogonal` | `delta_orthogonal(scale: RealNumeric=1.0, column_axis: int=-1, dtype: Any=float) -> Initializer` | Returns an initializer that generates delta orthogonal arrays (useful for CNNs). | |

## Compiler Infrastructure Requirements

| Status | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `LogicalNode` | `class LogicalNode(id, op_type, inputs, ...)` | IR/AST Representation for all operations. | Must maintain full graph lineage. |
| [x] | `ProxyTensor` | `class ProxyTensor(id, shape, dtype)` | Abstract tensor proxy for shape/dtype evaluation. | Required for `jax.eval_shape` without execution. |
| [x] | `evaluate_graph` | `evaluate_graph(graph, inputs)` | JIT compilation and graph evaluation pipeline. | Should cache compiled kernels. |
| [ ] | `Tracing Context` | `_tracer.start_tracing()`, `_tracer.stop_tracing()` | Context lifecycle management. | Captures dynamic shapes effectively. |
| [x] | `EagerMode` | `with EagerMode(): ...` | Seamless fallback for immediate execution. | Required for Python-level control flow. |

## Internal Compiler Bindings Required (`ml_switcheroo.*`)
The above JAX primitives map to the following low-level compiler bindings that must be implemented in `ml-switcheroo-compiler`:

### `ml_switcheroo.ops`
| Status | Binding | Notes |
|---|---|---|
| [x] | `ops.abs` | |
| [x] | `ops.acos` | |
| [x] | `ops.acosh` | |
| [x] | `ops.add` | |
| [x] | `ops.all` | |
| [x] | `ops.allclose` | |
| [x] | `ops.any` | |
| [x] | `ops.arange` | |
| [x] | `ops.argmax` | |
| [x] | `ops.argmin` | |
| [ ] | `ops.array_split` | |
| [x] | `ops.asin` | |
| [x] | `ops.asinh` | |
| [x] | `ops.atan` | |
| [x] | `ops.atan2` | |
| [x] | `ops.atanh` | |
| [x] | `ops.broadcast_to` | |
| [x] | `ops.cast` | |
| [x] | `ops.ceil` | |
| [x] | `ops.concatenate` | |
| [x] | `ops.cos` | |
| [x] | `ops.cosh` | |
| [x] | `ops.cumsum` | |
| [x] | `ops.divide` | |
| [x] | `ops.divmod` | |
| [x] | `ops.dot` | |
| [ ] | `ops.dsplit` | |
| [ ] | `ops.dstack` | |
| [x] | `ops.dynamic_slice` | |
| [x] | `ops.einsum` | |
| [x] | `ops.empty` | |
| [x] | `ops.equal` | |
| [x] | `ops.erf` | |
| [x] | `ops.exp` | |
| [x] | `ops.eye` | |
| [x] | `ops.floor` | |
| [x] | `ops.floor_divide` | |
| [x] | `ops.full` | |
| [x] | `ops.full_like` | |
| [x] | `ops.greater` | |
| [x] | `ops.greater_equal` | |
| [ ] | `ops.hsplit` | |
| [ ] | `ops.hstack` | |
| [x] | `ops.identity` | |
| [x] | `ops.inner` | |
| [x] | `ops.isfinite` | |
| [x] | `ops.isnan` | |
| [x] | `ops.less` | |
| [x] | `ops.less_equal` | |
| [x] | `ops.linspace` | |
| [x] | `ops.log` | |
| [x] | `ops.matmul` | |
| [x] | `ops.max` | |
| [x] | `ops.maximum` | |
| [x] | `ops.mean` | |
| [x] | `ops.min` | |
| [x] | `ops.minimum` | |
| [x] | `ops.mod` | |
| [x] | `ops.moveaxis` | |
| [x] | `ops.multiply` | |
| [x] | `ops.negative` | |
| [x] | `ops.ones` | |
| [x] | `ops.ones_like` | |
| [x] | `ops.outer` | |
| [x] | `ops.pad` | |
| [x] | `ops.permute` | |
| [x] | `ops.positive` | |
| [x] | `ops.power` | |
| [x] | `ops.prod` | |
| [x] | `ops.remainder` | |
| [x] | `ops.repeat` | |
| [x] | `ops.reshape` | |
| [x] | `ops.round` | |
| [x] | `ops.sign` | |
| [x] | `ops.sin` | |
| [x] | `ops.sinh` | |
| [x] | `ops.split` | |
| [x] | `ops.sqrt` | |
| [x] | `ops.square` | |
| [x] | `ops.squeeze` | |
| [x] | `ops.stack` | |
| [x] | `ops.strided_slice` | |
| [x] | `ops.subtract` | |
| [x] | `ops.sum` | |
| [x] | `ops.swapaxes` | |
| [x] | `ops.take` | |
| [x] | `ops.take_along_axis` | |
| [x] | `ops.tan` | |
| [x] | `ops.tanh` | |
| [x] | `ops.tensordot` | |
| [x] | `ops.tile` | |
| [x] | `ops.transpose` | |
| [x] | `ops.trunc` | |
| [x] | `ops.unsqueeze` | |
| [x] | `ops.update_slice` | |
| [x] | `ops.vdot` | |
| [ ] | `ops.vsplit` | |
| [ ] | `ops.vstack` | |
| [x] | `ops.where` | |
| [x] | `ops.zeros` | |
| [x] | `ops.zeros_like` | |

### `ml_switcheroo.control_flow`
| Status | Binding | Notes |
|---|---|---|
| [x] | `cf.cond` | |
| [ ] | `cf.scan` | |
| [ ] | `cf.vmap` | |

### `ml_switcheroo.random`
| Status | Binding | Notes |
|---|---|---|
| [x] | `random.PRNGKey` | |
| [ ] | `random.bernoulli` | |
| [ ] | `random.categorical` | |
| [ ] | `random.choice` | |
| [ ] | `random.fold_in` | |
| [ ] | `random.normal` | |
| [ ] | `random.permutation` | |
| [ ] | `random.randint` | |
| [x] | `random.split` | |
| [ ] | `random.truncated_normal` | |
| [ ] | `random.uniform` | |

### `ml_switcheroo.grad`
| Status | Binding | Notes |
|---|---|---|
| [ ] | `ir_grad` | Used as `ml_switcheroo.grad.grad` |
