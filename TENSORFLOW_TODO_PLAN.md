# TensorFlow Parity Plan for `ml-switcheroo-compiler` (Exhaustive Edition)

This document exhaustively details every single low-level operator, state management mechanism, and tracing capability that `ml-switcheroo-compiler` must implement. `zero-tensorflow` delegates all actual computation to this backend; therefore, missing any of these primitives will break the `zero-tensorflow` frontend parity tests against the genuine `tensorflow` library.

## 1. Core Data Types & Type Coercion
To emulate TensorFlow's strict type system and broadcasting rules, the compiler must support the following.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `cast` | `cast(x: Tensor, dtype: DType) -> Tensor` | Casts a tensor to a new type. | Must strictly follow TF promotion rules (differ from NumPy for 16-bit). |
| [x] | `bitcast` | `bitcast(input: Tensor, type: DType) -> Tensor` | Bitcasts a tensor without copying data. | Essential for low-level quantized operations and IR matching. |
| [ ] | `as_dtype` | `as_dtype(type_value) -> DType` | Converts a value to a DType object. | Used internally for type resolution passes. |
| [ ] | `complex` | `complex(real: Tensor, imag: Tensor) -> Tensor` | Converts real numbers to complex. | Emits `LogicalNode` for complex instantiation. |
| [x] | `real` | `real(input: Tensor) -> Tensor` | Returns the real part of a tensor. | No-op for real tensors, purely an extractor for complex. |
| [x] | `imag` | `imag(input: Tensor) -> Tensor` | Returns the imaginary part of a tensor. | Returns zeros for real inputs. |

## 2. State & Variable Management
TensorFlow heavily utilizes object-oriented state, particularly in `tf.Variable` and Keras components.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [ ] | `Variable` | `class Variable(initial_value, trainable=True)` | A tensor whose value can be mutated. | The compiler must `lift_state` on these objects during tracing. |
| [ ] | `read_variable_op` | `read_variable_op(resource: Tensor, dtype: DType) -> Tensor` | Reads the value of a variable. | Required for strict reference semantics in graph mode. |
| [ ] | `assign` | `assign(ref: Variable, value: Tensor) -> Tensor` | Assigns a new value to a variable. | Must trigger state functionalization in the IR graph. |
| [ ] | `assign_add` | `assign_add(ref: Variable, value: Tensor) -> Tensor` | Adds value to variable and returns. | Eager: in-place. Tracing: emits state-update node. |
| [ ] | `assign_sub` | `assign_sub(ref: Variable, value: Tensor) -> Tensor` | Subtracts value from variable. | - |
| [ ] | `scatter_nd_update` | `scatter_nd_update(ref: Variable, indices, updates)` | Applies sparse updates to a variable. | Requires specific IR support for sparse data structures. |
| [ ] | `scatter_nd_add` | `scatter_nd_add(ref: Variable, indices, updates)` | Applies sparse addition to a variable. | - |
| [ ] | `scatter_nd_sub` | `scatter_nd_sub(ref: Variable, indices, updates)` | Applies sparse subtraction to a variable. | - |

## 3. Autodiff, Tape & VJP Mappings
The compiler's Reverse-Mode Automatic Differentiation engine must support TF's tape semantics.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [ ] | `GradientTape` | `class GradientTape(persistent=False)` | Record operations for autodiff. | Must use thread-local tape contexts inside the compiler. |
| [ ] | `watch` | `watch(tensor: Tensor)` | Ensures `tensor` is traced by tape. | Required for tracking non-trainable inputs. |
| [ ] | `gradient` | `gradient(target, sources) -> List[Tensor]` | Computes gradient of targets w.r.t sources. | Runs topological sort and accumulates exact VJPs. |
| [ ] | `jacobian` | `jacobian(target, sources) -> List[Tensor]` | Computes jacobian of targets w.r.t sources. | Required for advanced second-order metrics/losses. |
| [ ] | `batch_jacobian` | `batch_jacobian(target, sources) -> List[Tensor]` | Computes batch jacobian. | Crucial for efficient vectorized second-order derivatives. |
| [ ] | `hessians` | `hessians(target, sources) -> List[Tensor]` | Computes the Hessian matrix. | Evaluates nested VJPs automatically. |
| [ ] | `stop_gradient` | `stop_gradient(input: Tensor) -> Tensor` | Stops gradient from flowing backwards. | Identity in forward pass, blocks edge traversal in backward. |
| [ ] | `custom_gradient` | `@custom_gradient` | Decorator for custom forward/backward passes. | Binds a python closure to a custom `LogicalNode` VJP map. |

## 4. Control Flow, Tracing & Debugging
Mapping dynamic Python loops and conditionals to static graph dialects.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `cond` | `cond(pred, true_fn, false_fn) -> Tensor` | Return `true_fn()` if `pred` else `false_fn()`. | Compiles into conditional `LogicalNode`. Requires branch shape matching. |
| [ ] | `while_loop` | `while_loop(cond, body, loop_vars)` | Repeat `body` while `cond` is true. | Emits loop/scan primitives into `ml-switcheroo-ir`. |
| [ ] | `case` | `case(pred_fn_pairs, default, exclusive=False)` | Create a case operation. | Generalization of `cond`. |
| [ ] | `function` | `@function(jit_compile=False)` | Compiles a function into a graph. | Sets tracing context; intercepts ops, builds `LogicalGraph`. |
| [ ] | `print` | `print(*inputs, **kwargs)` | Prints inputs to standard out/err. | Requires side-effect token chaining in compiled graphs. |
| [ ] | `Assert` | `Assert(condition, data, summarize=None)` | Asserts that `condition` is true. | Aborts execution statically/dynamically if False. |

## 5. Tensor Creation & Factories
Graph-mode instantiations of common tensor shapes.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `constant` | `constant(value, dtype=None, shape=None)` | Creates a constant tensor. | Traces into a literal node in the IR. |
| [x] | `zeros_like` | `zeros_like(tensor: Tensor, dtype=None) -> Tensor` | Creates a tensor of zeros like `tensor`. | - |
| [x] | `ones_like` | `ones_like(tensor: Tensor, dtype=None) -> Tensor` | Creates a tensor of ones like `tensor`. | - |
| [ ] | `fill` | `fill(dims, value, name=None) -> Tensor` | Creates a tensor filled with a scalar value. | Backs `tf.zeros` and `tf.ones`. |

## 6. Shape, Metadata & Manipulation
Core data-shuffling operations dictating memory layout and views.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `shape` | `shape(input: Tensor) -> Tensor` | Returns the shape of a tensor. | Evaluates dynamically in graph mode for unknown dims. |
| [ ] | `size` | `size(input: Tensor) -> Tensor` | Returns the size of a tensor. | Total element count. |
| [ ] | `rank` | `rank(input: Tensor) -> Tensor` | Returns the rank of a tensor. | Number of dimensions. |
| [x] | `reshape` | `reshape(tensor: Tensor, shape: Tuple) -> Tensor` | Reshapes a tensor. | Zero-copy metadata change in Eager mode. |
| [ ] | `expand_dims` | `expand_dims(input: Tensor, axis: int) -> Tensor` | Inserts a dimension of 1. | - |
| [x] | `squeeze` | `squeeze(input: Tensor, axis: List[int]) -> Tensor` | Removes dimensions of size 1. | - |
| [x] | `transpose` | `transpose(a: Tensor, perm: List[int]) -> Tensor` | Transposes `a` using `perm`. | Essential for NCHW <-> NHWC conversions. |
| [ ] | `concat` | `concat(values: List[Tensor], axis: int) -> Tensor` | Concatenates tensors along an axis. | - |
| [x] | `stack` | `stack(values: List[Tensor], axis: int) -> Tensor` | Stacks tensors into a higher rank. | - |
| [x] | `unstack` | `unstack(value: Tensor, num=None, axis=0)` | Unpacks a tensor into a list of tensors. | - |
| [x] | `split` | `split(value: Tensor, num_or_size_splits, axis=0)` | Splits a tensor into sub tensors. | - |
| [x] | `pad` | `pad(tensor: Tensor, paddings: Tensor, mode='CONSTANT')` | Pads a tensor. | Required by convolution layers. |
| [x] | `tile` | `tile(input: Tensor, multiples: Tensor) -> Tensor` | Constructs a tensor by tiling a given tensor. | - |
| [x] | `strided_slice` | `strided_slice(input_, begin, end, strides)` | Extracts a strided slice. | Extremely complex to parse ellipsis/newaxis logic accurately. |
| [x] | `gather` | `gather(params: Tensor, indices: Tensor, axis=0)` | Gather slices according to `indices`. | Highly used in NLP embeddings. |
| [x] | `gather_nd` | `gather_nd(params: Tensor, indices: Tensor)` | Gather slices into a Tensor with shape specified by `indices`. | - |
| [x] | `scatter_nd` | `scatter_nd(indices, updates, shape)` | Scatters `updates` into a new tensor of `shape`. | Functional equivalent of `scatter_nd_update`. |
| [x] | `where` | `where(condition: Tensor, x=None, y=None)` | Returns elements from `x` or `y` based on condition. | If `x`/`y` are None, returns coordinates of True elements. |
| [ ] | `boolean_mask` | `boolean_mask(tensor: Tensor, mask: Tensor)` | Apply boolean mask to tensor. | Equivalent to NumPy `tensor[mask]`. |
| [x] | `broadcast_to` | `broadcast_to(input: Tensor, shape: Tensor)` | Broadcast an array for a compatible shape. | - |
| [x] | `roll` | `roll(input: Tensor, shift, axis)` | Rolls the elements of a tensor along an axis. | - |
| [ ] | `reverse` | `reverse(tensor: Tensor, axis)` | Reverses specific dimensions of a tensor. | - |

## 7. Core Math (Unary & Binary)
Strict element-wise mathematics with autodiff VJPs.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `add` | `add(x: Tensor, y: Tensor) -> Tensor` | Element-wise addition. | Must support NumPy broadcast semantics. |
| [x] | `sub` | `subtract(x: Tensor, y: Tensor) -> Tensor` | Element-wise subtraction. | - |
| [x] | `mul` | `multiply(x: Tensor, y: Tensor) -> Tensor` | Element-wise multiplication. | - |
| [x] | `div` | `divide(x: Tensor, y: Tensor) -> Tensor` | Element-wise division. | Promotes integers to floats. |
| [ ] | `truediv` | `truediv(x: Tensor, y: Tensor) -> Tensor` | Divides x / y elementwise. | Explicit python `/` behavior. |
| [ ] | `floordiv` | `floordiv(x: Tensor, y: Tensor) -> Tensor` | Divides x / y elementwise, rounding toward -inf. | Explicit python `//` behavior. |
| [x] | `mod` | `math.mod(x: Tensor, y: Tensor) -> Tensor` | Returns element-wise remainder of division. | - |
| [x] | `pow` | `math.pow(x: Tensor, y: Tensor) -> Tensor` | Computes x to the power of y. | VJP: `y * x**(y-1)` and `x**y * log(x)`. |
| [x] | `maximum` | `math.maximum(x: Tensor, y: Tensor) -> Tensor` | Element-wise maximum of x and y. | VJP routes grad to the max element. |
| [x] | `minimum` | `math.minimum(x: Tensor, y: Tensor) -> Tensor` | Element-wise minimum of x and y. | VJP routes grad to the min element. |
| [ ] | `squared_difference`| `math.squared_difference(x, y)` | Returns `(x - y)(x - y)`. | Optimization for MSE loss graphs. |
| [x] | `abs` | `math.abs(x: Tensor) -> Tensor` | Computes absolute value. | Subgradient at 0 must be 0. |
| [x] | `neg` | `math.negative(x: Tensor) -> Tensor` | Computes numerical negative value element-wise. | - |
| [x] | `sign` | `math.sign(x: Tensor) -> Tensor` | Returns element-wise sign. | VJP is 0 everywhere except NaN. |
| [x] | `reciprocal` | `math.reciprocal(x: Tensor) -> Tensor` | Computes `1 / x` element-wise. | VJP: `-grad / (x * x)`. |
| [x] | `square` | `math.square(x: Tensor) -> Tensor` | Computes `x * x` element-wise. | - |
| [x] | `sqrt` | `math.sqrt(x: Tensor) -> Tensor` | Computes square root of x. | VJP: `grad / (2 * sqrt(x))`. |
| [x] | `rsqrt` | `math.rsqrt(x: Tensor) -> Tensor` | Computes reciprocal of square root of x. | VJP: `-0.5 * grad * rsqrt(x)**3`. |
| [x] | `exp` | `math.exp(x: Tensor) -> Tensor` | Computes `e^x`. | VJP: `grad * exp(x)`. |
| [x] | `expm1` | `math.expm1(x: Tensor) -> Tensor` | Computes `exp(x) - 1`. | Numerically stable for small x. |
| [x] | `log` | `math.log(x: Tensor) -> Tensor` | Computes natural logarithm of x. | - |
| [x] | `log1p` | `math.log1p(x: Tensor) -> Tensor` | Computes `log(1 + x)`. | Numerically stable for small x. |
| [x] | `sin` | `math.sin(x: Tensor) -> Tensor` | Computes sine. | VJP: `grad * cos(x)`. |
| [x] | `cos` | `math.cos(x: Tensor) -> Tensor` | Computes cosine. | VJP: `grad * -sin(x)`. |
| [x] | `tan` | `math.tan(x: Tensor) -> Tensor` | Computes tangent. | - |
| [x] | `asin` | `math.asin(x: Tensor) -> Tensor` | Computes inverse sine. | - |
| [x] | `acos` | `math.acos(x: Tensor) -> Tensor` | Computes inverse cosine. | - |
| [x] | `atan` | `math.atan(x: Tensor) -> Tensor` | Computes inverse tangent. | - |
| [x] | `sinh` | `math.sinh(x: Tensor) -> Tensor` | Computes hyperbolic sine. | - |
| [x] | `cosh` | `math.cosh(x: Tensor) -> Tensor` | Computes hyperbolic cosine. | - |
| [x] | `tanh` | `math.tanh(x: Tensor) -> Tensor` | Computes hyperbolic tangent. | VJP: `grad * (1 - tanh(x)**2)`. |
| [x] | `asinh` | `math.asinh(x: Tensor) -> Tensor` | Computes inverse hyperbolic sine. | - |
| [x] | `acosh` | `math.acosh(x: Tensor) -> Tensor` | Computes inverse hyperbolic cosine. | - |
| [x] | `atanh` | `math.atanh(x: Tensor) -> Tensor` | Computes inverse hyperbolic tangent. | - |
| [x] | `erf` | `math.erf(x: Tensor) -> Tensor` | Computes Gauss error function. | Used heavily in GELU activations. |
| [x] | `erfc` | `math.erfc(x: Tensor) -> Tensor` | Computes complementary error function. | - |
| [x] | `lgamma` | `math.lgamma(x: Tensor) -> Tensor` | Computes log of absolute value of Gamma function. | - |
| [x] | `digamma` | `math.digamma(x: Tensor) -> Tensor` | Computes Psi (digamma) function. | - |
| [x] | `round` | `math.round(x: Tensor) -> Tensor` | Rounds values to nearest integer. | VJP is 0. |
| [x] | `floor` | `math.floor(x: Tensor) -> Tensor` | Returns element-wise largest integer <= x. | VJP is 0. |
| [x] | `ceil` | `math.ceil(x: Tensor) -> Tensor` | Returns element-wise smallest integer >= x. | VJP is 0. |

## 8. Reductions & Logical Operations
Reducing spatial dimensions and conditional comparisons.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [ ] | `reduce_sum` | `reduce_sum(input_tensor, axis=None, keepdims=False)` | Computes the sum of elements across dimensions. | - |
| [ ] | `reduce_mean` | `reduce_mean(input_tensor, axis=None, keepdims=False)` | Computes the mean across dimensions. | Core operation for loss averages. |
| [ ] | `reduce_prod` | `reduce_prod(input_tensor, axis=None, keepdims=False)` | Computes the product of elements across dimensions. | - |
| [ ] | `reduce_max` | `reduce_max(input_tensor, axis=None, keepdims=False)` | Computes the maximum across dimensions. | VJP requires sparse broadcast to max index. |
| [ ] | `reduce_min` | `reduce_min(input_tensor, axis=None, keepdims=False)` | Computes the minimum across dimensions. | - |
| [ ] | `reduce_all` | `reduce_all(input_tensor, axis=None, keepdims=False)` | Computes logical AND across dimensions. | Output dtype is boolean. |
| [ ] | `reduce_any` | `reduce_any(input_tensor, axis=None, keepdims=False)` | Computes logical OR across dimensions. | Output dtype is boolean. |
| [x] | `argmax` | `argmax(input: Tensor, axis=None) -> Tensor` | Index with largest value. | Output is integer; non-differentiable. |
| [x] | `argmin` | `argmin(input: Tensor, axis=None) -> Tensor` | Index with smallest value. | Output is integer; non-differentiable. |
| [x] | `cumsum` | `math.cumsum(x, axis=0, exclusive=False, reverse=False)` | Cumulative sum. | Essential for RNNs and sequence masking. |
| [ ] | `cumprod` | `math.cumprod(x, axis=0, exclusive=False, reverse=False)` | Cumulative product. | - |
| [x] | `logical_and` | `logical_and(x: Tensor, y: Tensor) -> Tensor` | Element-wise logical AND. | Inputs must be boolean. |
| [x] | `logical_or` | `logical_or(x: Tensor, y: Tensor) -> Tensor` | Element-wise logical OR. | - |
| [x] | `logical_not` | `logical_not(x: Tensor) -> Tensor` | Element-wise logical NOT. | - |
| [x] | `logical_xor` | `logical_xor(x: Tensor, y: Tensor) -> Tensor` | Element-wise logical XOR. | - |
| [x] | `equal` | `math.equal(x: Tensor, y: Tensor) -> Tensor` | Returns `(x == y)` element-wise. | Evaluates exact equivalence (with broadcast). |
| [x] | `not_equal` | `math.not_equal(x: Tensor, y: Tensor) -> Tensor` | Returns `(x != y)` element-wise. | - |
| [x] | `less` | `math.less(x: Tensor, y: Tensor) -> Tensor` | Returns `(x < y)` element-wise. | - |
| [x] | `less_equal` | `math.less_equal(x: Tensor, y: Tensor) -> Tensor` | Returns `(x <= y)` element-wise. | - |
| [x] | `greater` | `math.greater(x: Tensor, y: Tensor) -> Tensor` | Returns `(x > y)` element-wise. | - |
| [x] | `greater_equal`| `math.greater_equal(x: Tensor, y: Tensor) -> Tensor` | Returns `(x >= y)` element-wise. | - |

## 9. Linear Algebra (`tf.linalg`)
Matrix operations necessary for dense networks and attention mechanisms.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `matmul` | `matmul(a: Tensor, b: Tensor, transpose_a=False, transpose_b=False)` | Multiplies matrix a by matrix b. | The backbone of Dense layers. |
| [x] | `tensordot` | `tensordot(a: Tensor, b: Tensor, axes)` | Tensor contraction of a and b along specified axes. | VJP splits dimensions manually. |
| [x] | `einsum` | `einsum(equation: str, *inputs: Tensor) -> Tensor` | Evaluates the Einstein summation convention. | Critical for multi-head attention blocks. |
| [x] | `norm` | `norm(tensor: Tensor, ord='euclidean', axis=None)` | Computes vector/matrix norm. | - |
| [ ] | `cholesky` | `cholesky(input: Tensor) -> Tensor` | Computes Cholesky decomposition of square matrices. | Advanced linear algebra op. |
| [ ] | `svd` | `svd(tensor: Tensor, full_matrices=False)` | Computes Singular Value Decomposition. | - |
| [ ] | `qr` | `qr(input: Tensor, full_matrices=False)` | Computes QR decomposition. | - |
| [ ] | `inv` | `inv(input: Tensor) -> Tensor` | Computes inverse of square matrices. | - |
| [ ] | `det` | `det(input: Tensor) -> Tensor` | Computes determinant of square matrices. | - |
| [ ] | `cross` | `cross(a: Tensor, b: Tensor)` | Computes pairwise cross product. | - |
| [ ] | `trace` | `trace(x: Tensor)` | Computes the trace of a tensor. | Sum along diagonal. |
| [x] | `diag` | `diag(diagonal: Tensor)` | Returns a diagonal tensor with a given diagonal values. | - |
| [ ] | `band_part` | `band_part(input: Tensor, num_lower, num_upper)` | Copy a tensor setting everything outside a central band to zero. | Used for causal attention masking. |

## 10. Neural Network Primitives (Activations)
Specific deep learning non-linearities with exact mathematical VJPs.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `relu` | `nn.relu(features: Tensor) -> Tensor` | Computes rectified linear: `max(features, 0)`. | Standard fast activation. |
| [ ] | `relu6` | `nn.relu6(features: Tensor) -> Tensor` | Computes Rectified Linear 6: `min(max(features, 0), 6)`. | - |
| [x] | `leaky_relu` | `nn.leaky_relu(features, alpha=0.2)` | Computes Leaky ReLU. | Prevents dying ReLUs via negative slope. |
| [x] | `elu` | `nn.elu(features, alpha=1.0)` | Computes exponential linear function. | - |
| [x] | `selu` | `nn.selu(features: Tensor) -> Tensor` | Computes scaled exponential linear function. | Internally scales output for self-normalization. |
| [x] | `gelu` | `nn.gelu(features, approximate=False)` | Computes Gaussian Error Linear Unit (GELU). | Must support exact (`erf`) and `tanh` approximations. |
| [ ] | `silu` | `nn.silu(features: Tensor) -> Tensor` | Computes Swish/SiLU activation. | `x * sigmoid(x)`. |
| [x] | `swish` | `nn.swish(features: Tensor) -> Tensor` | Alias for silu. | - |
| [x] | `sigmoid` | `math.sigmoid(x: Tensor) -> Tensor` | Computes `1 / (1 + exp(-x))`. | VJP: `grad * sigmoid(x) * (1 - sigmoid(x))`. |
| [ ] | `hard_sigmoid` | `nn.hard_sigmoid(x: Tensor) -> Tensor` | Piecewise linear approximation of sigmoid. | Fast inference. |
| [x] | `softmax` | `nn.softmax(logits: Tensor, axis=-1)` | Computes softmax activations. | Output sums to 1 along `axis`. |
| [x] | `log_softmax` | `nn.log_softmax(logits: Tensor, axis=-1)` | Computes log softmax activations. | Numerically stable alternative to `log(softmax(x))`. |
| [x] | `softplus` | `nn.softplus(features: Tensor) -> Tensor` | Computes softplus: `log(exp(features) + 1)`. | Smooth approximation of ReLU. |
| [ ] | `softsign` | `nn.softsign(features: Tensor) -> Tensor` | Computes softsign: `features / (abs(features) + 1)`. | - |

## 11. Neural Network Primitives (Convolutions & Pooling)
Core image, sequence, and volume sliding window operations.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `conv1d` | `nn.conv1d(input, filters, stride, padding)` | Computes a 1-D convolution. | Used for time-series. Must strictly support `VALID` and `SAME` padding logic. |
| [x] | `conv2d` | `nn.conv2d(input, filters, strides, padding)` | Computes a 2-D convolution. | Used for images. Handles NCHW and NHWC. |
| [x] | `conv3d` | `nn.conv3d(input, filters, strides, padding)` | Computes a 3-D convolution. | Used for volumes/video. |
| [ ] | `conv1d_transpose`| `nn.conv1d_transpose(...)` | Transposed 1D convolution. | Learns to upsample 1D sequences. |
| [ ] | `conv2d_transpose`| `nn.conv2d_transpose(...)` | Transposed 2D convolution. | Learns to upsample 2D images. |
| [ ] | `conv3d_transpose`| `nn.conv3d_transpose(...)` | Transposed 3D convolution. | Learns to upsample 3D volumes. |
| [ ] | `depthwise_conv2d`| `nn.depthwise_conv2d(...)` | Depthwise 2-D convolution. | Separates spatial filtering from feature generation. |
| [ ] | `separable_conv2d`| `nn.separable_conv2d(...)` | Separable 2-D convolution. | Depthwise followed by pointwise convolution. |
| [x] | `max_pool1d` | `nn.max_pool1d(input, ksize, strides, padding)` | Max pooling operation for 1D data. | Track indices for the backward pass. |
| [x] | `max_pool2d` | `nn.max_pool2d(input, ksize, strides, padding)` | Max pooling operation for 2D spatial data. | - |
| [x] | `max_pool3d` | `nn.max_pool3d(input, ksize, strides, padding)` | Max pooling operation for 3D data. | - |
| [x] | `avg_pool1d` | `nn.avg_pool1d(input, ksize, strides, padding)` | Average pooling operation for 1D data. | - |
| [x] | `avg_pool2d` | `nn.avg_pool2d(input, ksize, strides, padding)` | Average pooling operation for 2D data. | - |
| [x] | `avg_pool3d` | `nn.avg_pool3d(input, ksize, strides, padding)` | Average pooling operation for 3D data. | - |
| [ ] | `fractional_max_pool`| `nn.fractional_max_pool(...)` | Performs fractional max pooling. | Randomly generated pooling regions. |

## 12. Neural Network Primitives (Normalization & Misc)
Loss function helpers, normalizations, and structural ML layers.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [ ] | `bias_add` | `nn.bias_add(value, bias, data_format=None)` | Adds bias to value. | Specialized `add` optimized for feature maps broadcast. |
| [ ] | `batch_normalization`| `nn.batch_normalization(x, mean, var, offset, scale, eps)`| Batch normalization. | Tricky state handling. The compiler must intercept and route running mean/var properly. |
| [x] | `layer_norm` | `nn.layer_norm(x, scale, offset)` | Layer normalization. | Independently normalizes the last dimension. |
| [ ] | `l2_normalize` | `math.l2_normalize(x, axis=None, epsilon=1e-12)` | Normalizes along dimension using L2 norm. | - |
| [x] | `dropout` | `nn.dropout(x, rate, noise_shape=None, seed=None)` | Computes dropout. | Compiler trace must branch on `training=True` flag natively in IR. |
| [x] | `alpha_dropout` | `nn.alpha_dropout(x, rate, seed=None)` | Computes alpha dropout. | Maintains mean and variance for SELU blocks. |
| [ ] | `top_k` | `math.top_k(input, k=1, sorted=True)` | Finds values and indices of the `k` largest entries. | Required for decoding. |
| [ ] | `in_top_k` | `math.in_top_k(targets, predictions, k)` | Says whether the targets are in the top `K` predictions. | Used for SparseTopK metrics. |
| [ ] | `one_hot` | `one_hot(indices, depth, on_value=1, off_value=0)`| Returns a one-hot tensor. | Converts integer sparse classes to dense distributions. |
| [ ] | `ctc_loss` | `nn.ctc_loss(labels, logits, label_length, logit_length)`| Computes the CTC (Connectionist Temporal Classification) Loss. | Specialized sequence-to-sequence loss mapping. |

## 13. Random Number Generation
Implementation of stateless and stateful RNG mirroring `tf.random`.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [ ] | `random_normal` | `random.normal(shape, mean=0.0, stddev=1.0)` | Random values from a normal distribution. | Maps to stateful or seeded compiler generator. |
| [ ] | `random_uniform` | `random.uniform(shape, minval=0, maxval=None)`| Random values from a uniform distribution. | - |
| [ ] | `truncated_normal`| `random.truncated_normal(shape, mean, stddev)` | Outputs random values from a truncated normal distribution. | Prevents >2 stddev outliers. |
| [ ] | `categorical` | `random.categorical(logits, num_samples)` | Draws samples from a categorical distribution. | Necessary for discrete action sampling in RL. |
| [ ] | `poisson` | `random.poisson(shape, lam)` | Draws samples from a Poisson distribution. | - |
| [ ] | `gamma` | `random.gamma(shape, alpha)` | Draws samples from a Gamma distribution. | - |
| [ ] | `set_seed` | `random.set_seed(seed: int)` | Sets the global graph random seed. | Must guarantee fully deterministic `zero-zoo` headless validation. |
| [ ] | `stateless_random_normal`| `random.stateless_random_normal(shape, seed)` | Stateless random normal. | Pure functional RNG, requires strict PRNG key threading. |
| [ ] | `stateless_random_uniform`| `random.stateless_random_uniform(shape, seed)` | Stateless random uniform. | Pure functional RNG. |

## 14. Compilation, Devices & IO
Backend bindings.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | --- | --- | --- | --- |
| [x] | `device` | `device(device_name: str)` | Context manager to specify computation device. | Compiler maps this to `cpu`, `gpu`, or `webgpu` arenas. |
| [ ] | `TensorArray` | `class TensorArray(dtype, size, dynamic_size)` | Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays. | Used natively inside RNNs and `while_loop` tracing bounds. |
