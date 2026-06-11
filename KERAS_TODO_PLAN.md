# Exhaustive Keras Compatibility Implementation Plan

This document outlines the exhaustive requirements for `ml-switcheroo-compiler` to support `zero-keras` achieving 100% semantic and syntactic compatibility with the official Keras API. It encompasses state management, AD, compiler intrinsics, dynamic shaping, XLA-parity primitives, distributed training, and structural tree manipulations.

## 1. Compiler Core: State Lifting & Functionalization
Keras relies heavily on stateful abstractions (variables, metrics state, PRNG seeds). The compiler must intercept and lift these into pure functional graph bounds.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [x] | `lift_state` | `compiler.lift_state(fn: Callable) -> Callable` | Transforms a stateful function into a pure function taking state as explicit inputs and returning state as explicit outputs. | Must trace Keras `Variable` accesses and map them to `LogicalGraph` parameter nodes. |
| [ ] | `assign` | `ProxyTensor.assign(value: ProxyTensor)` | In-place replacement of tensor contents. | Translates to a state-update edge in the IR. Required for Keras weight updates. |
| [ ] | `assign_add` | `ProxyTensor.assign_add(value: ProxyTensor)` | In-place addition to tensor contents. | Heavily used in Keras metrics (e.g., `Mean`, `Sum`) to accumulate running state. |
| [ ] | `assign_sub` | `ProxyTensor.assign_sub(value: ProxyTensor)` | In-place subtraction from tensor contents. | Used in EMA (Exponential Moving Average) updates and specific stateful optimizers. |
| [x] | `cond` | `lax.cond(pred, true_fn, false_fn, *operands)` | Conditionally applies `true_fn` or `false_fn` based on boolean scalar `pred`. | Crucial for dynamic routing, optional execution paths, and dynamic masking. |
| [ ] | `while_loop` | `lax.while_loop(cond_fun, body_fun, init_val)` | Executes `body_fun` repeatedly while `cond_fun` evaluates to True. | Required for arbitrary-length sequences and custom dynamic training loops. |
| [ ] | `scan` | `lax.scan(f, init, xs, length=None)` | Applies `f` along the leading axis of `xs` while carrying state. | The backbone of optimized RNN unrolling (`LSTM`, `GRU`) in compiled graphs. |
| [x] | `dynamic_slice` | `lax.dynamic_slice(operand, start_indices, slice_sizes)` | Extracts a slice from an array at dynamically computed start indices. | Needed for sequence processing where window offsets are computed at runtime. |
| [ ] | `dynamic_update_slice` | `lax.dynamic_update_slice(operand, update, start_indices)` | Updates a slice of an array at dynamically computed start indices. | Critical for updating specific hidden states or memory banks dynamically. |

## 2. Automatic Differentiation (AD Engine)
The compiler must implement reverse-mode AD and provide exact VJPs for all loss functions, metrics, and complex nested neural architectures.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [x] | `grad` | `compiler.grad(fn: Callable, argnums: int = 0) -> Callable` | Returns a function that computes the gradient of `fn` with respect to specified arguments. | Core primitive powering `Model.fit` backpropagation and custom `train_step` methods. |
| [ ] | `value_and_grad` | `compiler.value_and_grad(fn: Callable, argnums: int = 0) -> Callable` | Returns a function that computes both the output of `fn` and its gradients. | Standard usage for optimization loops to avoid redundant forward passes during training. |
| [ ] | `vjp` | `compiler.vjp(fun: Callable, *primals)` | Vector-Jacobian Product; returns the primal output and a function to compute the VJP. | The mathematical core of the reverse-mode AD engine. Needed for all differentiable layers. |
| [ ] | `stop_gradient` | `compiler.stop_gradient(x: ProxyTensor) -> ProxyTensor` | Acts as an identity function in the forward pass but blocks gradients in the backward pass. | Used extensively in GANs, specific metric calculations, and custom gradient routing. |
| [ ] | `custom_vjp` | `compiler.custom_vjp(fn: Callable)` | Decorator to define custom forward and backward passes for a specific function. | Required for numerically unstable operations where exact derivatives fail (e.g. `log-sum-exp`). |
| [ ] | `segment_sum` | `ops.segment_sum(data, segment_ids, num_segments=None)` | Computes the sum of tensor elements grouped by `segment_ids`. | Essential for sparse gradient updates, specifically for `Embedding` layer backward passes. |
| [ ] | `closure_capture` | `Internal AD Mechanism` | Capable of differentiating functions that capture variables from outer scopes. | Required for Keras metrics and losses that close over hyperparameters or states. |

## 3. Deep Learning Primitives (LAX equivalents)
These are the foundational math operations that `keras.layers` and `keras.ops` wrap. The backend must map these to optimized execution kernels (e.g. WebGPU, WASM).

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `conv_general_dilated` | `lax.conv_general_dilated(lhs, rhs, window_strides, padding, ...)` | General N-dimensional convolution with support for strides, padding, and dilations. | Universal backend for `Conv1D`, `Conv2D`, `Conv3D`, `DepthwiseConv2D`, and separable variants. |
| [ ] | `reduce_window` | `lax.reduce_window(operand, init_value, computation, window_dimensions, ...)` | Applies a reduction function over a sliding window of the input. | Backing operation for all `MaxPooling` and `AveragePooling` layers. |
| [x] | `einsum` | `jnp.einsum(subscripts: str, *operands)` | Evaluates the Einstein summation convention on the operands. | Critical for `MultiHeadAttention`, `EinsumDense`, and advanced tensor contractions. |
| [ ] | `dot_general` | `lax.dot_general(lhs, rhs, dimension_numbers)` | General dot product with support for batching and contracting arbitrary dimensions. | Core primitive for `Dense`, linear transformations, and recurrent layers. |
| [ ] | `broadcast_in_dim` | `lax.broadcast_in_dim(operand, shape, broadcast_dimensions)` | Broadcasts an array to a target shape by matching specified dimensions. | Underpins nearly all element-wise binary operations, broadcasting, and metric alignments. |
| [x] | `pad` | `lax.pad(operand, padding_value, padding_config)` | Pads an array with a constant value according to a complex padding configuration. | Directly backs `ZeroPadding1D/2D/3D` and exact boundary conditions in convolutions. |
| [x] | `gather` | `lax.gather(operand, start_indices, dimension_numbers, slice_sizes)` | Extracts slices from an array at specified indices based on dimension numbers. | Used heavily in `Embedding` layers and sparse metric/loss evaluations. |
| [x] | `scatter` | `lax.scatter(operand, scatter_indices, updates, dimension_numbers)` | Scatters updates into an array at specified indices based on dimension numbers. | Used in certain complex loss functions, one-hot encodings, and sparse state updates. |
| [ ] | `top_k` | `lax.top_k(operand, k)` | Returns the top `k` values and their indices along the last dimension. | Critical for `TopKCategoricalAccuracy` and `SparseTopKCategoricalAccuracy`. |
| [ ] | `sort` | `lax.sort(operand, dimension=-1, is_stable=True)` | Sorts the elements of an array along a given dimension. | Required for rank-based metrics, custom non-maximum suppression, and sorting. |
| [ ] | `select` | `lax.select(pred, on_true, on_false)` | Selects elements from `on_true` or `on_false` based on a boolean mask. | Vectorized ternary operator; heavily used in activation functions like `ReLU` and masking. |

## 4. Structural Utilities (PyTrees)
Keras Models routinely accept and return arbitrarily nested dictionaries, lists, and tuples of tensors.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `tree_map` | `tree_util.tree_map(f, tree, *rest)` | Maps a function `f` over the leaves of a PyTree. | Crucial for applying gradient updates to nested optimizers or iterating over Model inputs. |
| [ ] | `tree_flatten` | `tree_util.tree_flatten(tree)` | Flattens a PyTree into a list of leaves and an auxiliary `treedef`. | Allows the compiler to convert multi-input/output nested Keras models into flat arrays. |
| [ ] | `tree_unflatten` | `tree_util.tree_unflatten(treedef, leaves)` | Reconstructs a PyTree from a `treedef` and a list of leaves. | Used to re-nest flat outputs from the compiled graph back into Keras dictionary outputs. |

## 5. Distributed Training & SPMD Primitives
To support `keras.distribution` (DataParallel and ModelParallel), the compiler must provide cross-device communication nodes.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `psum` | `lax.psum(x, axis_name)` | Computes an all-reduce sum over the specified mapped axis. | Synchronizes gradient sums across replicas during DataParallel training. |
| [ ] | `pmean` | `lax.pmean(x, axis_name)` | Computes an all-reduce mean over the specified mapped axis. | Averages gradients or metric states across multiple GPU/TPU devices. |
| [ ] | `pmap` | `compiler.pmap(fn, axis_name)` | Compiles a function to be executed in parallel across multiple devices. | The gateway to triggering distributed execution for Keras model parallelism. |

## 6. Signal & Image Processing Extensions
Certain Keras layers require highly specialized algorithmic kernels that cannot be easily composed from basic math.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `fft` | `fft.fft(a, n=None, axis=-1)` | Computes the one-dimensional discrete Fourier Transform. | Explicitly required by `STFTSpectrogram` and `MelSpectrogram` Keras audio layers. |
| [ ] | `rfft` | `fft.rfft(a, n=None, axis=-1)` | Computes the one-dimensional discrete Fourier Transform for real input. | Optimized audio feature extraction for real-valued audio streams. |
| [ ] | `image.resize` | `image.resize(image, shape, method='bilinear')` | Resizes an image to the given target shape using interpolation. | Backing implementation for `keras.layers.Resizing` and image augmentation pipelines. |

## 7. Random Number Generation (`ml_switcheroo.random`)
Keras relies on stateless, seeded PRNG operations for reproducibility (matching the JAX/XLA paradigm).

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [x] | `split` | `random.split(key: PRNGKey, num: int = 2) -> Array` | Splits a PRNG key into `num` new, mathematically independent keys. | Ensures independent random streams across different layers and training steps. |
| [ ] | `fold_in` | `random.fold_in(key: PRNGKey, data: int) -> PRNGKey` | Deterministically folds integer data into a PRNG key to create a uniquely derived key. | Used for step-based RNG injection in training loops (e.g., varying dropout masks per step). |
| [ ] | `uniform` | `random.uniform(key, shape=(), dtype='float32', minval=0.0, maxval=1.0)` | Samples from a uniform distribution over the half-open interval `[minval, maxval)`. | Required for `RandomUniform`, `GlorotUniform`, and `HeUniform` initializers. |
| [ ] | `normal` | `random.normal(key, shape=(), dtype='float32')` | Samples standard normal random values. | Required for `RandomNormal`, `GlorotNormal`, and `HeNormal` initializers. |
| [ ] | `truncated_normal` | `random.truncated_normal(key, lower, upper, shape=(), dtype='float32')` | Samples from a truncated normal distribution. | Vital for `TruncatedNormal` and bounds-restricted initializations to prevent extreme outliers. |
| [ ] | `bernoulli` | `random.bernoulli(key, p=0.5, shape=None)` | Samples boolean values with probability `p` of being True. | The core mathematical operation backing all `Dropout`, `SpatialDropout`, and `AlphaDropout` layers. |
| [ ] | `categorical` | `random.categorical(key, logits, axis=-1, shape=None)` | Samples from a categorical distribution defined by unnormalized log-probabilities. | Required for discrete sampling algorithms, language modeling outputs, and specialized Keras operations. |

## 8. Type System & XLA Parity Behaviors
Keras operations heavily depend on exact XLA promotion rules, precision limits, and dynamic shaping.

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `type_promotion` | `compiler.promote_types(a, b)` | Implements exact JAX type promotion semantics (e.g. float32 + int32 -> float32). | Essential. Keras relies on precise casting rules for mixed precision and stable evaluation. |
| [ ] | `bfloat16_support` | `Internal Backend Primitive` | Native mathematical support for Brain Floating Point format (bfloat16). | Modern Keras LLM and TPUs default to bfloat16. The backend compiler must lower this correctly. |
| [ ] | `complex_support` | `Internal Backend Primitive` | End-to-end support for `complex64` and `complex128` operations and AD. | Critical for `STFT`, `MelSpectrogram`, and Keras audio preprocessing layers. |
| [ ] | `string_tensors` | `StringLookup / TextVectorization Backing` | Support for batching and hashing string tokens. | Required for NLP preprocessing (`TextVectorization`, `StringLookup`). Often handled via hashing in the IR. |
| [ ] | `polymorphic_shapes` | `compiler.shape_inference` | Static tracing of symbolic dimension sizes (e.g. `(None, 32)`). | Needed for dynamic batch sizes and variable length sequences (`RNN`s, `LSTM`s) without recompiling. |

## 9. Array Operations (`ml_switcheroo.jnp` Backend)
Keras relies on a comprehensive array API matching NumPy/JAX. The following `jax.numpy` methods are currently missing from the compiler and must be implemented:

| Checkbox | Name | Signature | Docstring | Notes |
| :---: | :--- | :--- | :--- | :--- |
| [x] | `ComplexWarning` | `(...)` | The warning raised when casting a complex dtype to a real dtype. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `absolute` | `(x: 'ArrayLike', /) -> 'Array'` | Calculate the absolute value element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `acos` | `(x, /)` | Trigonometric inverse cosine, element-wise. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `acosh` | `(x, /)` | Inverse hyperbolic cosine, element-wise. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `angle` | `(z: 'ArrayLike', deg: 'bool' = False) -> 'Array'` | Return the angle of a complex valued number or array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `append` | `(arr: 'ArrayLike', values: 'ArrayLike', axis: 'int &#124; None' = None) -> 'Array'` | Return a new array with values appended to the end of the original array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `apply_along_axis` | `(func1d: 'Callable', axis: 'int', arr: 'ArrayLike', *args, **kwargs) -> 'Array'` | Apply a function to 1-D slices along the given axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `apply_over_axes` | `(func: 'Callable[[ArrayLike, int], Array]', a: 'ArrayLike', axes: 'Sequence[int]') -> 'Array'` | Apply a function repeatedly over multiple axes. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `argpartition` | `(a: 'ArrayLike', kth: 'int', axis: 'int' = -1) -> 'Array'` | Returns indices that partially sort an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `argsort` | `(a: 'ArrayLike', axis: 'int &#124; None' = -1, *, kind: 'None' = None, order: 'None' = None, stable: 'bool' = True, descending: 'bool' = False) -> 'Array'` | Returns the indices that would sort an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `argwhere` | `(a: 'ArrayLike', *, size: 'int &#124; None' = None, fill_value: 'ArrayLike &#124; None' = None) -> 'Array'` | Find the indices of nonzero array elements | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `around` | `(a: 'ArrayLike', decimals: 'int' = 0, out: 'None' = None) -> 'Array'` | Round an array to the given number of decimals. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `array_equiv` | `(a1: 'ArrayLike', a2: 'ArrayLike') -> 'Array'` | Returns True if input arrays are shape consistent and all elements equal. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `array_repr` | `(arr, max_line_width=None, precision=None, suppress_small=None)` | Return the string representation of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `array_str` | `(a, max_line_width=None, precision=None, suppress_small=None)` | Return a string representation of the data in an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `asin` | `(x, /)` | Inverse sine, element-wise. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `asinh` | `(x, /)` | Inverse hyperbolic sine element-wise. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `astype` | `(x: 'ArrayLike', dtype: 'DTypeLike &#124; None', /, *, copy: 'bool' = False, device: 'xc.Device &#124; Sharding &#124; None' = None) -> 'Array'` | This is implemented via :func:`jax.lax.convert_element_type`, which may | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `atan` | `(x, /)` | Trigonometric inverse tangent, element-wise. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `atan2` | `(x1, x2, /)` | Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `atanh` | `(x, /)` | Inverse hyperbolic tangent element-wise. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `atleast_1d` | `(*arys: 'ArrayLike') -> 'Array &#124; list[Array]'` | Convert inputs to arrays with at least one dimension. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `atleast_2d` | `(*arys: 'ArrayLike') -> 'Array &#124; list[Array]'` | View inputs as arrays with at least two dimensions. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `atleast_3d` | `(*arys: 'ArrayLike') -> 'Array &#124; list[Array]'` | View inputs as arrays with at least three dimensions. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `average` | `(a: 'ArrayLike', axis: 'Axis' = None, weights: 'ArrayLike &#124; None' = None, returned: 'bool' = False, keepdims: 'bool' = False) -> 'Array &#124; tuple[Array, Array]'` | Compute the weighted average along the specified axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `bartlett` | `(M: 'int') -> 'Array'` | Return the Bartlett window. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `bfloat16` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `bincount` | `(x: 'ArrayLike', weights: 'ArrayLike &#124; None' = None, minlength: 'int' = 0, *, length: 'int &#124; None' = None) -> 'Array'` | Count the number of occurrences of each value in an integer array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `bitwise_and` | `(x1, x2, /)` | Compute the bit-wise AND of two arrays element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `bitwise_count` | `(x: 'ArrayLike', /) -> 'Array'` | Counts the number of 1 bits in the binary representation of the absolute value | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `bitwise_invert` | `(x, /)` | Compute bit-wise inversion, or bit-wise NOT, element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `bitwise_left_shift` | `(x1, x2, /)` | Shift the bits of an integer to the left. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `bitwise_not` | `(x, /)` | Compute bit-wise inversion, or bit-wise NOT, element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `bitwise_or` | `(x1, x2, /)` | Compute the bit-wise OR of two arrays element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `bitwise_right_shift` | `(x1: 'ArrayLike', x2: 'ArrayLike', /) -> 'Array'` | Shift the bits of an integer to the right. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `bitwise_xor` | `(x1, x2, /)` | Compute the bit-wise XOR of two arrays element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `blackman` | `(M: 'int') -> 'Array'` | Return the Blackman window. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `block` | `(arrays: 'ArrayLike &#124; list[ArrayLike]') -> 'Array'` | Assemble an nd-array from nested lists of blocks. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `bool` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `bool_` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `broadcast_arrays` | `(*args: 'ArrayLike') -> 'list[Array]'` | Broadcast any number of arrays against each other. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `c_` | `N/A` | Concatenate slices, scalars and array-like objects along the last axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `can_cast` | `(...)` | can_cast(from_, to, casting='safe') | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `cbrt` | `(x, /)` | Return the cube-root of an array, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `cdouble` | `(x: 'Any') -> 'Array'` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `character` | `()` | Abstract base class of all character string scalar types. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `choose` | `(a: 'ArrayLike', choices: 'Sequence[ArrayLike]', out: 'None' = None, mode: 'str' = 'raise') -> 'Array'` | Construct an array from an index array and a list of arrays to choose from. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `column_stack` | `(tup: 'np.ndarray &#124; Array &#124; Sequence[ArrayLike]') -> 'Array'` | Stack 1-D arrays as columns into a 2-D array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `complex128` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `complex64` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `complex_` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `complexfloating` | `()` | Abstract base class of all complex number scalar types that are made up of | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `compress` | `(condition: 'ArrayLike', a: 'ArrayLike', axis: 'int &#124; None' = None, *, size: 'int &#124; None' = None, fill_value: 'ArrayLike' = 0, out: 'None' = None) -> 'Array'` | Compress an array along a given axis using a boolean condition. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `concat` | `(arrays: 'Sequence[ArrayLike]', /, *, axis: 'int &#124; None' = 0) -> 'Array'` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `conj` | `(x: 'ArrayLike', /) -> 'Array'` | Return the complex conjugate, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `conjugate` | `(x: 'ArrayLike', /) -> 'Array'` | Return the complex conjugate, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `convolve` | `(a: 'ArrayLike', v: 'ArrayLike', mode: 'str' = 'full', *, precision: 'PrecisionLike' = None, preferred_element_type: 'DTypeLike &#124; None' = None) -> 'Array'` | Convolution of two one dimensional arrays. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `copy` | `(a: 'ArrayLike', order: 'str &#124; None' = None) -> 'Array'` | Return an array copy of the given object. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `copysign` | `(x1: 'ArrayLike', x2: 'ArrayLike', /) -> 'Array'` | Change the sign of x1 to that of x2, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `corrcoef` | `(x: 'ArrayLike', y: 'ArrayLike &#124; None' = None, rowvar: 'bool' = True) -> 'Array'` | Return Pearson product-moment correlation coefficients. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `correlate` | `(a: 'ArrayLike', v: 'ArrayLike', mode: 'str' = 'valid', *, precision: 'PrecisionLike' = None, preferred_element_type: 'DTypeLike &#124; None' = None) -> 'Array'` | Correlation of two one dimensional arrays. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `count_nonzero` | `(a: 'ArrayLike', axis: 'Axis' = None, keepdims: 'bool' = False) -> 'Array'` | Counts the number of non-zero values in the array ``a``. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `cov` | `(m: 'ArrayLike', y: 'ArrayLike &#124; None' = None, rowvar: 'bool' = True, bias: 'bool' = False, ddof: 'int &#124; None' = None, fweights: 'ArrayLike &#124; None' = None, aweights: 'ArrayLike &#124; None' = None) -> 'Array'` | Estimate a covariance matrix, given data and weights. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `cross` | `(a, b, axisa: 'int' = -1, axisb: 'int' = -1, axisc: 'int' = -1, axis: 'int &#124; None' = None)` | Return the cross product of two (arrays of) vectors. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `csingle` | `(x: 'Any') -> 'Array'` | No docstring available. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `cumprod` | `(a: 'ArrayLike', axis: 'Axis' = None, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None) -> 'Array'` | Return the cumulative product of elements along a given axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `cumulative_sum` | `(x: 'ArrayLike', /, *, axis: 'int &#124; None' = None, dtype: 'DTypeLike &#124; None' = None, include_initial: 'bool' = False) -> 'Array'` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `deg2rad` | `(x: 'ArrayLike', /) -> 'Array'` | Convert angles from degrees to radians. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `degrees` | `(x: 'ArrayLike', /) -> 'Array'` | Convert angles from radians to degrees. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `delete` | `(arr: 'ArrayLike', obj: 'ArrayLike &#124; slice', axis: 'int &#124; None' = None, *, assume_unique_indices: 'bool' = False) -> 'Array'` | Delete entry or entries from an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `diag` | `(v: 'ArrayLike', k: 'int' = 0) -> 'Array'` | Extract a diagonal or construct a diagonal array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `diag_indices` | `(n: 'int', ndim: 'int' = 2) -> 'tuple[Array, ...]'` | Return the indices to access the main diagonal of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `diag_indices_from` | `(arr: 'ArrayLike') -> 'tuple[Array, ...]'` | Return the indices to access the main diagonal of an n-dimensional array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `diagflat` | `(v: 'ArrayLike', k: 'int' = 0) -> 'Array'` | Create a two-dimensional array with the flattened input as a diagonal. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `diagonal` | `(a: 'ArrayLike', offset: 'int' = 0, axis1: 'int' = 0, axis2: 'int' = 1) -> 'Array'` | Return specified diagonals. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `diff` | `(a: 'ArrayLike', n: 'int' = 1, axis: 'int' = -1, prepend: 'ArrayLike &#124; None' = None, append: 'ArrayLike &#124; None' = None) -> 'Array'` | Calculate the n-th discrete difference along the given axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `digitize` | `(x: 'ArrayLike', bins: 'ArrayLike', right: 'bool' = False) -> 'Array'` | Return the indices of the bins to which each value in input array belongs. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `double` | `(x: 'Any') -> 'Array'` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `dtype` | `(...)` | dtype(dtype, align=False, copy=False, [metadata]) | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `e` | `N/A` | Convert a string or number to a floating point number, if possible. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `ediff1d` | `(ary: 'ArrayLike', to_end: 'ArrayLike &#124; None' = None, to_begin: 'ArrayLike &#124; None' = None) -> 'Array'` | The differences between consecutive elements of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `einsum_path` | `(subscripts, /, *operands, optimize: 'bool &#124; str &#124; list[tuple[int, ...]]' = 'auto') -> 'tuple[list[tuple[int, ...]], Any]'` | Evaluates the optimal contraction path without evaluating the einsum. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `equal` | `(x1, x2, /)` | Return (x1 == x2) element-wise. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `euler_gamma` | `N/A` | Convert a string or number to a floating point number, if possible. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `extract` | `(condition: 'ArrayLike', arr: 'ArrayLike', *, size: 'int &#124; None' = None, fill_value: 'ArrayLike' = 0) -> 'Array'` | Return the elements of an array that satisfy a condition. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fabs` | `(x, /)` | Compute the absolute values element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fft` | `N/A` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fill_diagonal` | `(a: 'ArrayLike', val: 'ArrayLike', wrap: 'bool' = False, *, inplace: 'bool' = True) -> 'Array'` | Fill the main diagonal of the given array of any dimensionality. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `finfo` | `(dtype)` | finfo(dtype) | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `fix` | `(x: 'ArrayLike', out: 'None' = None) -> 'Array'` | Round to nearest integer towards zero. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `flatnonzero` | `(a: 'ArrayLike', *, size: 'int &#124; None' = None, fill_value: 'None &#124; ArrayLike &#124; tuple[ArrayLike, ...]' = None) -> 'Array'` | Return indices of nonzero elements in a flattened array | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `flexible` | `()` | Abstract base class of all scalar types without predefined length. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `flip` | `(m: 'ArrayLike', axis: 'int &#124; Sequence[int] &#124; None' = None) -> 'Array'` | Reverse the order of elements of an array along the given axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fliplr` | `(m: 'ArrayLike') -> 'Array'` | Reverse the order of elements of an array along axis 1. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `flipud` | `(m: 'ArrayLike') -> 'Array'` | Reverse the order of elements of an array along axis 0. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `float16` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float32` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float64` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float8_e4m3b11fnuz` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float8_e4m3fn` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float8_e4m3fnuz` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float8_e5m2` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float8_e5m2fnuz` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float_` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `float_power` | `(x1, x2, /)` | First array elements raised to powers from second array, element-wise. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `floating` | `()` | Abstract base class of all floating-point scalar types. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `fmax` | `(x1: 'ArrayLike', x2: 'ArrayLike') -> 'Array'` | Element-wise maximum of array elements. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fmin` | `(x1: 'ArrayLike', x2: 'ArrayLike') -> 'Array'` | Element-wise minimum of array elements. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fmod` | `(x1: 'ArrayLike', x2: 'ArrayLike', /) -> 'Array'` | Returns the element-wise remainder of division. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `frexp` | `(x: 'ArrayLike', /) -> 'tuple[Array, Array]'` | Decompose the elements of x into mantissa and twos exponent. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `from_dlpack` | `(x: 'Any', /, *, device: 'xc.Device &#124; Sharding &#124; None' = None, copy: 'bool &#124; None' = None) -> 'Array'` | Create a NumPy array from an object implementing the ``__dlpack__`` | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `frombuffer` | `(buffer: 'bytes &#124; Any', dtype: 'DTypeLike' = <class 'float'>, count: 'int' = -1, offset: 'int' = 0) -> 'Array'` | Interpret a buffer as a 1-dimensional array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fromfile` | `(*args, **kwargs)` | Unimplemented JAX wrapper for jnp.fromfile. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fromfunction` | `(function: 'Callable[..., Array]', shape: 'Any', *, dtype: 'DTypeLike' = <class 'float'>, **kwargs) -> 'Array'` | Construct an array by executing a function over each coordinate. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fromiter` | `(*args, **kwargs)` | Unimplemented JAX wrapper for jnp.fromiter. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `frompyfunc` | `(func: 'Callable[..., Any]', /, nin: 'int', nout: 'int', *, identity: 'Any' = None) -> 'ufunc'` | Create a JAX ufunc from an arbitrary JAX-compatible scalar function. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `fromstring` | `(string: 'str', dtype: 'DTypeLike' = <class 'float'>, count: 'int' = -1, *, sep: 'str') -> 'Array'` | A new 1-D array initialized from text data in a string. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `gcd` | `(x1: 'ArrayLike', x2: 'ArrayLike') -> 'Array'` | Returns the greatest common divisor of ``&#124;x1&#124;`` and ``&#124;x2&#124;`` | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `generic` | `()` | Base class for numpy scalar types. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `geomspace` | `(start: 'ArrayLike', stop: 'ArrayLike', num: 'int' = 50, endpoint: 'bool' = True, dtype: 'DTypeLike &#124; None' = None, axis: 'int' = 0) -> 'Array'` | Return numbers spaced evenly on a log scale (a geometric progression). | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `get_printoptions` | `()` | Return the current print options. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `gradient` | `(f: 'ArrayLike', *varargs: 'ArrayLike', axis: 'int &#124; Sequence[int] &#124; None' = None, edge_order: 'int &#124; None' = None) -> 'Array &#124; list[Array]'` | Return the gradient of an N-dimensional array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `greater` | `(x1, x2, /)` | Return the truth value of (x1 > x2) element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `greater_equal` | `(x1, x2, /)` | Return the truth value of (x1 >= x2) element-wise. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `hamming` | `(M: 'int') -> 'Array'` | Return the Hamming window. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `hanning` | `(M: 'int') -> 'Array'` | Return the Hanning window. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `heaviside` | `(x1: 'ArrayLike', x2: 'ArrayLike', /) -> 'Array'` | Compute the Heaviside step function. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `histogram` | `(a: 'ArrayLike', bins: 'ArrayLike' = 10, range: 'Sequence[ArrayLike] &#124; None' = None, weights: 'ArrayLike &#124; None' = None, density: 'bool &#124; None' = None) -> 'tuple[Array, Array]'` | Compute the histogram of a dataset. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `histogram2d` | `(x: 'ArrayLike', y: 'ArrayLike', bins: 'ArrayLike &#124; list[ArrayLike]' = 10, range: 'Sequence[None &#124; Array &#124; Sequence[ArrayLike]] &#124; None' = None, weights: 'ArrayLike &#124; None' = None, density: 'bool &#124; None' = None) -> 'tuple[Array, Array, Array]'` | Compute the bi-dimensional histogram of two data samples. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `histogram_bin_edges` | `(a: 'ArrayLike', bins: 'ArrayLike' = 10, range: 'None &#124; Array &#124; Sequence[ArrayLike]' = None, weights: 'ArrayLike &#124; None' = None) -> 'Array'` | Function to calculate only the edges of the bins used by the `histogram` | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `histogramdd` | `(sample: 'ArrayLike', bins: 'ArrayLike &#124; list[ArrayLike]' = 10, range: 'Sequence[None &#124; Array &#124; Sequence[ArrayLike]] &#124; None' = None, weights: 'ArrayLike &#124; None' = None, density: 'bool &#124; None' = None) -> 'tuple[Array, list[Array]]'` | Compute the multidimensional histogram of some data. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `hypot` | `(x1: 'ArrayLike', x2: 'ArrayLike', /) -> 'Array'` | Given the "legs" of a right triangle, return its hypotenuse. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `i0` | `(*args: 'Any', **kwargs: 'Any') -> 'ReturnValue'` | Modified Bessel function of the first kind, order 0. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `iinfo` | `(int_type)` | No docstring available. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `imag` | `(val: 'ArrayLike', /) -> 'Array'` | Return the imaginary part of the complex argument. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `index_exp` | `N/A` | A nicer way to build up index tuples for arrays. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `indices` | `(dimensions: 'Sequence[int]', dtype: 'DTypeLike' = <class 'jax.numpy.int32'>, sparse: 'bool' = False) -> 'Array &#124; tuple[Array, ...]'` | Return an array representing the indices of a grid. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `inexact` | `()` | Abstract base class of all numeric scalar types with a (potentially) | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `insert` | `(arr: 'ArrayLike', obj: 'ArrayLike &#124; slice', values: 'ArrayLike', axis: 'int &#124; None' = None) -> 'Array'` | Insert values along the given axis before the given indices. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `int16` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `int32` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `int4` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `int64` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `int8` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `int_` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `integer` | `()` | Abstract base class of all integer scalar types. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `interp` | `(x: 'ArrayLike', xp: 'ArrayLike', fp: 'ArrayLike', left: 'ArrayLike &#124; str &#124; None' = None, right: 'ArrayLike &#124; str &#124; None' = None, period: 'ArrayLike &#124; None' = None) -> 'Array'` | One-dimensional linear interpolation for monotonically increasing sample points. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `intersect1d` | `(ar1: 'ArrayLike', ar2: 'ArrayLike', assume_unique: 'bool' = False, return_indices: 'bool' = False) -> 'Array &#124; tuple[Array, Array, Array]'` | Compute the set intersection of two 1D arrays. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `invert` | `(x, /)` | Compute bit-wise inversion, or bit-wise NOT, element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `isclose` | `(a: 'ArrayLike', b: 'ArrayLike', rtol: 'ArrayLike' = 1e-05, atol: 'ArrayLike' = 1e-08, equal_nan: 'bool' = False) -> 'Array'` | Returns a boolean array where two arrays are element-wise equal within a | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `iscomplex` | `(x: 'ArrayLike') -> 'Array'` | Returns a bool array, where True if input element is complex. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `iscomplexobj` | `(x: 'Any') -> 'bool'` | Check for a complex type or an array of complex numbers. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `isdtype` | `(dtype: 'DTypeLike', kind: 'str &#124; DTypeLike &#124; tuple[str &#124; DTypeLike, ...]') -> 'bool'` | Returns a boolean indicating whether a provided dtype is of a specified kind. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `isin` | `(element: 'ArrayLike', test_elements: 'ArrayLike', assume_unique: 'bool' = False, invert: 'bool' = False) -> 'Array'` | Determine whether elements in ``element`` appear in ``test_elements``. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `isinf` | `(x: 'ArrayLike', /) -> 'Array'` | Test element-wise for positive or negative infinity. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `isneginf` | `(x, /, out=None)` | Test element-wise for negative infinity, return result as bool array. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `isposinf` | `(x, /, out=None)` | Test element-wise for positive infinity, return result as bool array. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `isreal` | `(x: 'ArrayLike') -> 'Array'` | Returns a bool array, where True if input element is real. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `isrealobj` | `(x: 'Any') -> 'bool'` | Return True if x is a not complex type or an array of complex numbers. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `isscalar` | `(element: 'Any') -> 'bool'` | Returns True if the type of `element` is a scalar type. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `issubdtype` | `(arg1: 'DTypeLike', arg2: 'DTypeLike') -> 'bool'` | Returns True if first argument is a typecode lower/equal in type hierarchy. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `iterable` | `(y)` | Check whether or not an object can be iterated over. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `ix_` | `(*args: 'ArrayLike') -> 'tuple[Array, ...]'` | Return a multi-dimensional grid (open mesh) from N one-dimensional sequences. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `kaiser` | `(M: 'int', beta: 'ArrayLike') -> 'Array'` | Return the Kaiser window. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `kron` | `(a: 'ArrayLike', b: 'ArrayLike') -> 'Array'` | Kronecker product of two arrays. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `lcm` | `(x1: 'ArrayLike', x2: 'ArrayLike') -> 'Array'` | Returns the lowest common multiple of ``&#124;x1&#124;`` and ``&#124;x2&#124;`` | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `ldexp` | `(x1: 'ArrayLike', x2: 'ArrayLike', /) -> 'Array'` | Returns x1 * 2**x2, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `left_shift` | `(x1, x2, /)` | Shift the bits of an integer to the left. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `less` | `(x1, x2, /)` | Return the truth value of (x1 < x2) element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `less_equal` | `(x1, x2, /)` | Return the truth value of (x1 <= x2) element-wise. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `lexsort` | `(keys: 'Array &#124; np.ndarray &#124; Sequence[ArrayLike]', axis: 'int' = -1) -> 'Array'` | Perform an indirect stable sort using a sequence of keys. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `linalg` | `N/A` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `load` | `(*args: 'Any', **kwargs: 'Any') -> 'Array'` | Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `logaddexp` | `(*args: 'Any', **kwargs: 'Any') -> 'ReturnValue'` | Logarithm of the sum of exponentiations of the inputs. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `logaddexp2` | `(*args: 'Any', **kwargs: 'Any') -> 'ReturnValue'` | Logarithm of the sum of exponentiations of the inputs in base-2. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `logical_and` | `(*args)` | Compute the truth value of x1 AND x2 element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `logical_not` | `(*args)` | Compute the truth value of NOT x element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `logical_or` | `(*args)` | Compute the truth value of x1 OR x2 element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `logical_xor` | `(*args)` | Compute the truth value of x1 XOR x2, element-wise. | Logical operations; essential for Keras masking, dropout, and conditional indexing. |
| [x] | `mask_indices` | `(*args, **kwargs)` | Return the indices to access (n, n) arrays, given a masking function. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `matrix_transpose` | `(x: 'ArrayLike', /) -> 'Array'` | Transpose the last two dimensions of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `median` | `(a: 'ArrayLike', axis: 'int &#124; tuple[int, ...] &#124; None' = None, out: 'None' = None, overwrite_input: 'bool' = False, keepdims: 'bool' = False) -> 'Array'` | Compute the median along the specified axis. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `mgrid` | `N/A` | Return dense multi-dimensional "meshgrid". | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `modf` | `(x: 'ArrayLike', /, out=None) -> 'tuple[Array, Array]'` | Return the fractional and integral parts of an array, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `nan_to_num` | `(x: 'ArrayLike', copy: 'bool' = True, nan: 'ArrayLike' = 0.0, posinf: 'ArrayLike &#124; None' = None, neginf: 'ArrayLike &#124; None' = None) -> 'Array'` | Replace NaN with zero and infinity with large finite numbers (default | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanargmax` | `(a: 'ArrayLike', axis: 'int &#124; None' = None, out: 'None' = None, keepdims: 'bool &#124; None' = None) -> 'Array'` | Return the indices of the maximum values in the specified axis ignoring | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanargmin` | `(a: 'ArrayLike', axis: 'int &#124; None' = None, out: 'None' = None, keepdims: 'bool &#124; None' = None) -> 'Array'` | Return the indices of the minimum values in the specified axis ignoring | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nancumprod` | `(a: 'ArrayLike', axis: 'Axis' = None, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None) -> 'Array'` | Return the cumulative product of array elements over a given axis treating Not a | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nancumsum` | `(a: 'ArrayLike', axis: 'Axis' = None, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None) -> 'Array'` | Return the cumulative sum of array elements over a given axis treating Not a | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanmax` | `(a: 'ArrayLike', axis: 'Axis' = None, out: 'None' = None, keepdims: 'bool' = False, initial: 'ArrayLike &#124; None' = None, where: 'ArrayLike &#124; None' = None) -> 'Array'` | Return the maximum of an array or maximum along an axis, ignoring any | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanmean` | `(a: 'ArrayLike', axis: 'Axis' = None, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None, keepdims: 'bool' = False, where: 'ArrayLike &#124; None' = None) -> 'Array'` | Compute the arithmetic mean along the specified axis, ignoring NaNs. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanmedian` | `(a: 'ArrayLike', axis: 'int &#124; tuple[int, ...] &#124; None' = None, out: 'None' = None, overwrite_input: 'bool' = False, keepdims: 'bool' = False) -> 'Array'` | Compute the median along the specified axis, while ignoring NaNs. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanmin` | `(a: 'ArrayLike', axis: 'Axis' = None, out: 'None' = None, keepdims: 'bool' = False, initial: 'ArrayLike &#124; None' = None, where: 'ArrayLike &#124; None' = None) -> 'Array'` | Return minimum of an array or minimum along an axis, ignoring any NaNs. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanpercentile` | `(a: 'ArrayLike', q: 'ArrayLike', axis: 'int &#124; tuple[int, ...] &#124; None' = None, out: 'None' = None, overwrite_input: 'bool' = False, method: 'str' = 'linear', keepdims: 'bool' = False, *, interpolation: 'str &#124; DeprecatedArg' = Deprecated) -> 'Array'` | Compute the qth percentile of the data along the specified axis, | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanprod` | `(a: 'ArrayLike', axis: 'Axis' = None, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None, keepdims: 'bool' = False, initial: 'ArrayLike &#124; None' = None, where: 'ArrayLike &#124; None' = None) -> 'Array'` | Return the product of array elements over a given axis treating Not a | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanquantile` | `(a: 'ArrayLike', q: 'ArrayLike', axis: 'int &#124; tuple[int, ...] &#124; None' = None, out: 'None' = None, overwrite_input: 'bool' = False, method: 'str' = 'linear', keepdims: 'bool' = False, *, interpolation: 'DeprecatedArg &#124; str' = Deprecated) -> 'Array'` | Compute the qth quantile of the data along the specified axis, | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanstd` | `(a: 'ArrayLike', axis: 'Axis' = None, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None, ddof: 'int' = 0, keepdims: 'bool' = False, where: 'ArrayLike &#124; None' = None) -> 'Array'` | Compute the standard deviation along the specified axis, while | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nansum` | `(a: 'ArrayLike', axis: 'Axis' = None, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None, keepdims: 'bool' = False, initial: 'ArrayLike &#124; None' = None, where: 'ArrayLike &#124; None' = None) -> 'Array'` | Return the sum of array elements over a given axis treating Not a | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `nanvar` | `(a: 'ArrayLike', axis: 'Axis' = None, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None, ddof: 'int' = 0, keepdims: 'bool' = False, where: 'ArrayLike &#124; None' = None) -> 'Array'` | Compute the variance along the specified axis, while ignoring NaNs. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `ndim` | `(a)` | Return the number of dimensions of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `newaxis` | `N/A` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `nextafter` | `(x1, x2, /)` | Return the next floating-point value after x1 towards x2, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `nonzero` | `(a: 'ArrayLike', *, size: 'int &#124; None' = None, fill_value: 'None &#124; ArrayLike &#124; tuple[ArrayLike, ...]' = None) -> 'tuple[Array, ...]'` | Return indices of nonzero elements of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `not_equal` | `(x1, x2, /)` | Return (x1 != x2) element-wise. | Crucial for numerical stability and Keras masking / NaN-safe losses. |
| [x] | `number` | `()` | Abstract base class of all numeric scalar types. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `object_` | `(...)` | Any Python object. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `ogrid` | `N/A` | Return open multi-dimensional "meshgrid". | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `packbits` | `(a: 'ArrayLike', axis: 'int &#124; None' = None, bitorder: 'str' = 'big') -> 'Array'` | Packs the elements of a binary-valued array into bits in a uint8 array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `partition` | `(a: 'ArrayLike', kth: 'int', axis: 'int' = -1) -> 'Array'` | Returns a partially-sorted copy of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `percentile` | `(a: 'ArrayLike', q: 'ArrayLike', axis: 'int &#124; tuple[int, ...] &#124; None' = None, out: 'None' = None, overwrite_input: 'bool' = False, method: 'str' = 'linear', keepdims: 'bool' = False, *, interpolation: 'str &#124; DeprecatedArg' = Deprecated) -> 'Array'` | Compute the q-th percentile of the data along the specified axis. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `permute_dims` | `(a: 'ArrayLike', /, axes: 'tuple[int, ...]') -> 'Array'` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `piecewise` | `(x: 'ArrayLike', condlist: 'Array &#124; Sequence[ArrayLike]', funclist: 'list[ArrayLike &#124; Callable[..., Array]]', *args, **kw) -> 'Array'` | Evaluate a piecewise-defined function. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `place` | `(arr: 'ArrayLike', mask: 'ArrayLike', vals: 'ArrayLike', *, inplace: 'bool' = True) -> 'Array'` | Change elements of an array based on conditional and input values. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `poly` | `(seq_of_zeros: 'Array') -> 'Array'` | Find the coefficients of a polynomial with the given sequence of roots. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `polyadd` | `(a1: 'Array', a2: 'Array') -> 'Array'` | Find the sum of two polynomials. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `polyder` | `(p: 'Array', m: 'int' = 1) -> 'Array'` | Return the derivative of the specified order of a polynomial. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `polydiv` | `(u: 'ArrayLike', v: 'ArrayLike', *, trim_leading_zeros: 'bool' = False) -> 'tuple[Array, Array]'` | Returns the quotient and remainder of polynomial division. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `polyfit` | `(x: 'Array', y: 'Array', deg: 'int', rcond: 'float &#124; None' = None, full: 'bool' = False, w: 'Array &#124; None' = None, cov: 'bool' = False) -> 'Array &#124; tuple[Array, ...]'` | Least squares polynomial fit. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `polyint` | `(p: 'Array', m: 'int' = 1, k: 'int &#124; None' = None) -> 'Array'` | Return an antiderivative (indefinite integral) of a polynomial. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `polymul` | `(a1: 'ArrayLike', a2: 'ArrayLike', *, trim_leading_zeros: 'bool' = False) -> 'Array'` | Find the product of two polynomials. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `polysub` | `(a1: 'Array', a2: 'Array') -> 'Array'` | Difference (subtraction) of two polynomials. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `polyval` | `(p: 'Array', x: 'Array', *, unroll: 'int' = 16) -> 'Array'` | Evaluate a polynomial at specific values. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `pow` | `(x1: 'ArrayLike', x2: 'ArrayLike', /) -> 'Array'` | First array elements raised to powers from second array, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `printoptions` | `(*args, **kwargs)` | Context manager for setting print options. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `promote_types` | `(a: 'DTypeLike', b: 'DTypeLike') -> 'DType'` | Returns the type to which a binary operation should cast its arguments. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `ptp` | `(a: 'ArrayLike', axis: 'Axis' = None, out: 'None' = None, keepdims: 'bool' = False) -> 'Array'` | Range of values (maximum - minimum) along an axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `put` | `(a: 'ArrayLike', ind: 'ArrayLike', v: 'ArrayLike', mode: 'str &#124; None' = None, *, inplace: 'bool' = True) -> 'Array'` | Replaces specified elements of an array with given values. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `quantile` | `(a: 'ArrayLike', q: 'ArrayLike', axis: 'int &#124; tuple[int, ...] &#124; None' = None, out: 'None' = None, overwrite_input: 'bool' = False, method: 'str' = 'linear', keepdims: 'bool' = False, *, interpolation: 'DeprecatedArg &#124; str' = Deprecated) -> 'Array'` | Compute the q-th quantile of the data along the specified axis. | Statistical reduction; backbone of Keras normalization layers (e.g. BatchNorm, LayerNorm). |
| [x] | `r_` | `N/A` | Concatenate slices, scalars and array-like objects along the first axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `rad2deg` | `(x: 'ArrayLike', /) -> 'Array'` | Convert angles from radians to degrees. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `radians` | `(x: 'ArrayLike', /) -> 'Array'` | Convert angles from degrees to radians. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `ravel_multi_index` | `(multi_index: 'Sequence[ArrayLike]', dims: 'Sequence[int]', mode: 'str' = 'raise', order: 'str' = 'C') -> 'Array'` | Convert multi-dimensional indices into flat indices. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `real` | `(val: 'ArrayLike', /) -> 'Array'` | Return the real part of the complex argument. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `reciprocal` | `(x: 'ArrayLike', /) -> 'Array'` | Return the reciprocal of the argument, element-wise. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `resize` | `(a: 'ArrayLike', new_shape: 'Shape') -> 'Array'` | Return a new array with the specified shape. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `result_type` | `(*args: 'Any') -> 'DType'` | Returns the type that results from applying the NumPy | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `right_shift` | `(x1: 'ArrayLike', x2: 'ArrayLike', /) -> 'Array'` | Right shift the bits of ``x1`` to the amount specified in ``x2``. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `roll` | `(a: 'ArrayLike', shift: 'ArrayLike &#124; Sequence[int]', axis: 'int &#124; Sequence[int] &#124; None' = None) -> 'Array'` | Roll array elements along a given axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `rollaxis` | `(a: 'ArrayLike', axis: 'int', start: 'int' = 0) -> 'Array'` | Roll the specified axis to a given position. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `roots` | `(p: 'ArrayLike', *, strip_zeros: 'bool' = True) -> 'Array'` | Return the roots of a polynomial with coefficients given in p. | Polynomial operations; used in advanced regularizers or custom metrics. |
| [x] | `rot90` | `(m: 'ArrayLike', k: 'int' = 1, axes: 'tuple[int, int]' = (0, 1)) -> 'Array'` | Rotate an array by 90 degrees in the plane specified by axes. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `round` | `(a: 'ArrayLike', decimals: 'int' = 0, out: 'None' = None) -> 'Array'` | Round an array to the given number of decimals. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `round_` | `(a: 'ArrayLike', decimals: 'int' = 0, out: 'None' = None) -> 'Array'` | Round an array to the given number of decimals. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `s_` | `N/A` | A nicer way to build up index tuples for arrays. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `save` | `(file, arr, allow_pickle=True, fix_imports=True)` | Save an array to a binary file in NumPy ``.npy`` format. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `savez` | `(file, *args, **kwds)` | Save several arrays into a single file in uncompressed ``.npz`` format. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `searchsorted` | `(a: 'ArrayLike', v: 'ArrayLike', side: 'str' = 'left', sorter: 'ArrayLike &#124; None' = None, *, method: 'str' = 'scan') -> 'Array'` | Perform a binary search within a sorted array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `select` | `(condlist: 'Sequence[ArrayLike]', choicelist: 'Sequence[ArrayLike]', default: 'ArrayLike' = 0) -> 'Array'` | Return an array drawn from elements in choicelist, depending on conditions. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `set_printoptions` | `(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None, sign=None, floatmode=None, *, legacy=None)` | Set printing options. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `setdiff1d` | `(ar1: 'ArrayLike', ar2: 'ArrayLike', assume_unique: 'bool' = False, *, size: 'int &#124; None' = None, fill_value: 'ArrayLike &#124; None' = None) -> 'Array'` | Compute the set difference of two 1D arrays. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `setxor1d` | `(ar1: 'ArrayLike', ar2: 'ArrayLike', assume_unique: 'bool' = False) -> 'Array'` | Compute the set-wise xor of elements in two arrays. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `signbit` | `(x: 'ArrayLike', /) -> 'Array'` | Returns element-wise True where signbit is set (less than zero). | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `signedinteger` | `()` | Abstract base class of all signed integer scalar types. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `sinc` | `(x: 'ArrayLike', /) -> 'Array'` | Return the normalized sinc function. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `single` | `(x: 'Any') -> 'Array'` | No docstring available. | Trigonometric operation; critical for activation functions and signal processing. |
| [x] | `size` | `(a, axis=None)` | Return the number of elements along a given axis. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `sort` | `(a: 'ArrayLike', axis: 'int &#124; None' = -1, *, kind: 'None' = None, order: 'None' = None, stable: 'bool' = True, descending: 'bool' = False) -> 'Array'` | Return a sorted copy of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `sort_complex` | `(a: 'ArrayLike') -> 'Array'` | Sort a complex array using the real part first, then the imaginary part. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `trace` | `(a: 'ArrayLike', offset: 'int &#124; ArrayLike' = 0, axis1: 'int' = 0, axis2: 'int' = 1, dtype: 'DTypeLike &#124; None' = None, out: 'None' = None) -> 'Array'` | Return the sum along diagonals of the array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `trapezoid` | `(y: 'ArrayLike', x: 'ArrayLike &#124; None' = None, dx: 'ArrayLike' = 1.0, axis: 'int' = -1) -> 'Array'` | Integrate along the given axis using the composite trapezoidal rule. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `tri` | `(N: 'int', M: 'int &#124; None' = None, k: 'int' = 0, dtype: 'DTypeLike &#124; None' = None) -> 'Array'` | An array with ones at and below the given diagonal and zeros elsewhere. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `tril` | `(m: 'ArrayLike', k: 'int' = 0) -> 'Array'` | Lower triangle of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `tril_indices` | `(n: 'int', k: 'int' = 0, m: 'int &#124; None' = None) -> 'tuple[Array, Array]'` | Return the indices for the lower-triangle of an (n, m) array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `tril_indices_from` | `(arr: 'ArrayLike', k: 'int' = 0) -> 'tuple[Array, Array]'` | Return the indices for the lower-triangle of arr. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `trim_zeros` | `(filt, trim='fb')` | Trim the leading and/or trailing zeros from a 1-D array or sequence. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `triu` | `(m: 'ArrayLike', k: 'int' = 0) -> 'Array'` | Upper triangle of an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `triu_indices` | `(n: 'int', k: 'int' = 0, m: 'int &#124; None' = None) -> 'tuple[Array, Array]'` | Return the indices for the upper-triangle of an (n, m) array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `triu_indices_from` | `(arr: 'ArrayLike', k: 'int' = 0) -> 'tuple[Array, Array]'` | Return the indices for the upper-triangle of arr. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `ufunc` | `(func: 'Callable[..., Any]', /, nin: 'int', nout: 'int', *, name: 'str &#124; None' = None, nargs: 'int &#124; None' = None, identity: 'Any' = None, update_doc=False)` | Functions that operate element-by-element on whole arrays. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `uint` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `uint16` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `uint32` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `uint4` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `uint64` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `uint8` | `(x: 'Any') -> 'Array'` | No docstring available. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `union1d` | `(ar1: 'ArrayLike', ar2: 'ArrayLike', *, size: 'int &#124; None' = None, fill_value: 'ArrayLike &#124; None' = None) -> 'Array'` | Compute the set union of two 1D arrays. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unique` | `(ar: 'ArrayLike', return_index: 'bool' = False, return_inverse: 'bool' = False, return_counts: 'bool' = False, axis: 'int &#124; None' = None, *, equal_nan: 'bool' = True, size: 'int &#124; None' = None, fill_value: 'ArrayLike &#124; None' = None)` | Return the unique values from an array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unique_all` | `(x: 'ArrayLike', /, *, size: 'int &#124; None' = None, fill_value: 'ArrayLike &#124; None' = None) -> '_UniqueAllResult'` | Return unique values from x, along with indices, inverse indices, and counts. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unique_counts` | `(x: 'ArrayLike', /, *, size: 'int &#124; None' = None, fill_value: 'ArrayLike &#124; None' = None) -> '_UniqueCountsResult'` | Return unique values from x, along with counts. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unique_inverse` | `(x: 'ArrayLike', /, *, size: 'int &#124; None' = None, fill_value: 'ArrayLike &#124; None' = None) -> '_UniqueInverseResult'` | Return unique values from x, along with indices, inverse indices, and counts. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unique_values` | `(x: 'ArrayLike', /, *, size: 'int &#124; None' = None, fill_value: 'ArrayLike &#124; None' = None) -> 'Array'` | Return unique values from x, along with indices, inverse indices, and counts. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unpackbits` | `(a: 'ArrayLike', axis: 'int &#124; None' = None, count: 'int &#124; None' = None, bitorder: 'str' = 'big') -> 'Array'` | Unpacks elements of a uint8 array into a binary-valued output array. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unravel_index` | `(indices: 'ArrayLike', shape: 'Shape') -> 'tuple[Array, ...]'` | Convert flat indices into multi-dimensional indices. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unsignedinteger` | `()` | Abstract base class of all unsigned integer scalar types. | DType casting/creation; fundamental to Keras mixed-precision and state allocation. |
| [x] | `unstack` | `(x: 'ArrayLike', /, *, axis: 'int' = 0) -> 'tuple[Array, ...]'` | No docstring available. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `unwrap` | `(p: 'ArrayLike', discont: 'ArrayLike &#124; None' = None, axis: 'int' = -1, period: 'ArrayLike' = 6.283185307179586) -> 'Array'` | Unwrap by taking the complement of large deltas with respect to the period. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `vander` | `(x: 'ArrayLike', N: 'int &#124; None' = None, increasing: 'bool' = False) -> 'Array'` | Generate a Vandermonde matrix. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `vecdot` | `(x1: 'ArrayLike', x2: 'ArrayLike', /, *, axis: 'int' = -1, precision: 'PrecisionLike' = None, preferred_element_type: 'DTypeLike &#124; None' = None) -> 'Array'` | Perform a conjugate multiplication of two batched vectors. | Standard JAX/NumPy API required for generic array manipulation. |
| [x] | `vectorize` | `(pyfunc, *, excluded=frozenset(), signature=None)` | Define a vectorized function with broadcasting. | Standard JAX/NumPy API required for generic array manipulation. |
