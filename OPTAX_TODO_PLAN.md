# ml-switcheroo-compiler Todo Plan for zero-optax Compliance

This document outlines the exhaustive list of array operations, math primitives, neural network functions, and algorithmic abstractions that `ml-switcheroo-compiler` (and its Python frontend shim `ml_switcheroo`) must implement to achieve 100% semantic and syntactic testing parity between `zero-optax` and the reference `optax` library.

## Core Array Creation and Manipulation (`ml_switcheroo.jnp`)

| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `jnp.array` | `(object: Any, dtype: Optional[Any] = None) -> Array` | Creates an array. | Must support converting Python lists/tuples and scalars. |
| [ ] | `jnp.asarray` | `(a: Any, dtype: Optional[Any] = None) -> Array` | Converts the input to an array. | No-op if `a` is already an Array of the matching dtype. Essential for coercing inputs in loss functions. |
| [x] | `jnp.zeros` | `(shape: Union[int, Sequence[int]], dtype: Optional[Any] = None) -> Array` | Returns a new array of given shape and type, filled with zeros. | Used heavily in state initialization and padding logic. |
| [x] | `jnp.zeros_like` | `(a: Any, dtype: Optional[Any] = None) -> Array` | Returns an array of zeros with the same shape and type as a given array. | Used in optimizer state initialization (e.g., Adam `mu` and `nu`). |
| [x] | `jnp.full_like` | `(a: Any, fill_value: Any, dtype: Optional[Any] = None) -> Array` | Returns a full array with the same shape and type as a given array. | |
| [ ] | `jnp.expand_dims` | `(a: Array, axis: Union[int, Sequence[int]]) -> Array` | Expands the shape of an array. | Crucial for broadcasting shapes before mathematical operations. |
| [x] | `jnp.squeeze` | `(a: Array, axis: Optional[Union[int, Sequence[int]]] = None) -> Array` | Removes single-dimensional entries from the shape of an array. | |
| [x] | `jnp.take_along_axis` | `(arr: Array, indices: Array, axis: int) -> Array` | Takes values from the input array by matching 1d index and data slices. | Used in ranking and sparsemax implementations. |
| [x] | `jnp.where` | `(condition: Array, x: Optional[Array] = None, y: Optional[Array] = None) -> Array` | Return elements chosen from `x` or `y` depending on `condition`. | Extremely important for Huber loss, safe softmax, and piecewise schedules. |
| [x] | `jnp.inf` | `float` | IEEE 754 floating point representation of (positive) infinity. | Constant required for safe masking (e.g., `-jnp.inf` in masked softmax). |
| [ ] | `jnp.ndarray` | `class` | Base array class. | Needed for type checking `isinstance(x, jnp.ndarray)`. |

## Mathematical Primitives (`ml_switcheroo.jnp`)

| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `jnp.abs` | `(x: Array) -> Array` | Calculate the absolute value element-wise. | Used in Huber loss, L1 regularization. |
| [x] | `jnp.exp` | `(x: Array) -> Array` | Calculate the exponential of all elements in the input array. | Required for softmax, cross-entropy, scheduling. |
| [x] | `jnp.log` | `(x: Array) -> Array` | Natural logarithm, element-wise. | Required for cross-entropy formulations. |
| [x] | `jnp.log1p` | `(x: Array) -> Array` | Return the natural logarithm of one plus the input array, element-wise. | Used for numerically stable computations. |
| [x] | `jnp.sqrt` | `(x: Array) -> Array` | Return the non-negative square-root of an array, element-wise. | Heavily used in Adam, Adagrad, RMSProp denominators. |
| [x] | `jnp.square` | `(x: Array) -> Array` | Return the element-wise square of the input. | Used in squared error, Adam second moment tracking. |
| [x] | `jnp.sign` | `(x: Array) -> Array` | Returns an element-wise indication of the sign of a number. | |
| [ ] | Operator Overloads | `+, -, *, /, **` | Arithmetic operations. | The Array class must support standard Python magic methods for arithmetic with correct broadcasting. |

## Reductions & Aggregations (`ml_switcheroo.jnp`)

| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `jnp.sum` | `(a: Array, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Array` | Sum of array elements over a given axis. | Fundamental for loss aggregation. |
| [x] | `jnp.mean` | `(a: Array, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Array` | Compute the arithmetic mean along the specified axis. | Fundamental for loss aggregation across batches. |
| [x] | `jnp.max` | `(a: Array, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Array` | Return the maximum of an array or maximum along an axis. | Used in safe softmax (subtracting max before exp). |
| [x] | `jnp.minimum` | `(x1: Array, x2: Array) -> Array` | Element-wise minimum of array elements. | Used for clipping bounds. |
| [x] | `jnp.maximum` | `(x1: Array, x2: Array) -> Array` | Element-wise maximum of array elements. | Used in hinge losses and bounds clamping. |

## Logical & Element-wise Operations (`ml_switcheroo.jnp`)

| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `jnp.any` | `(a: Array, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Array` | Test whether any array element along a given axis evaluates to True. | |
| [x] | `jnp.bitwise_or` | `(x1: Array, x2: Array) -> Array` | Compute the bit-wise OR of two arrays element-wise. | Used for binary mask combination. |
| [x] | `jnp.floor` | `(x: Array) -> Array` | Return the floor of the input, element-wise. | Used in schedule phase computations. |
| [ ] | Logical Operators | `>, <, >=, <=, ==, !=` | Comparison operations. | The Array class must support standard Python magic methods for comparisons, returning boolean arrays. |

## Neural Network Primitives (`ml_switcheroo.jnn` / `zero_jax.nn`)

| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `jnn.one_hot` | `(x: Array, num_classes: int, dtype: Optional[Any] = None) -> Array` | One-hot encodes the given integer indices. | Required for `softmax_cross_entropy_with_integer_labels`. |
| [x] | `jnn.sigmoid` | `(x: Array) -> Array` | Sigmoid activation function. | `1 / (1 + exp(-x))` |
| [x] | `jnn.logsumexp` | `(a: Array, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False, b: Optional[Array] = None) -> Array` | Log-sum-exp reduction. | Essential for numerical stability in cross-entropy and softmax. |
| [ ] | `jnn.log_sigmoid` | `(x: Array) -> Array` | Log-sigmoid activation function. | `log(sigmoid(x))` implemented stably. |

## Advanced & Algorithmic Primitives

| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `jnp.searchsorted` | `(a: Array, v: Array, side: str = 'left') -> Array` | Find indices where elements should be inserted to maintain order. | Crucial for piecewise schedules. |
| [ ] | `jax.lax.scan` | `(f: Callable, init: Any, xs: Any, length: Optional[int] = None) -> Tuple[Any, Any]` | Scan a function over leading array axes while carrying along state. | Mandatory for the CTC loss forward-backward algorithm. |
| [ ] | `jnp.sort` | `(a: Array, axis: int = -1) -> Array` | Return a sorted copy of an array. | Required for sparsemax projection logic. |

## Tree Utilities (`zero_jax.tree_util`)

| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `tree_map` | `(f: Callable, tree: Any, *rest: Any) -> Any` | Maps a multi-argument function over a collection of PyTrees. | The backbone of optax; maps optimizer updates over model weights. |
| [ ] | `tree_flatten` | `(tree: Any) -> Tuple[List[Any], Any]` | Flattens a PyTree into a list of leaves and a treedef. | Used in `inject_hyperparams`. |
| [ ] | `tree_unflatten` | `(treedef: Any, leaves: List[Any]) -> Any` | Reconstructs a PyTree from a treedef and a list of leaves. | Used in `inject_hyperparams`. |

## Automatic Differentiation (`ml_switcheroo` core)

| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `grad` | `(fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable` | Creates a function that evaluates the gradient of `fun`. | Necessary for testing `jax.grad(zero_optax.loss) == optax.loss_grad()`. |
| [ ] | `custom_jvp` / `custom_vjp` | `(fun: Callable) -> Callable` | Set up a custom Jacobian-vector product or Vector-Jacobian product rule. | Required if complex operations (like CTC loss) need manually specified gradients for stability. |

## Python `math` Module Analogs
While pure Python `math` functions are used in pure-function schedules, if they are ever pushed into JAX traces, their JAX equivalents are required.
| Checkbox | Name | Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `math.floor` / `jnp.floor` | `(x: float) -> int` | Round down to nearest integer. | Used in exponential decay. |
| [x] | `math.pow` / `jnp.power` | `(x: float, y: float) -> float` | Exponentiation. | Used in polynomial schedules. |
| [x] | `math.cos` / `jnp.cos` | `(x: float) -> float` | Cosine. | Used extensively in cosine annealing and one-cycle schedules. |
| [ ] | `math.pi` / `jnp.pi` | `float` | Pi constant. | Used in cosine schedules. |
