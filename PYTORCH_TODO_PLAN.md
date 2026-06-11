# `ml-switcheroo-compiler` PyTorch Implementation Plan (Detailed)

This document outlines the exhaustive checklist of everything `ml-switcheroo-compiler` needs to implement so that tests between the `zero-pytorch` shim and official `pytorch` pass 100% syntactically and semantically.


## 1. Core Compiler Engine & Data Structures

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `TracerTape` | `class TracerTape()` | Manages concurrent, thread-safe AOT tracing via `threading.local` | Core tracking for operations |
| [x] | `ProxyTensor` | `class ProxyTensor()` | Intercepts eager operations using Python dunder overloads | e.g. `__add__`, `__mul__`, `__matmul__` |
| [ ] | `ShapeInference` | `def infer_shape(a_shape, b_shape)` | Numpy-compatible static shape inference for complex broadcast logic | Crucial for validation pre-compile |
| [x] | `EagerMode` | `class EagerModeFallback()` | Runtime evaluation fallback relying purely on `numpy` | Allows stepping through without compiling |

## 2. Core Tensor Operations (via ProxyTensor / `ml-switcheroo.core`)
*All operations must support both eager `numpy` execution and AST node tracing recording.*

### Unary Math

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `abs` | `(*args, **kwargs)` | Applies the abs operation. | Map to IR Node |
| [x] | `acos` | `(*args, **kwargs)` | Applies the acos operation. | Map to IR Node |
| [x] | `acosh` | `(*args, **kwargs)` | Applies the acosh operation. | Map to IR Node |
| [x] | `asin` | `(*args, **kwargs)` | Applies the asin operation. | Map to IR Node |
| [x] | `asinh` | `(*args, **kwargs)` | Applies the asinh operation. | Map to IR Node |
| [x] | `atan` | `(*args, **kwargs)` | Applies the atan operation. | Map to IR Node |
| [x] | `atanh` | `(*args, **kwargs)` | Applies the atanh operation. | Map to IR Node |
| [x] | `cbrt` | `(*args, **kwargs)` | Applies the cbrt operation. | Map to IR Node |
| [x] | `ceil` | `(*args, **kwargs)` | Applies the ceil operation. | Map to IR Node |
| [x] | `cos` | `(*args, **kwargs)` | Applies the cos operation. | Map to IR Node |
| [x] | `cosh` | `(*args, **kwargs)` | Applies the cosh operation. | Map to IR Node |
| [x] | `deg2rad` | `(*args, **kwargs)` | Applies the deg2rad operation. | Map to IR Node |
| [x] | `digamma` | `(*args, **kwargs)` | Applies the digamma operation. | Map to IR Node |
| [x] | `erf` | `(*args, **kwargs)` | Applies the erf operation. | Map to IR Node |
| [x] | `erfc` | `(*args, **kwargs)` | Applies the erfc operation. | Map to IR Node |
| [ ] | `erfinv` | `(*args, **kwargs)` | Applies the erfinv operation. | Map to IR Node |
| [x] | `exp` | `(*args, **kwargs)` | Applies the exp operation. | Map to IR Node |
| [x] | `exp2` | `(*args, **kwargs)` | Applies the exp2 operation. | Map to IR Node |
| [x] | `expm1` | `(*args, **kwargs)` | Applies the expm1 operation. | Map to IR Node |
| [x] | `fix` | `(*args, **kwargs)` | Applies the fix operation. | Map to IR Node |
| [x] | `floor` | `(*args, **kwargs)` | Applies the floor operation. | Map to IR Node |
| [x] | `isfinite` | `(*args, **kwargs)` | Applies the isfinite operation. | Map to IR Node |
| [x] | `isinf` | `(*args, **kwargs)` | Applies the isinf operation. | Map to IR Node |
| [x] | `isnan` | `(*args, **kwargs)` | Applies the isnan operation. | Map to IR Node |
| [x] | `lgamma` | `(*args, **kwargs)` | Applies the lgamma operation. | Map to IR Node |
| [x] | `log` | `(*args, **kwargs)` | Applies the log operation. | Map to IR Node |
| [x] | `log10` | `(*args, **kwargs)` | Applies the log10 operation. | Map to IR Node |
| [x] | `log1p` | `(*args, **kwargs)` | Applies the log1p operation. | Map to IR Node |
| [x] | `log2` | `(*args, **kwargs)` | Applies the log2 operation. | Map to IR Node |
| [x] | `logical_not` | `(*args, **kwargs)` | Applies the logical_not operation. | Map to IR Node |
| [ ] | `logit` | `(*args, **kwargs)` | PyTorch `logit` equivalent | Map to IR Node |
| [ ] | `mvlgamma` | `(*args, **kwargs)` | PyTorch `mvlgamma` equivalent | Map to IR Node |
| [ ] | `nan_to_num` | `(*args, **kwargs)` | PyTorch `nan_to_num` equivalent | Map to IR Node |
| [x] | `neg` | `(*args, **kwargs)` | PyTorch `neg` equivalent | Map to IR Node |
| [x] | `positive` | `(*args, **kwargs)` | Applies the positive operation. | Map to IR Node |
| [x] | `rad2deg` | `(*args, **kwargs)` | Applies the rad2deg operation. | Map to IR Node |
| [x] | `reciprocal` | `(*args, **kwargs)` | Applies the reciprocal operation. | Map to IR Node |
| [x] | `round` | `(*args, **kwargs)` | Applies the round operation. | Map to IR Node |
| [x] | `rsqrt` | `(*args, **kwargs)` | Applies the rsqrt operation. | Map to IR Node |
| [x] | `sigmoid` | `(*args, **kwargs)` | PyTorch `sigmoid` equivalent | Map to IR Node |
| [x] | `sign` | `(*args, **kwargs)` | Applies the sign operation. | Map to IR Node |
| [ ] | `signbit` | `(*args, **kwargs)` | PyTorch `signbit` equivalent | Map to IR Node |
| [x] | `sin` | `(*args, **kwargs)` | Applies the sin operation. | Map to IR Node |
| [x] | `sinc` | `(*args, **kwargs)` | Applies the sinc operation. | Map to IR Node |
| [x] | `sinh` | `(*args, **kwargs)` | Applies the sinh operation. | Map to IR Node |
| [x] | `sqrt` | `(*args, **kwargs)` | Applies the sqrt operation. | Map to IR Node |
| [x] | `square` | `(*args, **kwargs)` | Applies the square operation. | Map to IR Node |
| [x] | `tan` | `(*args, **kwargs)` | Applies the tan operation. | Map to IR Node |
| [x] | `tanh` | `(*args, **kwargs)` | Applies the tanh operation. | Map to IR Node |
| [x] | `trunc` | `(*args, **kwargs)` | Applies the trunc operation. | Map to IR Node |


### Binary Math

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `add` | `(*args, **kwargs)` | Applies the add operation. | Map to IR Node |
| [x] | `atan2` | `(*args, **kwargs)` | Applies the atan2 operation. | Map to IR Node |
| [x] | `bitwise_and` | `(*args, **kwargs)` | Applies the bitwise_and operation. | Map to IR Node |
| [x] | `bitwise_or` | `(*args, **kwargs)` | Applies the bitwise_or operation. | Map to IR Node |
| [x] | `bitwise_xor` | `(*args, **kwargs)` | Applies the bitwise_xor operation. | Map to IR Node |
| [x] | `copysign` | `(*args, **kwargs)` | Applies the copysign operation. | Map to IR Node |
| [x] | `divide` | `(*args, **kwargs)` | Applies the divide operation. | Map to IR Node |
| [x] | `float_power` | `(*args, **kwargs)` | Applies the float_power operation. | Map to IR Node |
| [x] | `floor_divide` | `(*args, **kwargs)` | Applies the floor_divide operation. | Map to IR Node |
| [x] | `fmax` | `(*args, **kwargs)` | Applies the fmax operation. | Map to IR Node |
| [x] | `fmin` | `(*args, **kwargs)` | Applies the fmin operation. | Map to IR Node |
| [x] | `fmod` | `(*args, **kwargs)` | Applies the fmod operation. | Map to IR Node |
| [x] | `gcd` | `(*args, **kwargs)` | Applies the gcd operation. | Map to IR Node |
| [x] | `greater` | `(*args, **kwargs)` | Applies the greater operation. | Map to IR Node |
| [x] | `greater_equal` | `(*args, **kwargs)` | Applies the greater_equal operation. | Map to IR Node |
| [x] | `heaviside` | `(*args, **kwargs)` | Applies the heaviside operation. | Map to IR Node |
| [x] | `hypot` | `(*args, **kwargs)` | Applies the hypot operation. | Map to IR Node |
| [x] | `lcm` | `(*args, **kwargs)` | Applies the lcm operation. | Map to IR Node |
| [x] | `ldexp` | `(*args, **kwargs)` | Applies the ldexp operation. | Map to IR Node |
| [x] | `left_shift` | `(*args, **kwargs)` | Applies the left_shift operation. | Map to IR Node |
| [x] | `less` | `(*args, **kwargs)` | Applies the less operation. | Map to IR Node |
| [x] | `less_equal` | `(*args, **kwargs)` | Applies the less_equal operation. | Map to IR Node |
| [x] | `logaddexp` | `(*args, **kwargs)` | Applies the logaddexp operation. | Map to IR Node |
| [x] | `logaddexp2` | `(*args, **kwargs)` | Applies the logaddexp2 operation. | Map to IR Node |
| [x] | `logical_and` | `(*args, **kwargs)` | Applies the logical_and operation. | Map to IR Node |
| [x] | `logical_or` | `(*args, **kwargs)` | Applies the logical_or operation. | Map to IR Node |
| [x] | `logical_xor` | `(*args, **kwargs)` | Applies the logical_xor operation. | Map to IR Node |
| [x] | `maximum` | `(*args, **kwargs)` | Applies the maximum operation. | Map to IR Node |
| [x] | `minimum` | `(*args, **kwargs)` | Applies the minimum operation. | Map to IR Node |
| [x] | `mod` | `(*args, **kwargs)` | Applies the mod operation. | Map to IR Node |
| [x] | `multiply` | `(*args, **kwargs)` | Applies the multiply operation. | Map to IR Node |
| [x] | `nextafter` | `(*args, **kwargs)` | Applies the nextafter operation. | Map to IR Node |
| [x] | `not_equal` | `(*args, **kwargs)` | Applies the not_equal operation. | Map to IR Node |
| [x] | `power` | `(*args, **kwargs)` | Applies the power operation. | Map to IR Node |
| [x] | `remainder` | `(*args, **kwargs)` | Applies the remainder operation. | Map to IR Node |
| [x] | `right_shift` | `(*args, **kwargs)` | Applies the right_shift operation. | Map to IR Node |
| [x] | `subtract` | `(*args, **kwargs)` | Applies the subtract operation. | Map to IR Node |
| [ ] | `true_divide` | `(*args, **kwargs)` | PyTorch `true_divide` equivalent | Map to IR Node |
| [ ] | `xlogy` | `(*args, **kwargs)` | PyTorch `xlogy` equivalent | Map to IR Node |


### Reductions

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `all` | `(*args, **kwargs)` | Applies the all operation. | Map to IR Node |
| [x] | `any` | `(*args, **kwargs)` | Applies the any operation. | Map to IR Node |
| [x] | `argmax` | `(*args, **kwargs)` | Applies the argmax operation. | Map to IR Node |
| [x] | `argmin` | `(*args, **kwargs)` | Applies the argmin operation. | Map to IR Node |
| [x] | `count_nonzero` | `(*args, **kwargs)` | Applies the count_nonzero operation. | Map to IR Node |
| [x] | `logsumexp` | `(*args, **kwargs)` | Applies the logsumexp operation. | Map to IR Node |
| [x] | `max` | `(*args, **kwargs)` | Applies the max operation. | Map to IR Node |
| [x] | `mean` | `(*args, **kwargs)` | Applies the mean operation. | Map to IR Node |
| [x] | `min` | `(*args, **kwargs)` | Applies the min operation. | Map to IR Node |
| [x] | `norm` | `(*args, **kwargs)` | Applies the norm operation. | Map to IR Node |
| [x] | `prod` | `(*args, **kwargs)` | Applies the prod operation. | Map to IR Node |
| [x] | `std` | `(*args, **kwargs)` | Applies the std operation. | Map to IR Node |
| [x] | `sum` | `(*args, **kwargs)` | Applies the sum operation. | Map to IR Node |
| [x] | `variance` | `(*args, **kwargs)` | Applies the variance operation. | Map to IR Node |


### Creation Ops

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `arange` | `(*args, **kwargs)` | Applies the arange operation. | Map to IR Node |
| [x] | `empty` | `(*args, **kwargs)` | Applies the empty operation. | Map to IR Node |
| [x] | `eye` | `(*args, **kwargs)` | Applies the eye operation. | Map to IR Node |
| [x] | `full` | `(*args, **kwargs)` | Applies the full operation. | Map to IR Node |
| [x] | `full_like` | `(*args, **kwargs)` | Applies the full_like operation. | Map to IR Node |
| [x] | `identity` | `(*args, **kwargs)` | Applies the identity operation. | Map to IR Node |
| [x] | `linspace` | `(*args, **kwargs)` | Applies the linspace operation. | Map to IR Node |
| [x] | `meshgrid` | `(*args, **kwargs)` | Applies the meshgrid operation. | Map to IR Node |
| [x] | `ones` | `(*args, **kwargs)` | Applies the ones operation. | Map to IR Node |
| [x] | `ones_like` | `(*args, **kwargs)` | Applies the ones_like operation. | Map to IR Node |
| [x] | `zeros` | `(*args, **kwargs)` | Applies the zeros operation. | Map to IR Node |
| [x] | `zeros_like` | `(*args, **kwargs)` | Applies the zeros_like operation. | Map to IR Node |


### Linear Algebra (Linalg)

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `cholesky` | `(*args, **kwargs)` | Applies the cholesky operation. | Map to IR Node |
| [ ] | `det` | `(*args, **kwargs)` | Applies the det operation. | Map to IR Node |
| [x] | `diag` | `(*args, **kwargs)` | Applies the diag operation. | Map to IR Node |
| [x] | `dot` | `(*args, **kwargs)` | Applies the dot operation. | Map to IR Node |
| [ ] | `eigh` | `(*args, **kwargs)` | Applies the eigh operation. | Map to IR Node |
| [x] | `eigvalsh` | `(*args, **kwargs)` | Applies the eigvalsh operation. | Map to IR Node |
| [x] | `inner` | `(*args, **kwargs)` | Applies the inner operation. | Map to IR Node |
| [ ] | `inv` | `(*args, **kwargs)` | Applies the inv operation. | Map to IR Node |
| [x] | `matmul` | `(*args, **kwargs)` | Applies the matmul operation. | Map to IR Node |
| [x] | `matrix_power` | `(*args, **kwargs)` | Applies the matrix_power operation. | Map to IR Node |
| [x] | `outer` | `(*args, **kwargs)` | Applies the outer operation. | Map to IR Node |
| [ ] | `pinv` | `(*args, **kwargs)` | Applies the pinv operation. | Map to IR Node |
| [ ] | `qr` | `(*args, **kwargs)` | Applies the qr operation. | Map to IR Node |
| [x] | `slogdet` | `(*args, **kwargs)` | Applies the slogdet operation. | Map to IR Node |
| [ ] | `svd` | `(*args, **kwargs)` | Applies the svd operation. | Map to IR Node |
| [x] | `tensordot` | `(*args, **kwargs)` | Applies the tensordot operation. | Map to IR Node |
| [x] | `vdot` | `(*args, **kwargs)` | Applies the vdot operation. | Map to IR Node |


### Shape and Indexing

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `broadcast_to` | `(*args, **kwargs)` | Applies the broadcast_to operation. | Map to IR Node |
| [x] | `concatenate` | `(*args, **kwargs)` | Applies the concatenate operation. | Map to IR Node |
| [x] | `expand` | `(*args, **kwargs)` | Applies the expand operation. | Map to IR Node |
| [x] | `flatten` | `(*args, **kwargs)` | Applies the flatten operation. | Map to IR Node |
| [x] | `gather` | `(*args, **kwargs)` | Applies the gather operation. | Map to IR Node |
| [x] | `moveaxis` | `(*args, **kwargs)` | Applies the moveaxis operation. | Map to IR Node |
| [x] | `permute` | `(*args, **kwargs)` | Applies the permute operation. | Map to IR Node |
| [x] | `repeat` | `(*args, **kwargs)` | Applies the repeat operation. | Map to IR Node |
| [x] | `reshape` | `(*args, **kwargs)` | Applies the reshape operation. | Map to IR Node |
| [x] | `roll` | `(*args, **kwargs)` | Applies the roll operation. | Map to IR Node |
| [x] | `shape` | `(*args, **kwargs)` | Applies the shape operation. | Map to IR Node |
| [x] | `split` | `(*args, **kwargs)` | Applies the split operation. | Map to IR Node |
| [x] | `squeeze` | `(*args, **kwargs)` | Applies the squeeze operation. | Map to IR Node |
| [x] | `stack` | `(*args, **kwargs)` | Applies the stack operation. | Map to IR Node |
| [x] | `swapaxes` | `(*args, **kwargs)` | Applies the swapaxes operation. | Map to IR Node |
| [x] | `take` | `(*args, **kwargs)` | Applies the take operation. | Map to IR Node |
| [x] | `tile` | `(*args, **kwargs)` | Applies the tile operation. | Map to IR Node |
| [x] | `transpose` | `(*args, **kwargs)` | Applies the transpose operation. | Map to IR Node |
| [x] | `unsqueeze` | `(*args, **kwargs)` | Applies the unsqueeze operation. | Map to IR Node |
| [x] | `unstack` | `(*args, **kwargs)` | Applies the unstack operation. | Map to IR Node |
| [x] | `where` | `(*args, **kwargs)` | Applies the where operation. | Map to IR Node |
| [x] | `tril` | `(*args, **kwargs)` | Applies the tril operation. | Map to IR Node |
| [x] | `triu` | `(*args, **kwargs)` | Applies the triu operation. | Map to IR Node |


### Random Number Generation

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `rand` | `(*args, **kwargs)` | PyTorch `rand` equivalent | Map to IR Node |
| [ ] | `randn` | `(*args, **kwargs)` | PyTorch `randn` equivalent | Map to IR Node |
| [ ] | `randint` | `(*args, **kwargs)` | PyTorch `randint` equivalent | Map to IR Node |
| [x] | `seed` | `(*args, **kwargs)` | PyTorch `seed` equivalent | Map to IR Node |
| [ ] | `manual_seed` | `(*args, **kwargs)` | PyTorch `manual_seed` equivalent | Map to IR Node |


## 3. Reverse-Mode AD Engine (`compiler.grad`)

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `TopologicalSort` | `def topsort(node)` | Topological Graph Sorting algorithm | Dependency resolution for backprop |
| [ ] | `VJPs` | `def vjp(node, grad_out)` | Defines exact mathematical VJPs for all Operations | Jacobian-vector products |
| [ ] | `GradientAccumulation` | `def accumulate_grads(grads)` | Gradient Accumulation routines for nodes with multiple dependents | Resolves fan-out scenarios |
| [ ] | `AutogradMatching` | `def verify_autograd()` | Provide strict tensor equality checks and chaining rules | Ensures match with `zero-pytorch.autograd` |

## 4. Tensor State & Autograd Utilities

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `backward` | `Tensor.backward(*args, **kwargs)` | Triggers the reverse-mode auto-differentiation | Maps to `compiler.grad` |
| [ ] | `view` | `Tensor.view(*shape)` | Returns a new tensor with the same data but different size | Aliased shape logic |
| [ ] | `contiguous` | `Tensor.contiguous()` | Returns a contiguous in memory tensor | Ensures memory layout |
| [ ] | `item` | `Tensor.item()` | Returns the value of this tensor as a standard Python number | Forces eager evaluation |
| [ ] | `detach` | `Tensor.detach()` | Returns a new Tensor, detached from the current graph | Stops tracing/gradient flow |
| [ ] | `no_grad` | `class no_grad()` | Context manager that disabled gradient calculation | Disables `TracerTape` recording |
| [ ] | `set_grad_enabled` | `class set_grad_enabled(mode)` | Context manager for gradient calculation | Controls `TracerTape` state |

## 5. Neural Network Primitives (`ml_switcheroo.nn`)
*These operations are dynamically bound by `zero-pytorch.nn.functional` and constitute the semantic layers of PyTorch.*

### Convolutions

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `conv1d` | `(*args, **kwargs)` | Applies the conv1d operation. | Lowered from module |
| [x] | `conv2d` | `(*args, **kwargs)` | Applies the conv2d operation. | Lowered from module |
| [x] | `conv3d` | `(*args, **kwargs)` | Applies the conv3d operation. | Lowered from module |
| [x] | `conv_transpose1d` | `(*args, **kwargs)` | Applies the conv_transpose1d operation. | Lowered from module |
| [x] | `conv_transpose2d` | `(*args, **kwargs)` | Applies the conv_transpose2d operation. | Lowered from module |
| [x] | `conv_transpose3d` | `(*args, **kwargs)` | Applies the conv_transpose3d operation. | Lowered from module |


### Pooling

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `avg_pool1d` | `(*args, **kwargs)` | Applies the avg_pool1d operation. | Lowered from module |
| [x] | `avg_pool2d` | `(*args, **kwargs)` | Applies the avg_pool2d operation. | Lowered from module |
| [x] | `avg_pool3d` | `(*args, **kwargs)` | Applies the avg_pool3d operation. | Lowered from module |
| [x] | `max_pool1d` | `(*args, **kwargs)` | Applies the max_pool1d operation. | Lowered from module |
| [x] | `max_pool2d` | `(*args, **kwargs)` | Applies the max_pool2d operation. | Lowered from module |
| [x] | `max_pool3d` | `(*args, **kwargs)` | Applies the max_pool3d operation. | Lowered from module |
| [ ] | `adaptive_avg_pool1d` | `(*args, **kwargs)` | PyTorch `nn.functional.adaptive_avg_pool1d` equivalent | Lowered from module |
| [x] | `adaptive_avg_pool2d` | `(*args, **kwargs)` | Applies the adaptive_avg_pool2d operation. | Lowered from module |
| [ ] | `adaptive_avg_pool3d` | `(*args, **kwargs)` | PyTorch `nn.functional.adaptive_avg_pool3d` equivalent | Lowered from module |
| [ ] | `adaptive_max_pool1d` | `(*args, **kwargs)` | PyTorch `nn.functional.adaptive_max_pool1d` equivalent | Lowered from module |
| [ ] | `adaptive_max_pool2d` | `(*args, **kwargs)` | PyTorch `nn.functional.adaptive_max_pool2d` equivalent | Lowered from module |
| [ ] | `adaptive_max_pool3d` | `(*args, **kwargs)` | PyTorch `nn.functional.adaptive_max_pool3d` equivalent | Lowered from module |
| [x] | `fractional_max_pool2d` | `(*args, **kwargs)` | Applies the fractional_max_pool2d operation. | Lowered from module |


### Normalization

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `batch_norm` | `(*args, **kwargs)` | Applies the batch_norm operation. | Lowered from module |
| [x] | `layer_norm` | `(*args, **kwargs)` | Applies the layer_norm operation. | Lowered from module |
| [x] | `group_norm` | `(*args, **kwargs)` | Applies the group_norm operation. | Lowered from module |
| [x] | `instance_norm` | `(*args, **kwargs)` | Applies the instance_norm operation. | Lowered from module |
| [x] | `rms_norm` | `(*args, **kwargs)` | Applies the rms_norm operation. | Lowered from module |


### Activations

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `celu` | `(*args, **kwargs)` | Applies the celu operation. | Lowered from module |
| [x] | `elu` | `(*args, **kwargs)` | Applies the elu operation. | Lowered from module |
| [x] | `gelu` | `(*args, **kwargs)` | Applies the gelu operation. | Lowered from module |
| [x] | `glu` | `(*args, **kwargs)` | Applies the glu operation. | Lowered from module |
| [x] | `hardswish` | `(*args, **kwargs)` | Applies the hardswish operation. | Lowered from module |
| [x] | `leaky_relu` | `(*args, **kwargs)` | Applies the leaky_relu operation. | Lowered from module |
| [x] | `mish` | `(*args, **kwargs)` | Applies the mish operation. | Lowered from module |
| [x] | `relu` | `(*args, **kwargs)` | Applies the relu operation. | Lowered from module |
| [x] | `selu` | `(*args, **kwargs)` | Applies the selu operation. | Lowered from module |
| [x] | `sigmoid` | `(*args, **kwargs)` | Applies the sigmoid operation. | Lowered from module |
| [x] | `softmax` | `(*args, **kwargs)` | Applies the softmax operation. | Lowered from module |
| [x] | `log_softmax` | `(*args, **kwargs)` | Applies the log_softmax operation. | Lowered from module |
| [x] | `softplus` | `(*args, **kwargs)` | Applies the softplus operation. | Lowered from module |
| [x] | `swish` | `(*args, **kwargs)` | Applies the swish operation. | Lowered from module |
| [x] | `tanh` | `(*args, **kwargs)` | Applies the tanh operation. | Lowered from module |


### Loss Primitives

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `adaptive_log_softmax_with_loss` | `(*args, **kwargs)` | PyTorch `nn.functional.adaptive_log_softmax_with_loss` equivalent | Lowered from module |


### Dropout

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `dropout` | `(*args, **kwargs)` | Applies the dropout operation. | Lowered from module |
| [x] | `alpha_dropout` | `(*args, **kwargs)` | Applies the alpha_dropout operation. | Lowered from module |
| [x] | `feature_alpha_dropout` | `(*args, **kwargs)` | Applies the feature_alpha_dropout operation. | Lowered from module |
| [x] | `spatial_dropout` | `(*args, **kwargs)` | Applies the spatial_dropout operation. | Lowered from module |


### Recurrent & Attention

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `rnn_cell` | `(*args, **kwargs)` | Applies the rnn_cell operation. | Lowered from module |
| [x] | `lstm_cell` | `(*args, **kwargs)` | Applies the lstm_cell operation. | Lowered from module |
| [x] | `gru_cell` | `(*args, **kwargs)` | Applies the gru_cell operation. | Lowered from module |
| [x] | `scaled_dot_product_attention` | `(*args, **kwargs)` | Applies the scaled_dot_product_attention operation. | Lowered from module |


### Utilities & Vision

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `complex` | `(*args, **kwargs)` | Applies the complex operation. | Lowered from module |
| [x] | `embedding` | `(*args, **kwargs)` | Applies the embedding operation. | Lowered from module |
| [x] | `pad` | `(*args, **kwargs)` | Applies the pad operation. | Lowered from module |
| [x] | `upsample_bilinear` | `(*args, **kwargs)` | Applies the upsample_bilinear operation. | Lowered from module |
| [x] | `upsample_nearest` | `(*args, **kwargs)` | Applies the upsample_nearest operation. | Lowered from module |


## 6. Compiler Passes & Optimizations

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `lift_state` | `def lift_state(graph)` | Dynamically lifts mutable states into functional inputs/outputs | Handles `torch.nn.Parameter` updates |
| [ ] | `OptimizerSupport` | `def lift_optimizer_state(graph)` | Traces optimizer state updates (momentum, velocity, steps) | Stateful updates for Optimizers |
| [ ] | `DCE` | `def dead_code_elimination(graph)` | Prune unreferenced operations | Memory / compute optimization |
| [ ] | `CSE` | `def common_subexpression_elimination(graph)` | De-duplicate identically computed nodes | Prevents redundant ops |
| [ ] | `IR_Translation` | `def to_logical_graph(tape)` | Output cleanly to `ml-switcheroo-ir` compliant `LogicalGraph` | Must tie back to user AST refs |