# Flax (NNX) Compatibility Implementation Plan

This document tracks the explicit `flax.nnx` namespace primitives, architectural components, and layers that `ml-switcheroo-compiler` must successfully trace, execute, and update state for. This checklist assumes that underlying JAX array operations (`JAX_TODO_PLAN.md`) and optimizers (`OPTAX_TODO_PLAN.md`) are tracked separately.

The focus here is entirely on structural compatibility, state management (PyTree flattening of modules), and high-level layer verification.

## 1. Core Architecture & State Management
The `flax.nnx` paradigm centers on Python objects that manage mutable state variables, which are flattened into functional purity at the compiler boundary. The compiler must correctly understand these boundaries.

| Status | Name | Class Signature | Docstring | Compiler Support Notes |
| :---: | :--- | :--- | :--- | :--- |
| [x] | `Module` | `class Module:` | Base class for all neural network modules in NNX. | Compiler must support extracting dynamic `__dict__` state into tracing context and re-injecting variables post-execution. |
| [ ] | `GraphDef` | `class GraphDef(cls, ...):` | Represents the static structure of a Module graph. | Compiler PyTree handlers must correctly separate this static definition from dynamic variables. |
| [x] | `State` | `class State(dict[str, Any]):` | A nested dictionary structure for `nnx` variables. | Compiler must support structural traversal, updates, and PyTree flattening of this custom dictionary type. |
| [ ] | `Variable` | `class Variable:` | A base class representing a stateful variable in the framework. | Compiler needs to track metadata (type, shape, dtype) wrapping the underlying `ProxyTensor`. |
| [ ] | `Param` | `class Param(Variable):` | A variable representing a trainable parameter (e.g., weights/biases). | Compiler must compute gradients exclusively for nodes flagged with this type. |
| [ ] | `BatchStat` | `class BatchStat(Variable):` | A variable representing non-trainable state (e.g., moving averages). | Compiler must trace mutations (EMA updates) to this node *without* requiring gradients. |
| [ ] | `Rng` | `class Rng(Variable):` | A variable holding a random number generator stream. | Compiler must manage dynamic PRNG key splitting and state updates seamlessly. |

## 2. Linear & Dense Layers
Verification that the compiler can properly fuse and optimize common linear transformations mapped from `nnx`.

| Status | Name | Class Signature | Docstring | Compiler Support Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `Dense` | `Dense(in_features: int, out_features: int, use_bias: bool)` | A standard linear transformation layer. | Standard `dot` + `add` operation. Ensure weight initialization matches `zero_jax.nn.initializers`. |
| [x] | `Linear` | `Linear(in_features: int, out_features: int, use_bias: bool)` | Alias/variant of standard linear (dense) layer. | Same structural compiler requirements as `Dense`. |
| [ ] | `LinearGeneral` | `LinearGeneral(in_features: int\|Seq, out_features: int\|Seq)` | A general linear transformation layer (N-dimensional). | Exercises the compiler's `tensordot` primitive over varied axes mappings. |
| [x] | `Einsum` | `Einsum(einsum_str: str, kernel_shape: tuple)` | A module that performs a linear transformation using an einsum equation. | Compiler must fully support Einstein summation notation parsing and execution. |
| [ ] | `LoRA` | `LoRA(...)` | Low-Rank Adaptation injection module. | Compiler must support freezing primary `Param` nodes while training LoRA weights. |
| [ ] | `LoRALinear` | `LoRALinear(in_features: int, out_features: int)` | A pre-configured Linear layer with LoRA adapters. | Verifies composite graph creation (Base Linear + LoRA bypass). |

## 3. Convolutional Layers
Spatial feature extraction layers.

| Status | Name | Class Signature | Docstring | Compiler Support Notes |
| :---: | :--- | :--- | :--- | :--- |
| [x] | `Conv` | `Conv(in_features: int, out_features: int, kernel_size: tuple, strides: tuple, padding: str\|tuple)` | A general n-dimensional convolutional layer. | The compiler must accurately map this to `conv_general_dilated` taking into account arbitrary NCHW/NHWC data formats. |
| [ ] | `ConvTranspose`| `ConvTranspose(in_features: int, out_features: int, kernel_size: tuple)` | A general n-dimensional transposed convolution. | Verifies the `conv_transpose` operation and associated spatial upsampling padding rules. |

## 4. Attention & Embeddings
Sequence and lookup modeling.

| Status | Name | Class Signature | Docstring | Compiler Support Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `Embed` | `Embed(num_embeddings: int, features: int)` | A simple lookup table that stores embeddings of a fixed dictionary. | Compiler must efficiently support `take`/`gather` operations without expanding OHE matrices. |
| [ ] | `MultiHeadAttention` | `MultiHeadAttention(num_heads: int, qkv_features: int)` | Standard Multi-Head Attention implementation. | Verifies complex `reshape` -> `transpose` -> `dot` -> `softmax` sequence execution. |
| [ ] | `MultiHeadDotProductAttention` | `MultiHeadDotProductAttention(num_heads: int, qkv_features: int)` | Core attention kernel without projection layers. | Must verify numerical stability via stable `softmax` and masked attention implementations. |

## 5. Normalization Layers
Testing mutable graph variables and reduction statistics.

| Status | Name | Class Signature | Docstring | Compiler Support Notes |
| :---: | :--- | :--- | :--- | :--- |
| [x] | `BatchNorm` | `BatchNorm(num_features: int, use_running_average: bool)` | Batch Normalization layer. | **Crucial:** Compiler must intercept the running `mean`/`var` mutations and push the updated `BatchStat` back to the Python object. |
| [x] | `LayerNorm` | `LayerNorm(num_features: int, reduction_axes: int)` | Layer Normalization layer. | Verifies `mean` and `var` reductions over specific spatial/channel dimensions. |
| [ ] | `RMSNorm` | `RMSNorm(num_features: int)` | Root Mean Square Normalization layer. | Verifies efficient, un-centered variance normalization (`rsqrt(mean(x^2))`). |

## 6. Stochastic Layers
Testing PRNG stream consumption and masking.

| Status | Name | Class Signature | Docstring | Compiler Support Notes |
| :---: | :--- | :--- | :--- | :--- |
| [x] | `Dropout` | `Dropout(rate: float, rng_collection: str)` | A dropout layer (randomly zeroing elements). | Compiler must consume `rng_collection` (usually 'dropout'), split the key, and trace the bernoulli mask. |

## 7. Containers
Graph composition verification.

| Status | Name | Class Signature | Docstring | Compiler Support Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `Sequential` | `Sequential(*layers)` | Applies a sequence of modules sequentially. | Compiler must flatten the sequential chain into a single continuous compute graph. |
| [ ] | `List` | `List(modules: Iterable[Module])` | A module that holds a list of sub-modules. | Verification of list-based PyTree state extraction. |
| [ ] | `Dict` | `Dict(modules: dict[str, Module])` | A module that holds a dictionary of sub-modules. | Verification of dict-based PyTree state extraction. |

## 8. Functional API / Transforms
Flax `nnx` exposes lifted versions of JAX transformations that automatically handle `State` extraction and injection across transform boundaries.

| Status | Name | Class Signature | Docstring | Compiler Support Notes |
| :---: | :--- | :--- | :--- | :--- |
| [ ] | `Jit` | `Jit(module_constructor: Callable)` | JIT-compiles the execution of a sub-module. | Compiler must support sub-graph JIT boundaries and passing mutable state in/out of the AOT trace. |
| [x] | `Vmap` | `Vmap(module_constructor: Callable)` | Vectorizes the execution of a sub-module. | Needs to lift the state parameters across the batch dimension. |
| [x] | `Scan` | `Scan(module_constructor: Callable)` | Loops over a sequence of inputs, maintaining module state. | Essential for RNNs. Compiler must manage tied parameters while evolving `BatchStat` or `Rng` states. |
| [ ] | `Remat` | `Remat(module_constructor: Callable)` | Checkpoints a sub-module to save memory during backprop. | Compiler must intercept this and drop the forward pass activations from the tape, recomputing them during reverse mode. |
| [x] | `Pmap` | `Pmap(module_constructor: Callable)` | Parallelizes module execution across multiple devices. | State needs to be sharded or broadcast appropriately. |
