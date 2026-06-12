# Chex Compiler Implementation Plan (Exhaustive)

This document serves as the exhaustive roadmap for everything `ml-switcheroo-compiler` must implement to support `zero-chex`. The goal is to guarantee that the `zero-chex` shim achieves 100% semantic and syntactic test-passing parity with the official `google-deepmind/chex` library when running on the `ml-switcheroo` backend.

---

## 1. Core IR & Execution Engine (Jittable Assertions)
`chexify` wraps JAX's `checkify`, meaning assertions can be embedded inside compiled JIT graphs. The compiler must support deferred error emission and token-threading to prevent assertion-dropping during optimization.

*   **Assertion & Check Nodes (IR Lowering):** The IR must implement an `Assert(condition, error_trace)` or `CheckAndThrow` Logical Node. Because standard XLA/Switcheroo graphs are purely functional, these nodes must utilize a **Token Threading** mechanism. The `Assert` node takes an execution token and returns a new token, ensuring Dead Code Elimination (DCE) does not optimize away pure math assertions that have no tensor outputs.
*   **Async Assertion Status:** To support `async_check=True`, the IR must be able to emit a non-blocking device-to-host (D2H) status queue for assertions. The runtime API must expose a method queryable via `block_until_chexify_assertions_complete` to drain this queue.
*   **Dynamic Control Flow Integration:** Assertions placed inside `If` / `Cond` / `While` branches must only evaluate when the control flow path is taken. The token threading must correctly merge across `Cond` boundaries using `AfterAll` or `Tuple` ops.
*   **Dynamic Shape Support:** Support `Shape` extraction statically (at trace time via `ShapedArray`) or dynamically (in IR via a `GetDimensionSize` op) for assertions like `assert_axis_dimension`.

## 2. Tracing Context & Transformation Manipulation
Chex frequently checks whether functions are traced, limits trace counts, and patches transformation behaviors for testing.

*   **Trace Counters:** The compiler's frontend tracing engine (the `zero-jax` bridge) must expose a global thread-safe trace counter API. Every time a `core.jaxpr` is built, this counter increments. This satisfies `assert_max_traces` and `clear_trace_counter`.
*   **Transformation Bypassing:**
    *   `fake_jit`: The execution engine must support bypassing the JIT optimization pipeline. The compiler frontend must intercept the `@jit` decorator and execute eagerly, or run the unoptimized trace-graph via a direct C++ interpreter loop.
    *   `fake_pmap`: The compiler must map parallel mapped (`pmap`) axes to vectorized (`vmap`) loops or standard `scan`/`while` loops. The compiler's IR must resolve `AxisName` primitives correctly without requiring actual multi-device communication (CollectiveOps).
*   **Global Toggles:** Native frontend hooks must respect `disable_asserts` and `enable_asserts`, effectively short-circuiting trace-time and compile-time IR emission for all `assert_*` methods.

## 3. Hardware, Devices, and Topology Context
Chex tests hardware allocation and distribution. The compiler's runtime must emulate or accurately expose hardware traits via a PJRT-compatible abstraction.

*   **Device Interrogation:** Respond correctly to `assert_devices_available` queries by exposing standard topological metadata (e.g., node, socket, core ID mapping) mirroring `jax.devices()`.
*   **Backend & Topology Mocking:** Expose backend flags for `assert_gpu_available` and `assert_tpu_available`. Crucially, implement host device emulation (forcing the compiler to spawn *N* CPU worker threads) to satisfy `set_n_cpu_devices`.
*   **Memory Space Tags:** IR Tensor wrappers (`Array`) must carry physical memory placement tags (e.g., `HBM`, `SRAM`, `DRAM`) to support `assert_tree_is_on_device` and `assert_tree_is_on_host`.

## 4. Sharding and SPMD
*   **Sharding Specifications:** The compiler's array objects must expose `sharding` attributes (implementing `OpSharding` protobuf equivalents) to satisfy `assert_tree_is_sharded`.
*   **Cross-Replica Semantics:** The typing system must distinguish `ArrayBatched`, `ArrayDevice`, and `ArraySharded`, ensuring these semantic wrappers survive trace-time operations and lower to proper GSPMD partitioning in the backend.

## 5. Type System, Mathematics, & PyTree Registration
*   **PyTree Flattening:** The compiler must feature a robust, C++-backed `PyTreeDef` structure mirroring JAX's `tree_util` for performant flattening before tracing. This includes supporting `register_dataclass_type_with_jax_tree_util`.
*   **Autodiff (`jvp`/`vjp`):** The compiler must provide exact forward and reverse mode automatic differentiation capabilities (calculating analytical Jacobians) so `assert_numerical_grads` produces matching IR against numerical perturbations.
*   **Precision Lowering:** Implement `allclose` IR nodes. Critically, implement exact Units in the Last Place (ULP) bit-level comparisons for `assert_trees_all_close_ulp`.

---

## Exhaustive API Checklist & Compiler Requirements Matrix

### Table 1: Tensor, Shape & Rank Assertions
These require deep integration with the compiler's Shape Inference and Tracing systems.

| [x] | Name | Function Signature | Docstring | Compiler Implementation Notes |
|---|---|---|---|---|
| [ ] | `assert_axis_dimension` | `(tensor, axis: int, expected: int)` | Checks that `tensor.shape[axis] == expected`. | Requires IR to support `ShapeIndex` extraction. Must resolve at trace-time for static shapes, and lower to an `AssertEq` graph node for dynamic shapes. |
| [ ] | `assert_axis_dimension_comparator` | `(tensor, axis: int, pass_fn: Callable, error_string: str)` | Asserts that `pass_fn(tensor.shape[axis])` passes. | If dynamic shapes are used, `pass_fn` must be traceable into IR, otherwise it must execute statically during the frontend pass. |
| [ ] | `assert_axis_dimension_gt` | `(tensor, axis: int, val: int)` | Checks that `tensor.shape[axis] > val`. | Lowers to `Greater(GetDimSize(tensor, axis), val)`. |
| [ ] | `assert_axis_dimension_gteq` | `(tensor, axis: int, val: int)` | Checks that `tensor.shape[axis] >= val`. | Lowers to `GreaterEqual(GetDimSize(tensor, axis), val)`. |
| [ ] | `assert_axis_dimension_lt` | `(tensor, axis: int, val: int)` | Checks that `tensor.shape[axis] < val`. | Lowers to `Less(GetDimSize(tensor, axis), val)`. |
| [ ] | `assert_axis_dimension_lteq` | `(tensor, axis: int, val: int)` | Checks that `tensor.shape[axis] <= val`. | Lowers to `LessEqual(GetDimSize(tensor, axis), val)`. |
| [x] | `assert_equal_rank` | `(inputs: Sequence[Array])` | Checks that all arrays have the same rank. | Ranks are always static in XLA. Evaluates strictly at trace-time using `tensor.ndim`. |
| [ ] | `assert_equal_shape` | `(inputs: Sequence[Array], dims: int\|Seq=None)` | Checks that all arrays have the same shape. | If dynamic shapes exist, lowers to multiple `AssertEq` nodes. Requires broadcasting logic to validate. |
| [x] | `assert_equal_shape_prefix` | `(inputs: Sequence[Array], prefix_len: int)` | Checks that the leading `prefix_dims` dims of all inputs have same shape. | Trace-time array slicing `shape[:prefix_len]`. |
| [x] | `assert_equal_shape_suffix` | `(inputs: Sequence[Array], suffix_len: int)` | Checks that the final `suffix_len` dims of all inputs have same shape. | Trace-time array slicing `shape[-suffix_len:]`. |
| [x] | `assert_equal_size` | `(inputs: Sequence[Array])` | Checks that all arrays have the same size. | Calculates mathematical product of dimensions. |
| [x] | `assert_rank` | `(inputs, expected_ranks)` | Checks that the rank of all inputs matches expected. | Strict frontend check. |
| [x] | `assert_shape` | `(inputs, expected_shapes)` | Checks that the shape of all inputs matches expected. | Trace-time resolution over `ellipsis` logic. |
| [x] | `assert_size` | `(inputs, expected_sizes)` | Checks that the size of all inputs matches expected. | Trace-time validation against expected total element count. |
| [x] | `assert_type` | `(inputs, expected_types)` | Checks that the type of all inputs matches. | Must perfectly align with `zero_jax` dtype representations, including complex types. |

### Table 2: PyTree Assertions
The compiler's `tree_util` module must handle these operations identically to JAX C++ bindings.

| [x] | Name | Function Signature | Docstring | Compiler Implementation Notes |
|---|---|---|---|---|
| [ ] | `assert_tree_all_finite` | `(tree_like)` | Checks that all leaves in a tree are finite. | Must flatten tree, map `IsFinite` IR nodes, and lower a global `ReduceAll` (logical AND) across all leaves. |
| [x] | `assert_tree_has_only_ndarrays` | `(tree)` | Checks that all leaves are n-dimensional arrays. | Frontend trace-time check against the compiler's `DeviceArray` or `Tracer` classes. |
| [x] | `assert_tree_shape_prefix` | `(tree, shape_prefix)` | Checks that all leaves' shapes have the same prefix. | Trace-time verification post-flatten. |
| [x] | `assert_tree_shape_suffix` | `(tree, shape_suffix)` | Checks that all leaves' shapes have the same suffix. | Trace-time verification post-flatten. |
| [x] | `assert_tree_no_nones` | `(tree)` | Checks that a tree does not contain `None`. | Frontend flattening validation. |
| [ ] | `assert_trees_all_close` | `(trees, rtol=1e-06, atol=0.0)` | Checks that all trees have leaves with approx equal values. | Must compile to an `Abs(A - B) <= Atol + Rtol * Abs(B)` IR subgraph. |
| [ ] | `assert_trees_all_close_ulp` | `(trees, maxulp=1)` | Checks that tree leaves differ by at most `maxulp` ULP. | Requires low-level `BitcastConvertType` IR nodes to integer arrays for exact IEEE 754 bit-level ULP distance logic. |
| [ ] | `assert_trees_all_equal` | `(trees, strict=False)` | Checks that all trees have leaves with exactly equal values. | Lowers to strict `Equal` IR nodes and a global `ReduceAll`. |
| [x] | `assert_trees_all_equal_comparator`| `(equality_comparator, ...)`| Checks that all trees are equal as per custom comparator. | If compiled, `equality_comparator` must be a valid, traceable compiler function. |
| [x] | `assert_trees_all_equal_dtypes` | `(trees)` | Checks that trees' leaves have the same dtype. | Trace-time metadata check. |
| [x] | `assert_trees_all_equal_shapes` | `(trees)` | Checks that trees have same structure and leaves' shapes. | Combines `treedef` comparison and trace-time shape checks. |
| [x] | `assert_trees_all_equal_shapes_and_dtypes` | `(trees)` | Checks same structure, shape, and dtype. | Combined frontend metadata check. |
| [x] | `assert_trees_all_equal_sizes` | `(trees)` | Checks same structure and leaves' sizes. | Combined frontend metadata check. |
| [x] | `assert_trees_all_equal_structs` | `(trees)` | Checks that trees have the same structure. | Direct comparison of `PyTreeDef` hashes. |

### Table 3: Hardware & Device Assertions
Requires a mockable PJRT device topology within the compiler backend.

| [x] | Name | Function Signature | Docstring | Compiler Implementation Notes |
|---|---|---|---|---|
| [ ] | `assert_devices_available` | `(n: int, devtype: str, backend=None, not_less_than=False)` | Checks that `n` devices of type are available. | Must interrogate the compiler's device runtime topology APIs. |
| [ ] | `assert_gpu_available` | `(backend=None)` | Checks that at least one GPU device is available. | Hooks into compiler backend target flags (`CUDA`/`ROCm`). |
| [ ] | `assert_tpu_available` | `(backend=None)` | Checks that at least one TPU device is available. | Hooks into compiler backend target flags (`TPU`). |
| [x] | `assert_tree_is_on_device` | `(tree, platform, device=None)` | Checks leaves are in device memory (HBM). | The compiler's array objects must expose memory space locations. Must validate `HBM` or equivalent. |
| [x] | `assert_tree_is_on_host` | `(tree, allow_cpu_device=True, allow_sharded=False)`| Checks leaves are in host memory (CPU). | Must validate `DRAM` / CPU backing buffers. |
| [x] | `assert_tree_is_sharded` | `(tree, devices)` | Checks leaves are sharded across specified devices. | Arrays must expose a `sharding` property returning the equivalent of JAX's `NamedSharding` or `PositionalSharding`. |

### Table 4: JAX Manipulation, Context & Tracing
The compiler must expose deep internal APIs to manipulate how graphs are built.

| [x] | Name | Function Signature | Docstring | Compiler Implementation Notes |
|---|---|---|---|---|
| [ ] | `chexify` | `(fn, async_check=True, errors)` | Wraps a transformed function to enable Chex value assertions. | Generates the token-threaded `checkify` IR. Modifies function outputs to return an `(error_trace, output)` tuple. |
| [ ] | `block_until_chexify_assertions_complete` | `()` | Waits until all async checks complete. | Must flush the compiler's D2H (Device-to-Host) async queues to ensure pending assertions resolve before moving to the next Python line. |
| [x] | `assert_max_traces` | `(fn=None, n=None)` | Checks that a function is traced at most `n` times. | Requires the compiler frontend to maintain a global trace evaluation counter. |
| [x] | `assert_numerical_grads` | `(f, f_args, order, atol=0.01)` | Checks that autodiff and numerical gradients match. | Compiler must support both forward (`jvp`) and reverse (`vjp`) mode AD lowering. Will compile a subgraph calculating empirical limits. |
| [x] | `clear_trace_counter` | `()` | Clears Chex traces' counter. | Global thread-safe state reset for the frontend tracerb |
| [x] | `disable_asserts` | `()` | Disables all Chex assertions. | Sets a global context variable bypassing all graph node injections. |
| [x] | `enable_asserts` | `()` | Enables Chex assertions. | Restores global context variable. |
| [x] | `fake_jit` | `(enable_patching=True)` | Context manager for patching `jax.jit` with identity. | Patches the compiler's primary dispatch mechanism to run eager fallback code instead of XLA compilation. |
| [x] | `fake_pmap` | `(enable_patching=True, ...)` | Context manager for patching `jax.pmap` with `jax.vmap`. | Translates multi-replica SP/MD calls into single-replica vectorized loop operations. |
| [x] | `fake_pmap_and_jit` | `(enable_pmap=True, enable_jit=True)` | Patches both jit and pmap. | Combines patching operations safely. |
| [x] | `restrict_backends` | `(allowed=None, forbidden=None)` | Disallows compilation for certain backends. | Must inject a pre-compilation validation step that halts execution if the requested target ISA matches `forbidden`. |
| [ ] | `set_n_cpu_devices` | `(n: int)` | Forces XLA to use `n` CPU threads as host devices. | Must re-initialize the compiler runtime/PJRT client with a custom thread-pool topology. |

### Table 5: Scalar, Math & Logic Assertions

| [x] | Name | Function Signature | Docstring | Compiler Implementation Notes |
|---|---|---|---|---|
| [x] | `assert_scalar` | `(x: float \| int)` | Checks that `x` is a scalar. | Validates `x.ndim == 0` at trace time. |
| [ ] | `assert_scalar_in` | `(x, min_, max_, included=True)` | Checks that argument is a scalar within segment. | Lowers to `And(GreaterEqual(x, min), LessEqual(x, max))` IR nodes. |
| [ ] | `assert_scalar_negative` | `(x)` | Checks that a scalar is negative. | Lowers to `Less(x, 0)` IR node. |
| [ ] | `assert_scalar_non_negative`| `(x)` | Checks that a scalar is non-negative. | Lowers to `GreaterEqual(x, 0)` IR node. |
| [ ] | `assert_scalar_positive` | `(x)` | Checks that a scalar is positive. | Lowers to `Greater(x, 0)` IR node. |
| [x] | `assert_equal` | `(first, second)` | Checks that two objects are equal (`==`). | Trace-time logical check. |
| [x] | `assert_exactly_one_is_none`| `(first, second)` | Checks that one and only one argument is `None`. | Trace-time logical check. |
| [x] | `assert_not_both_none` | `(first, second)` | Checks that at least one argument is not `None`. | Trace-time logical check. |
| [ ] | `assert_is_broadcastable` | `(shape_a, shape_b)` | Checks that `shape_a` is broadcastable to `shape_b`. | Compiler frontend must implement NumPy-compatible broadcasting rule resolution. |
| [ ] | `assert_is_divisible` | `(numerator, denominator)` | Checks divisibility. | IR Lowering `Equal(Remainder(num, den), 0)`. |

### Table 6: Deprecations, Utilities, Classes & Types

| [x] | Name | Function Signature | Docstring | Compiler Implementation Notes |
|---|---|---|---|---|
| [ ] | `dataclass` | `(cls=None, ...)` | JAX-friendly wrapper for `dataclasses.dataclass`. | The compiler frontend must hook this directly into `PyTree` registration to support compiler tracing over nested classes. |
| [x] | `mappable_dataclass` | `(cls)` | Exposes dataclass as `collections.abc.Mapping`. | Same as above. |
| [x] | `params_product` | `(params_lists, named=False)`| Generates a cartesian product of params. | Utility. |
| [x] | `create_deprecated_function_alias` | `(fun, new_name, ...)`| Create a deprecated alias. | Utility. |
| [x] | `warn_deprecated_function` | `(fun, replacement=None)` | Decorator marking function deprecated. | Utility. |
| [x] | `warn_only_n_pos_args_in_future` | `(fun, n)` | Warns if > `n` positional args are passed. | Utility. |
| [x] | `if_args_not_none` | `(fn, args, kwargs)` | Wrap chex assertion to eval only if args not None. | Dynamic evaluation wrapper. |
| [ ] | `ArrayBatched` / `ArrayDevice` / `ArrayNumpy` / `ArraySharded` | `Type Aliases` | Assorted Array base classes. | Compiler type system must expose metadata to satisfy `isinstance` checks against these wrappers. |
| [x] | `ChexVariantType` | `Enum` | Enumeration of available Chex variants. | E.g., `with_jit`, `without_jit`. Handled natively. |
| [ ] | `Device` | `Class` | Descriptor of an available device. | Matches PJRT Device structure. |
| [x] | `Dimensions` | `Class` | Maps strings to shape tuples. | Utility class. |
| [ ] | `PyTreeDef` | `Class` | Represents a flattened tree structure. | C++ structure binding representation. |
| [x] | `TestCase` | `Class` | Chex tests that use variants. | Base testing framework for variant replay. |
### Table 7: Type Aliases, Utilities & Enums (Addendum)

| [x] | Name | Function Signature | Docstring | Compiler Implementation Notes |
|---|---|---|---|---|
| [ ] | `Array` | `Type Alias` | A generic type alias for an array. | The compiler type system must resolve this to the native backend tensor type. |
| [x] | `ArrayBatched` | `Type Alias` | A batched array. | Supported natively by tracing. |
| [x] | `ArrayDevice` | `Type Alias` | An array placed on a device. | Supported natively by tracing. |
| [x] | `ArrayDeviceTree` | `Type Alias` | A PyTree of device arrays. | Supported natively by flattening logic. |
| [x] | `ArrayDType` | `Type Alias` | Valid DType for arrays. | Checked against supported frontend precision (e.g. F32, BF16). |
| [x] | `ArrayNumpy` | `Type Alias` | A NumPy ndarray. | Natively castable if encountered at trace edges. |
| [x] | `ArrayNumpyTree` | `Type Alias` | A PyTree of NumPy arrays. | Flattening support for host-based tracing. |
| [x] | `ArraySharded` | `Type Alias` | A sharded array. | Handled via GSPMD tracing attributes. |
| [x] | `ArrayTree` | `Type Alias` | A PyTree of arrays. | Flattening support. |
| [x] | `ChexifyChecks` | `Enum` | Enum config for Chexify runtime. | Maps to checking modes (user, internal). |
| [x] | `Numeric` | `Type Alias` | Scalar numeric value. | Native type mapping. |
| [x] | `PRNGKey` | `Type Alias` | A PRNG key array. | Mapped to integer bitwise Arrays. |
| [x] | `Scalar` | `Type Alias` | Python scalar. | Native type mapping. |
| [x] | `Shape` | `Type Alias` | A tuple representing a shape. | Handled dynamically or via Dimension inference. |
| [x] | `all_variants` | `(with_pmap=True, ...)` | Decorator that wraps test in all variants. | Test runner generator. |
| [x] | `get_err_regex` | `(err)` | Utility to get exception regex. | Test utility function. |
| [ ] | `register_dataclass_type_with_jax_tree_util` | `(dataclass_type)` | Registers a type to tree_util. | Direct alias to C++ backend registration method. |
| [x] | `variants` | `(variants=(), ...)` | Decorator that tests specified variants. | Test runner generator. |
| [x] | `warn_keyword_args_only_in_future` | `(fun)` | Deprecation warning for kwarg usage. | Test utility function. |
