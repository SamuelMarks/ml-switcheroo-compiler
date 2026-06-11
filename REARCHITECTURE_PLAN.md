# Rearchitecture Plan: Modularity for the `ml-switcheroo-compiler`

To effectively scale the compiler to fulfill the massive `*TODO.md` backlog without degrading code quality, a major rearchitecture is necessary. The current state relies on large monolithic files (e.g., `jnp.py` ~2000 lines) and hardcoded `if/elif` blocks inside code generators (`backends/`).

## Core Architectural Principles

1. **Open-Closed Principle (OCP)**: Adding a new mathematical operation (e.g., `sin`) should involve adding code in **one place**, not updating 10 different files (interpreter, 5 backends, shape inferencer, tracer).
2. **Package-over-Module Pattern**: Always prefer the `module/__init__.py` structure over a single `module.py`. Logic must be aggressively decomposed into semantically separate files within a directory, with the public API exposed entirely through the directory's `__init__.py` using explicit `__all__` declarations.
3. **Strict Separation of Concerns**: Frontends delegate, the Compiler traces & optimizes, and Backends emit. 
4. **Pluggable Architecture**: Passes, diagnostics, and operations must be dynamically discoverable or explicitly registered in modular directories.

---

## 1. Unified Operation Registry (`ml_switcheroo/ops/`)

The biggest bottleneck is mapping `N` ops to `M` backends. We must abandon `if node.op_type == "Add":` logic in favor of an **Op Registry**. Operations will self-report their properties.

### Action Plan & Checklist:
- [ ] Create `ml_switcheroo/ops/base.py` defining the `OpDef` base class and the `@register_op` global decorator.
- [ ] Define the `infer_shape(self, *args, **kwargs)` abstract method contract on `OpDef`.
- [ ] Define the `numpy_eval(self, *args, **kwargs)` abstract method contract for the Eager fallback.
- [ ] Define the AD integration methods: `vjp(self, cotangent, *args)` and `jvp(self, tangent, *args)`.
- [ ] Define abstract backend emitter templates: `emit_jax`, `emit_pytorch`, `emit_mlx`, `emit_keras`, `emit_tensorflow`.
- [ ] Scaffold `ops/unary/__init__.py` and migrate all single-operand math (e.g., `sin`, `cos`, `exp`, `log`).
- [ ] Scaffold `ops/binary/__init__.py` and migrate two-operand math (e.g., `add`, `mul`, `pow`, `maximum`).
- [ ] Scaffold `ops/linalg/__init__.py` and migrate linear algebra primitives (e.g., `matmul`, `dot`, `einsum`).
- [ ] Scaffold `ops/creation/__init__.py` and migrate tensor instantiation (e.g., `zeros`, `ones`, `arange`, `full`).
- [ ] Scaffold `ops/reductions/__init__.py` and migrate axis-reducing ops (e.g., `sum`, `mean`, `max`, `min`).
- [ ] Scaffold `ops/shape/__init__.py` and migrate structural ops (e.g., `reshape`, `transpose`, `broadcast_to`).
- [ ] Update `ops/__init__.py` to automatically load and expose all registered operations.

## 2. Backend De-duplication (`ml_switcheroo/backends/`)

Currently, `JAXCodeGenerator`, `PyTorchCodeGenerator`, etc., all have their own duplicate tree-walking loops, varying wildly in quality and completeness.

### Action Plan & Checklist:
- [ ] Create `backends/base_generator.py` with the abstract `BaseGenerator` class.
- [ ] Implement `IRGraph` topological sorting strictly within `BaseGenerator`.
- [ ] Implement robust, collision-free deterministic variable naming (`var_1`, `var_2`) in `BaseGenerator`.
- [ ] Implement the main execution loop in `BaseGenerator.generate()` that iterates over nodes and queries the `OpRegistry` for the specific backend string templates.
- [ ] Refactor `backends/jax.py` to inherit from `BaseGenerator` and only define JAX header/footer boilerplate.
- [ ] Refactor `backends/pytorch.py` to inherit from `BaseGenerator` and define PyTorch `nn.Module` routing.
- [ ] Refactor `backends/keras.py` to inherit from `BaseGenerator` and define functional Keras API routing.
- [ ] Refactor `backends/mlx.py` to inherit from `BaseGenerator`.
- [ ] Refactor `backends/tensorflow.py` to inherit from `BaseGenerator`.
- [ ] Create `backends/__init__.py` to export all refactored generator classes.

## 3. De-monolithing the Frontend Shims (`jnp.py` -> `jnp/`)

`jnp.py` is nearly 2000 lines long. While it serves as the frontend shim for `zero-jax`, it is unwieldy and violates the Package-over-Module rule.

### Action Plan & Checklist:
- [ ] Create `ml_switcheroo/jnp/` directory and touch `__init__.py`.
- [ ] Extract the core `ndarray` wrapper class definition into `jnp/array.py`.
- [ ] Migrate standard arithmetic magic methods (`__add__`, `__mul__`, `__rmul__`) to `jnp/math_ops.py`.
- [ ] Migrate reshaping, transposing, and slicing logic to `jnp/manipulation.py`.
- [ ] Migrate dot products, matrix multiplications, and tensor dot wrappers to `jnp/linalg.py`.
- [ ] Migrate array creation methods (`jnp.zeros`, `jnp.arange`) to `jnp/creation.py`.
- [ ] Migrate random number generation bindings to `jnp/random.py`.
- [ ] Configure `jnp/__init__.py` with a strict `__all__` export list to guarantee 1:1 API compatibility with the original monolithic `jnp.py` and official JAX.

## 4. Pass Manager for Middle-End (`ml_switcheroo/transforms/`)

`optimization.py` has stubbed transformations. IR optimization must be modularized into a pipeline architecture.

### Action Plan & Checklist:
- [ ] Create `transforms/pass_manager.py` with a `PassManager` orchestrator class.
- [ ] Implement fixpoint iteration logic in `PassManager.run_until_converged()` to re-run passes until the IR stabilizes.
- [ ] Convert `transforms/passes/dce.py` (Dead Code Elimination) to safely prune unconsumed nodes.
- [ ] Convert `transforms/passes/cse.py` (Common Subexpression Elimination) to merge duplicate sub-graphs.
- [ ] Migrate `constant_folding` from `optimization.py` to `transforms/passes/constant_folding.py`.
- [ ] Implement `transforms/passes/shape_inference_pass.py` to statically resolve symbolic shapes and catch `ShapeMismatchError` pre-backend.
- [ ] Implement `transforms/passes/lift_state.py` to extract implicit state (Flax/PyTorch variables) into explicit functional I/O.
- [ ] Ensure `transforms/__init__.py` exports the `PassManager` and default pipeline configs.

## 5. Removing Legacy Stubs (`plugins_legacy.py`)

The codebase contains legacy reimplementation attempts that bloat the compiler core. The compiler should only understand math, not specific neural network architectures.

### Action Plan & Checklist:
- [ ] Delete `plugins_legacy.py` entirely.
- [ ] Remove the 37 empty classes (e.g., `AttentionPacking`, `BatchNormPlugin`, `AutoFSDPWrapper`).
- [ ] Scrub the codebase of any imports referencing `plugins_legacy`.
- [ ] Document in `ARCHITECTURE.md` that all high-level NN layers must be decomposed into core ops by the `zero-*` frontend libraries, not handled as opaque plugins in the compiler.

## 6. Modular Diagnostics (`ml_switcheroo/diagnostics/`)

The current `diagnostics.py` contains mostly dummy string/integer returns.

### Action Plan & Checklist:
- [ ] Remove the original `diagnostics.py` file.
- [ ] Create `diagnostics/` package with `__init__.py`.
- [ ] Implement `diagnostics/flop_counter.py` that queries `OpDef` classes for their asymptotic complexity based on input shapes.
- [ ] Implement `diagnostics/memory_profiler.py` to track peak tensor allocations and buffer sizes during Eager execution.
- [ ] Implement `diagnostics/shape_debugger.py` to generate rich string outputs or DOT graphs of the IR with fully resolved shape annotations.
- [ ] Implement `diagnostics/numerical_anomaly.py` to trace `NaN` and `Inf` cascades during Eager mode execution.

## 7. Modular Interpreter (`ml_switcheroo/interpreter/`)

The interpreter currently falls back to `NotImplementedError` for most operations due to a massive `if/else` block.

### Action Plan & Checklist:
- [ ] Remove the original `interpreter.py` file.
- [ ] Create `interpreter/` package with `__init__.py`.
- [ ] Implement `interpreter/evaluator.py` to perform the main traversal over the topologically sorted IR graph.
- [ ] Implement `interpreter/environment.py` to manage tensor memory mappings, proxy state, and thread-local scoping during evaluation.
- [ ] Refactor the execution logic in `evaluator.py` to dynamically dispatch execution to the `numpy_eval` methods defined in the `OpRegistry`, completely eliminating the `if node.op_type == ...` blocks.

## 8. Strict Backend Isolation & Leak Detection

A critical risk in a multi-backend compiler is dependency leaking. `numpy` must not leak into a non-numpy backend, and `jax` must not leak into the PyTorch generator.

### Action Plan & Checklist:
- [ ] **Define Boundaries**: Establish rule that `ir/`, `ops/`, and `transforms/` are completely framework-agnostic.
- [ ] **Restrict Numpy**: Ensure `numpy` is strictly confined to `numpy_backend/`, the `numpy_eval` methods in the OpRegistry, and `interpreter/`. It must never leak into IR generation.
- [ ] **Isolate Backends**: Ensure `import jax` is strictly confined to `backends/jax.py` (and eventually `backends/jax/`). Prevent `import torch` from existing outside of `backends/pytorch/`.
- [ ] **Build `detect_leaks.py`**: Create a custom static analysis script in `scripts/detect_leaks.py`.
- [ ] **AST Parsing**: Implement Python `ast.NodeVisitor` to parse `ast.Import` and `ast.ImportFrom` nodes across all `src/` files.
- [ ] **Rule Config**: Define an internal mapping of forbidden imports per directory.
- [ ] **CI Enforcement**: Configure `detect_leaks.py` to return an exit code `1` and fail the build if a boundary is violated.

## 9. Linting & Pre-commit Hooks

To enforce the new architectural principles, OCP, and leak detection rules automatically, robust pre-commit tooling is strictly required.

### Action Plan & Checklist:
- [ ] Create `.pre-commit-config.yaml` at the root of the workspace.
- [ ] Add `check-yaml`, `trailing-whitespace`, and `end-of-file-fixer` from `pre-commit/pre-commit-hooks`.
- [ ] Add `ruff` linter hook targeting all `src/` directories to maintain code quality and fast import sorting.
- [ ] Add `ruff-format` hook to enforce PEP8 and standard spacing uniformly.
- [ ] Add `mypy` hook configured with `--strict`, `--disallow-untyped-defs`, and `--warn-return-any` to guarantee robust type safety across dynamic OpRegistry bindings.
- [ ] Integrate the custom AST-based leak detector (`scripts/detect_leaks.py`) as a `language: system` local pre-commit hook so developers catch framework pollution locally *before* code is pushed.
- [ ] Verify all checks pass locally via `pre-commit run --all-files`.
