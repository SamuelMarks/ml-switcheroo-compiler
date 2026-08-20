# ML Switcheroo Compiler: Exhaustive Implementation Gap Plan

This document outlines the exhaustive plan to resolve discrepancies between claimed architectural features and actual implementation within `ml-switcheroo-compiler`, strictly adhering to the architectural separation rules.

**Quality Metrics Enforcement:**
- [x] Ensure 100% doc coverage (arg, function, class, file, module) for all new and modified code.
- [x] Ensure 100% test coverage (function, line, branch) for all new and modified code.
- [x] Enforce strong typing (no `Any` or `ignore` unless strictly necessary for polymorphic/duck typing framework agnosticism, and only where currently justified).
- [x] **Dependency Rule:** OUTSIDE OF TESTS, NO use of any 3rd party dependency (exceptions: pyyaml, pillow, pydantic, ./cdd-python, h5py) OUTSIDE of a backend-specific directory (where one additional dependency is allowed).
- [x] Ensure all precommit hooks pass before marking any item as complete.

---

## 1. Edge Backend Kernel Completeness (WebGPU/WASM/C++)

**Problem:** Documentation claims "Direct-to-Edge Compilation" translating the graph into WGSL and WASM SIMD for edge execution. However, most operations (thousands defined in schema) trigger `UnimplementedMathError` because hardware templates (e.g., `wgsl_templates.yaml`, `wasm_templates.yaml`, `cpp_templates.yaml`) are missing or heavily stubbed.

**Action Plan:**

### 1.1 WebGPU (WGSL) Backend Completeness
- [x] Audit `src/ml_switcheroo_compiler/backends/edge/wgsl/wgsl_templates.yaml` against all ops defined in `src/ml_switcheroo_compiler/ops/definitions/`.
- [x] Implement missing WGSL kernel templates for all fundamental math operations (Trigonometric, Exponential, Logarithmic).
- [x] Implement missing WGSL kernel templates for all reduction operations (Sum, Prod, Min, Max, Mean, Variance, etc.).
- [x] Implement missing WGSL kernel templates for advanced neural network operations (Activations, Norms, Advanced Convolutions).
- [x] Implement missing WGSL kernel templates for indexing, slicing, and tensor manipulation operations.
- [x] Add comprehensive tests in `tests/backends/edge/` verifying WGSL template generation for all newly supported operations.

### 1.2 WASM SIMD Backend Completeness
- [x] Audit `src/ml_switcheroo_compiler/backends/edge/wasm_simd/wasm_templates.yaml` against all ops defined in `src/ml_switcheroo_compiler/ops/definitions/`.
- [x] Implement missing WASM SIMD kernel templates for fundamental math operations.
- [x] Implement missing WASM SIMD kernel templates for reduction operations.
- [x] Implement missing WASM SIMD kernel templates for advanced neural network operations.
- [x] Implement missing WASM SIMD kernel templates for tensor manipulation operations.
- [x] Add comprehensive tests in `tests/backends/edge/` verifying WASM template generation for all newly supported operations.

### 1.3 LLVM/C++ Backend Completeness
- [x] Audit `src/ml_switcheroo_compiler/backends/llvm_cpp/cpp_templates.yaml` against all ops defined in `src/ml_switcheroo_compiler/ops/definitions/`.
- [x] Implement missing C++ kernel templates for fundamental math operations.
- [x] Implement missing C++ kernel templates for reduction operations.
- [x] Implement missing C++ kernel templates for advanced neural network operations.
- [x] Implement missing C++ kernel templates for tensor manipulation operations.
- [x] Add comprehensive tests in `tests/backends/llvm_cpp/` verifying C++ template generation for all newly supported operations.

---

## 2. Memory-Efficient Checkpointing & Gradient Flow

**Problem:** The README claims "memory-efficient checkpointing/rematerialization and custom gradient hooks." However, the VJP rule for `Checkpoint` is hardcoded to `ZerosLike($cotangent)`, meaning gradients do not actually flow back (they are zeroed out). Additionally, there is no custom gradient hook mechanism.

**Action Plan:**

### 2.1 Rematerialization (Checkpointing) VJP Implementation
- [x] Locate `Checkpoint` in `src/ml_switcheroo_compiler/transforms/autodiff_rules/autodiff_rules.yaml`.
- [x] Replace the `ZerosLike($cotangent)` VJP rule for `Checkpoint` with logic that correctly re-evaluates the forward subgraph (rematerialization) during the backward pass and calculates the accurate vector-Jacobian product.
- [x] Modify `src/ml_switcheroo_compiler/transforms/autodiff.py` or related AD engine files to support executing a subgraph's VJP dynamically during the backward pass traversal when a `Checkpoint` node is encountered.
- [x] Create rigorous tests in `tests/grad/test_checkpointing_coverage.py` that mathematically verify gradient correctness when using `checkpoint()` (comparing against a non-checkpointed version of the same graph).

### 2.2 Custom Gradient Hooks Mechanism
- [x] Design and implement a universal custom gradient hook registry/mechanism within the core autodiff engine (`src/ml_switcheroo_compiler/transforms/autodiff.py` or `src/ml_switcheroo_compiler/grad/api.py`). This must adhere to the "No API Shell" rule (it must be a backend-agnostic core feature).
- [x] Ensure the mechanism allows users (or Tier 3 frontend shells) to register a callback function that is invoked during the backward pass execution (or trace) for a specific tensor or node.
- [x] Create tests in `tests/grad/test_autodiff.py` verifying that custom gradient hooks are successfully registered, invoked with the correct cotangents, and can correctly modify the gradient flow during AD.

---

## 3. Distributed Pipeline Parallelism

**Problem:** The compiler claims "Distributed Parity... and pipeline parallelism primitives." However, the `PipelineParallelismStrategy` class methods are skeletal, heavily relying on `pass` or bypass logic. Actual graph partitioning and distributed collective orchestration are incomplete.

**Action Plan:**

### 3.1 Implement Pipeline Graph Partitioning
- [x] Replace the `pass` stubs in `src/ml_switcheroo_compiler/distributed/strategy.py` within the `PipelineParallelismStrategy` class (specifically `unroll_pipeline` and `split_pipeline`).
- [x] Implement algorithm to statically partition an `IRGraph` into `N` stages based on the provided pipeline topology configuration.
- [x] Inject necessary communication primitives (e.g., `Send`, `Recv`) between the partitioned subgraphs to handle inter-stage data dependencies.
- [x] Create tests in `tests/distributed/test_pipeline_primitives.py` to verify the correctness of the generated partitioned graphs and the injected communication nodes.

### 3.2 Implement 1F1B Schedule Unrolling
- [x] Implement the 1F1B (One Forward, One Backward) scheduling logic within `unroll_pipeline` in `src/ml_switcheroo_compiler/distributed/strategy.py`.
- [x] Ensure the schedule correctly interleaves forward and backward execution phases to minimize pipeline bubbles.
- [x] Create tests verifying the topological ordering and scheduling of the 1F1B unrolled graph.

### 3.3 Fix TCP Socket Mockups & Fallbacks
- [x] Audit `src/ml_switcheroo_compiler/distributed/strategy.py` and `src/ml_switcheroo_compiler/backends/numpy/eager/distributed.py` for `pass` statements related to socket bindings, server starts, and joins.
- [x] Implement robust error handling and proper socket lifecycle management in the distributed strategy context manager.
- [x] Implement concrete distributed fallback logic in backend eager modes (e.g., replace the `pass` in `mlx.core.distributed.recv` fallback in `src/ml_switcheroo_compiler/backends/mlx/eager.py`).

---

## 4. Architectural Adherence: Polyfills vs. Universal IR (The "No Math in Frontends" Rule)

**Problem:** Complex neural network operations (e.g., CTC Decoders, Isotonic Regression, specialized ConvBackprops) are currently implemented in `src/ml_switcheroo_compiler/backends/numpy/eager/nn_polyfills.py` using NumPy/SciPy eager loops. This violates the claim that the compiler seamlessly translates *all* logic via the Universal IR. These polyfills cannot be cross-compiled to edge targets.

**Action Plan:**

### 4.1 Lower Polyfills to Universal IR
- [x] Identify all operations currently defined in `src/ml_switcheroo_compiler/backends/numpy/eager/nn_polyfills.py` (e.g., `IsotonicRegression`, `ConvTranspose`, `DepthwiseConv2dBackpropFilter`, `InTopK`, `CtcBeamSearchDecoder`, `QuantizedConv`).
- [x] For each identified operation, implement a lowering pass or rewrite rule in `src/ml_switcheroo_compiler/transforms/passes/` or define a composite operation in `src/ml_switcheroo_compiler/ops/` that decomposes the high-level operation into fundamental, universally supported IR primitives (e.g., standard map, scan, while loops, basic arithmetic).
- [x] Verify that these decomposed operations can be successfully traced, optimized, and emitted by all major backends (including Edge WGSL/WASM).

### 4.2 Deprecate and Remove `nn_polyfills.py`
- [x] Once all polyfilled operations are successfully lowered into the Universal IR, remove their implementations from `nn_polyfills.py`.
- [x] Remove `scipy` dependencies from the numpy eager backend where they were only used for polyfills (if applicable, though scipy is allowed, relying on it for core ML compiler logic prevents edge emission).
- [x] Create tests ensuring that the previously polyfilled operations now execute correctly across multiple backends (e.g., JAX, NumPy, and WASM) with identical numerical results, proving they are genuinely compiled.
