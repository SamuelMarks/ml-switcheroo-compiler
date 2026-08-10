# ML Switcheroo Compiler: Exhaustive TODO & Remediation Plan

This document outlines the exhaustive task list required to bridge the gaps in the `ml-switcheroo-compiler`'s implementation, while strictly enforcing the project's rigorous quality metrics and architectural constraints.

## 1. Quality Metrics & Ecosystem Constraints (Global Requirements)
These constraints MUST be applied across all subsequent phases and code modifications.

### 1.1 Strict Dependency & Decoupling Rules
- [x] **Core Dependency Audit:** Verify that `src/ml_switcheroo_compiler/` (outside of `backends/`) uses NO 3rd-party dependencies other than `pydantic`, `./cdd-python`, and `h5py`.
- [x] **Backend Dependency Isolation:** Verify that each backend directory (e.g., `backends/mlx/`, `backends/jax/`, `backends/pytorch/`) contains exactly ONE primary 3rd-party dependency.
- [x] **No Upward Calls:** Ensure ZERO imports from any `zero-*` frontend repository or `zero-zoo`. Circular dependencies break the compilation pipeline.
- [x] **Backend Decoupling:** Ensure the tracing engine, IR graph builder (`core/`, `tracing/`, `ir/`), and middle-end (`transforms/`) NEVER import backend-specific execution libraries (e.g., `import torch`, `import mlx`).
- [x] **Frontend Independence:** Strip any framework-specific syntax mimicry (e.g., `nn.Module` logic, frontend exceptions, `dim` vs `axis` aliases) from the compiler.
- [x] **N-to-M Universal Utility:** Ensure every new operation in `ops/` and every pass in `transforms/` can be lowered to the IR and emitted by at least two distinct backends.
- [x] **Mathematical Centralization:** Ensure all math, Automatic Differentiation (AD), forward/backward passes, and JVPs/VJPs are exclusively implemented in the compiler, never delegated to a frontend shell.

### 1.2 Documentation Coverage (Target: 100%)
- [x] **Module-Level Docs:** Add comprehensive docstrings to all `.py` files.
- [x] **Class-Level Docs:** Document the purpose and architecture of every class.
- [x] **Function/Method-Level Docs:** Document every function and method.
- [x] **Arg/Return Docs:** Ensure all docstrings specify types, arguments, and return values that perfectly match the actual code signatures.

### 1.3 Test Coverage (Target: 100%)
- [x] **Function Coverage:** Ensure 100% of functions are executed by the test suite.
- [x] **Line Coverage:** Ensure 100% of lines are executed.
- [x] **Branch Coverage:** Ensure 100% of logical branches (if/else, loops) are tested.
- [x] **Test Realism:** Remove all dummy `try...except Exception: pass` tests and replace them with strict mathematical assertions comparing eager execution vs compiled graphs.

### 1.4 Code Quality & Typing
- [x] **Strong Typing (100%):** Eliminate all implicit `Any` types; enforce strict typing using `Generic`, `TypeVar`, `Callable`, and explicit return types.
- [x] **Pre-commit Hooks:** Ensure `black`, `isort`, `flake8`/`pylint`, and `mypy` pass locally and in CI without warnings.

---

## 2. Phase 1: True Multi-Node Distributed Collectives
Currently, `strategy.py` uses local `threading` and mocked barriers. This must be replaced with real networking.
- [x] **Remove Mocked Context:** Strip out `MockDistributedContext`, `threading.Barrier`, and `threading.Lock` from `src/ml_switcheroo_compiler/backends/numpy/eager/distributed.py` and `strategy.py`.
- [x] **Implement Real IPC/RPC Layer:** Build a purely standard-library based network topology using `socket`, `asyncio`, and/or `multiprocessing` (respecting the strict 3rd-party dependency rules).
- [x] **Implement Networked AllReduce:** Implement a ring-allreduce or parameter-server `AllReduce` over TCP/IP sockets.
- [x] **Implement Networked AllGather:** Implement an `AllGather` primitive across actual socket connections.
- [x] **Implement Networked ReduceScatter:** Implement a `ReduceScatter` primitive.
- [x] **Implement Networked AllToAll:** Implement an `AllToAll` primitive.
- [x] **Distributed Test Harness:** Create tests in `tests/distributed/` that spawn multiple sub-processes to mathematically verify multi-node equivalence against a single-node eager baseline.

## 3. Phase 2: C++ LLVM Backend - NDArray & BLAS Integration
The current `CppGenerator` generates naive, nested loops. It needs strided views and optimized math.
- [x] **C++ NDArray View Struct:** Implement a lightweight, header-only C++ tensor view struct (`backends/llvm_cpp/`) that handles strides, shapes, offsets, and multi-dimensional indexing.
- [x] **C++ Broadcasting Semantics:** Implement dynamic broadcasting logic within the C++ NDArray view for binary operations.
- [x] **Standard BLAS/LAPACK Integration:** Add standard C++ compiler flags and conditionally compiled code paths to link against generic BLAS libraries for `MatMul` and `DotGeneral`.
- [x] **Optimized Tiled Loops:** For systems without BLAS, replace the current naive C++ loops with cache-aware, blocked/tiled matrix multiplication loops.
- [x] **C++ E2E Test Pipeline:** Update `tests/backends/llvm_cpp/` to compile the emitted C++ strings using `clang++`/`g++` in a subprocess and mathematically execute them against baseline outputs.

## 4. Phase 3: Advanced Edge Kernels (WebGPU & WASM Optimization)
- [x] **WebGPU Workgroup Tiling:** Refactor `_get_wgsl_for_op` in `webgpu.py` to use shared memory (workgroup memory) caching for `MatMul` and `Conv2D` to eliminate redundant VRAM reads.
- [x] **WebGPU Kernel Fusion:** Implement a specialized WGSL emitter pass that fuses consecutive elementwise operations (e.g., `Add` + `Relu`) into a single shader dispatch, reducing pipeline overhead.
- [x] **WASM Advanced Loop Unrolling:** Enhance `wasm.py` to aggressively unroll loops when array sizes are known statically at compile time.
- [x] **WASM Native v128 Fallbacks:** Replace remaining `std::` scalar peel loops for complex math (e.g., `Tanh`, `Sigmoid`, `LogSoftmax`) with fast polynomial approximations using native v128 WASM intrinsics.
- [x] **Edge Execution Test Coverage:** Ensure `test_webgpu.py` and `test_wasm.py` mathematically validate the outputs of the optimized kernels instead of just testing AST/string substrings.

## 5. Phase 4: Middle-End IR Checkpointing & Rematerialization Passes
- [x] **Scaffold Rematerialization Pass:** Create `src/ml_switcheroo_compiler/transforms/passes/rematerialization.py`.
- [x] **Implement Memory Cost Model:** Build a heuristic function to estimate the memory footprint of intermediate IR nodes.
- [x] **Implement Compute Cost Model:** Build a heuristic function to estimate the FLOPs/latency of recomputing IR nodes.
- [x] **Graph Rewriting for Recomputation:** Implement the pass logic to drop high-memory/low-compute nodes during the forward pass and inject `Recompute` logic in the backward pass graph.
- [x] **Pass Manager Integration:** Register the rematerialization pass in `pass_manager.py` to run seamlessly before `autodiff` graph expansion.
- [x] **IR State Validation:** Ensure the `Recompute` and `Checkpoint` IR nodes adhere to the Universal Utility rule and can be successfully emitted by all backends.

## 6. Phase 5: Mathematical Verification of Higher-Order AD
- [x] **Audit `grad.py` Tests:** Identify all stubbed tests (e.g., `test_hessian` containing only `pass`) in `tests/grad/test_grad_hvp.py` and `tests/grad/test_grad_missing3.py`.
- [x] **Implement Analytical Baselines:** Write hardcoded, known analytical gradients for the test functions to use as sources of truth.
- [x] **Test HVP (Hessian-Vector Product):** Implement concrete assertions for `hvp` on polynomials and neural network primitives.
- [x] **Test `jacfwd` (Forward-Mode Jacobian):** Implement concrete assertions for full Jacobian matrix extraction.
- [x] **Test `jacrev` (Reverse-Mode Jacobian):** Implement concrete assertions and mathematically verify equivalence between `jacfwd` and `jacrev`.
- [x] **Test `hessian`:** Implement mathematical assertions for the full Hessian matrix computation.
- [x] **Edge Case Coverage:** Add tests for higher-order AD through control flow (`WhileLoop`, `Cond`) to ensure 100% branch coverage in the gradient tracing logic.
